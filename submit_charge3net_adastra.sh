#!/bin/bash
# ChargE3Net fine-tuning on Adastra (CINES, AMD MI250X), half-node DDP.
# See ADASTRA.md for setup details and known gotchas.
#
# Half-node resource layout (g1xxx mi250-shared has 8 GCDs, 128 CPUs, 256 GB):
#   - 4 GCDs (gpus-per-node=4)
#   - 64 CPUs (16 per task * 4 tasks)
#   - 128 GB RAM
#   - 4 tasks, one per GCD, for torch DistributedDataParallel
#
# Effective batch = batch-size * world_size = 16 * 4 = 64 (matches the
# upstream paper's train_mp_e3_final.yaml: batch_size=16, nnodes=2 x nprocs=2).
#SBATCH --job-name=charge3net_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=c1816212
#SBATCH --constraint=MI250
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=125000M
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

# --- Paths ---
# Submit dir must be on a scratch with inode headroom (cad16353 currently); the
# account (--account=c1816212 above) handles billing independently. See ADASTRA.md.
SETUP="${LEMATRHO_ADASTRA_SETUP:-/lus/scratch/CT10/cad16353/msiron/charge3net_setup}"
WORK_DIR="$SETUP/LeMat-Rho"
DATA_DIR="$SETUP/charge3net_data"
CKPT_DIR="$SETUP/charge3net_checkpoints"
MP_CKPT="$SETUP/charge3net/models/charge3net_mp.pt"

mkdir -p "$CKPT_DIR"

# --- Environment ---
# Proxy is required for any outbound HTTP (pip, HF, W&B). Already in ~/.bashrc
# on Adastra but we re-export here so the job script is self contained.
export HTTP_PROXY=http://proxy-l-adastra.cines.fr:3128
export HTTPS_PROXY=$HTTP_PROXY
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTP_PROXY

source "$SETUP/venv311/bin/activate"

export PYTHONPATH="$WORK_DIR:$SETUP/charge3net:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Load W&B key from .env if present.
if [ -f "$WORK_DIR/.env" ]; then
    set -a
    source "$WORK_DIR/.env"
    set +a
fi

# --- Distributed-training env vars (read by train.py's _setup_ddp) ---
# SLURM sets SLURM_NTASKS, SLURM_PROCID, SLURM_LOCALID for us via srun.
# torch.distributed wants WORLD_SIZE / RANK / LOCAL_RANK plus MASTER_ADDR
# / MASTER_PORT. We export them once here, srun propagates to each task.
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
# RANK / LOCAL_RANK are per-task — set in the wrapper srun command below.

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Job dir: $WORK_DIR"
echo "WORLD_SIZE=$WORLD_SIZE  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
rocm-smi || true

python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
"

cd "$WORK_DIR"

# --- Train ---
# Auto-resume from latest.pt if present, otherwise start from the pretrained
# Materials Project checkpoint (charge3net_mp.pt).
RESUME_FLAG=""
if [ -f "$CKPT_DIR/latest.pt" ]; then
    RESUME_FLAG="--resume-from $CKPT_DIR/latest.pt"
    echo "Resuming from $CKPT_DIR/latest.pt"
fi

# --- Knobs vs Jean Zay (NVIDIA A100) ---
# - batch-size: 16 per GPU (vs Jean Zay's 4 per GPU). MI250X has 64 GB HBM2e
#   per GCD; this matches the paper's per-GPU batch.
# - DDP across 4 GCDs gives effective batch = 64 (also matches the paper).
# - val-probes: 1000 to match paper validation granularity.
# - wandb-mode: offline. Adastra compute nodes can reach api.wandb.ai
#   intermittently through the proxy; previous job 4969727 timed out for
#   1h47m before crashing. The train.py wandb.init is now wrapped in
#   try/except so even an offline-mode failure degrades gracefully —
#   training continues with wandb disabled. Use `wandb sync wandb/`
#   from a login node afterwards to push the offline run.
#
# srun launches 4 tasks (--ntasks-per-node=4 from #SBATCH). Each task sees
# SLURM_PROCID = global rank, SLURM_LOCALID = local rank within node.
srun --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    # Each task sees ALL 4 GCDs the job was allocated; torch.cuda.set_device(local_rank)
    # inside _setup_ddp picks the right one. Restricting visibility per-task here
    # would make every task target the same "GCD 0" within its own visibility set.
    echo "task RANK=$RANK LOCAL_RANK=$LOCAL_RANK on $(hostname) (will use cuda:$LOCAL_RANK)"
    python3 -m charge3net_ft.train \
        --parquet-dir "'"$DATA_DIR"'" \
        --ckpt-path "'"$MP_CKPT"'" \
        --save-dir "'"$CKPT_DIR"'" \
        --epochs 50 \
        --batch-size 16 \
        --lr 5e-4 \
        --train-probes 200 \
        --val-probes 1000 \
        --num-workers 8 \
        --wandb-project lemat-rho-charge3net \
        --wandb-entity dtts \
        --wandb-mode offline \
        '"$RESUME_FLAG"'
'

echo "Done. Exit code: $?"
