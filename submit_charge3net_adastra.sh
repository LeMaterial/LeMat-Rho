#!/bin/bash
# ChargE3Net fine-tuning on Adastra (CINES, AMD MI250X), half-node DDP.
#
# Two training modes (select via LEMATRHO_TRAINING_MODE env):
#   pretrained   (default) — fine-tune from charge3net_mp.pt (MP, 245 epochs)
#   from_scratch          — train from random init for direct comparison
#
# Env vars:
#   LEMATRHO_TRAINING_MODE   pretrained | from_scratch     (default: pretrained)
#   LEMATRHO_ADASTRA_SETUP   override $SETUP                (default: /lus/scratch/CT10/cad16353/msiron/charge3net_setup)
#   LEMATRHO_DRY_RUN         1 to print the resolved train command and exit
#                            (used by tests/test_submit_script.py)
#
# Submit examples:
#   sbatch submit_charge3net_adastra.sh                                                       # pretrained
#   sbatch --export=ALL,LEMATRHO_TRAINING_MODE=from_scratch submit_charge3net_adastra.sh      # from-scratch
#
# Half-node resource layout (g1xxx mi250-shared has 8 GCDs, 128 CPUs, 256 GB):
#   - 4 GCDs (gpus-per-node=4)
#   - 64 CPUs (16 per task * 4 tasks)
#   - 128 GB RAM
#   - 4 tasks, one per GCD, for torch DistributedDataParallel
# Effective batch = batch-size * world_size = 16 * 4 = 64 (matches the
# upstream paper's train_mp_e3_final.yaml: batch_size=16, nnodes=2 x nprocs=2).
#SBATCH --job-name=charge3net_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=c1816212
#SBATCH --constraint=MI250
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
# No --mem here on purpose: SLURM allocates memory proportional to our CPU
# share (64 of 128 logical CPUs = ~128 GB out of the 256 GB node). The
# earlier --mem=125000M was being read as "asking for half the node memory"
# and contributed to SLURM auto-bumping us to EXCLUSIVE mode. Letting SLURM
# pick lets the other half of the node stay schedulable for other jobs.
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
MP_CKPT="$SETUP/charge3net/models/charge3net_mp.pt"

# --- Training mode -----------------------------------------------------------
TRAINING_MODE="${LEMATRHO_TRAINING_MODE:-pretrained}"
case "$TRAINING_MODE" in
    pretrained)
        CKPT_PATH="$MP_CKPT"
        CKPT_DIR="$SETUP/charge3net_checkpoints"
        export WANDB_NAME="pretrained_mp"
        ;;
    from_scratch)
        CKPT_PATH=""  # no --ckpt-path -> ChargE3NetWrapper inits from random
        CKPT_DIR="$SETUP/charge3net_checkpoints_fromscratch"
        export WANDB_NAME="from_scratch"
        ;;
    *)
        echo "ERROR: LEMATRHO_TRAINING_MODE must be 'pretrained' or 'from_scratch'," \
             "got '$TRAINING_MODE'" >&2
        exit 2
        ;;
esac

mkdir -p "$CKPT_DIR" 2>/dev/null || true

# --- Build train command -----------------------------------------------------
# Constructed early so LEMATRHO_DRY_RUN can short-circuit before sourcing venv.
TRAIN_ARGS=(
    --parquet-dir "$DATA_DIR"
    --save-dir "$CKPT_DIR"
    --epochs 50
    --batch-size 16
    --lr 5e-4
    --train-probes 200
    --val-probes 1000
    # num-workers=2 (down from 8): with 4 DDP ranks each forking workers, the
    # previous setting created 32 worker processes total and the per-worker
    # _TABLE_CACHE in data.py OOM-killed jobs 4971293/4971343 at ~140 GB
    # cumulative RSS. The LRU eviction we landed in data.py would help on
    # its own, but lowering worker count further drops cache pressure with
    # zero loss in throughput at this dataset/grid size.
    --num-workers 2
    --wandb-project lemat-rho-charge3net
    --wandb-entity dtts
    --wandb-mode offline
)
if [ -n "$CKPT_PATH" ]; then
    TRAIN_ARGS+=(--ckpt-path "$CKPT_PATH")
fi
if [ -f "$CKPT_DIR/latest.pt" ]; then
    TRAIN_ARGS+=(--resume-from "$CKPT_DIR/latest.pt")
fi

if [ "${LEMATRHO_DRY_RUN:-0}" = "1" ]; then
    echo "WANDB_NAME=$WANDB_NAME"
    echo "TRAINING_MODE=$TRAINING_MODE"
    echo "CKPT_DIR=$CKPT_DIR"
    printf 'python -m charge3net_ft.train'
    for arg in "${TRAIN_ARGS[@]}"; do
        printf ' %s' "$arg"
    done
    printf '\n'
    exit 0
fi

# --- Environment -------------------------------------------------------------
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

# --- NCCL / DDP reliability tweaks ---
# Job 4977567 (2026-05-21) ran 2h41m, then died from NCCL TCPStore
# "Broken pipe / should dump flag" on the DDP heartbeat. Memory was
# fine (14 GB/task with the LRU cache fix). The crash is on the
# inter-rank communication channel, not the model. These three env
# vars expand the timeout budget so a transient slow rank doesn't
# tear down the whole job.
#   NCCL_TIMEOUT                       per-collective timeout (seconds)
#   NCCL_ASYNC_ERROR_HANDLING=1        clean shutdown on rank failure
#                                      (no cascading hangs)
#   TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC   how long a rank can stall
#                                      before HeartbeatMonitor tears
#                                      down the process group
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000  # capture more debug info on next crash

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
echo "Training mode: $TRAINING_MODE (wandb name: $WANDB_NAME)"
echo "Checkpoint dir: $CKPT_DIR"
echo "WORLD_SIZE=$WORLD_SIZE  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
rocm-smi || true

python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
"

cd "$WORK_DIR"

# --- Train ------------------------------------------------------------------
# srun launches 4 tasks (--ntasks-per-node=4 from #SBATCH). Each task sees
# SLURM_PROCID = global rank, SLURM_LOCALID = local rank within node.
# The TRAIN_ARGS array is exported as a quoted string so the srun-spawned
# bash can reconstruct it.
TRAIN_ARGS_QUOTED=""
for arg in "${TRAIN_ARGS[@]}"; do
    TRAIN_ARGS_QUOTED+=" $(printf '%q' "$arg")"
done
export TRAIN_ARGS_QUOTED

srun --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    # Each task sees ALL 4 GCDs the job was allocated; torch.cuda.set_device(local_rank)
    # inside _setup_ddp picks the right one. Restricting visibility per-task here
    # would make every task target the same "GCD 0" within its own visibility set.
    echo "task RANK=$RANK LOCAL_RANK=$LOCAL_RANK on $(hostname) (will use cuda:$LOCAL_RANK)"
    eval "python3 -m charge3net_ft.train $TRAIN_ARGS_QUOTED"
'

echo "Done. Exit code: $?"
