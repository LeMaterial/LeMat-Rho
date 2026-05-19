#!/bin/bash
# ChargE3Net fine-tuning on Adastra (CINES, AMD MI250X).
# See ADASTRA.md for setup details and known gotchas.
#SBATCH --job-name=charge3net_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=c1816212
#SBATCH --constraint=MI250
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
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

# HIP / CUDA device alignment (AMD ROCm). HIP_VISIBLE_DEVICES is the AMD
# equivalent of CUDA_VISIBLE_DEVICES; PyTorch reads CUDA_VISIBLE_DEVICES,
# so we mirror one into the other.
if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export HIP_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    else
        export HIP_VISIBLE_DEVICES=0
    fi
fi
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

export PYTHONPATH="$WORK_DIR:$SETUP/charge3net:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Load W&B key from .env if present.
if [ -f "$WORK_DIR/.env" ]; then
    set -a
    source "$WORK_DIR/.env"
    set +a
fi

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Job dir: $WORK_DIR"
rocm-smi || true

python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
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

# Knobs match Jean Zay's submit_charge3net.sh apart from a) larger batch size
# (MI250X has 64 GB HBM2e per GCD; A100 ran batch=4) and b) wandb online (the
# Adastra proxy gives us live internet, no offline-then-sync dance).
python3 -m charge3net_ft.train \
    --parquet-dir "$DATA_DIR" \
    --ckpt-path "$MP_CKPT" \
    --save-dir "$CKPT_DIR" \
    --epochs 50 \
    --batch-size 8 \
    --lr 5e-4 \
    --train-probes 200 \
    --val-probes 1000 \
    --num-workers 8 \
    --wandb-project lemat-rho-charge3net \
    --wandb-entity dtts \
    --wandb-mode online \
    $RESUME_FLAG

echo "Done. Exit code: $?"
