#!/bin/bash
# BOA (boa_ft) training on Adastra (CINES, AMD MI250X), single GCD.
#
# Trains BOA from scratch on the LeMat-Rho charge-density dataset. Mirrors the
# structure of submit_charge3net_adastra.sh (MI250 headers, proxy, venv311_fresh), but
# BOA is driven by Hydra rather than argparse.
#
# Env vars:
#   LEMATRHO_ADASTRA_SETUP   override $SETUP  (default: /lus/scratch/CT10/cad16353/msiron/charge3net_setup)
#   LEMATRHO_DRY_RUN         1 to print the resolved train command and exit
#
# Submit:
#   sbatch submit_boa_adastra.sh
#
# Single-GCD layout (start small before scaling to DDP):
#   - 1 GCD, 16 CPUs, memory proportional to CPU share.
#SBATCH --job-name=boa_ft
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
# account (--account=c1816212 above) handles billing independently.
SETUP="${LEMATRHO_ADASTRA_SETUP:-/lus/scratch/CT10/cad16353/msiron/charge3net_setup}"
WORK_DIR="$SETUP/LeMat-Rho"
BOA_CLONE="$SETUP/boa"
DATA_ROOT="$SETUP/boa_data"
MODEL_ROOT="$SETUP/boa_models"

# BOA path variables (read by configs/paths/default.yaml).
export PROJECT_ROOT="$BOA_CLONE"
export BOA_DATA="$DATA_ROOT"
export BOA_MODELS="$MODEL_ROOT"
mkdir -p "$BOA_DATA" "$BOA_MODELS" 2>/dev/null || true

# --- Build train command -----------------------------------------------------
# Read the element set discovered at preprocess time so the basis matches data.
ATOMIC_NUMBERS_FILE="$BOA_DATA/lematrho/atomic_numbers.json"

TRAIN_CMD=(
    python "$BOA_CLONE/boa/train.py"
    "hydra.searchpath=[file://$WORK_DIR/boa_ft/configs]"
    experiment=lematrho
    trainer.accelerator=gpu
    +trainer.devices=1
    logger=tensorboard
)

if [ "${LEMATRHO_DRY_RUN:-0}" = "1" ]; then
    if [ -f "$ATOMIC_NUMBERS_FILE" ]; then
        ATOMIC_NUMBERS=$(python -c "import json; print(json.load(open('$ATOMIC_NUMBERS_FILE')))")
        TRAIN_CMD+=("data.basis_info.atomic_numbers=$ATOMIC_NUMBERS")
    fi
    printf '%s ' "${TRAIN_CMD[@]}"
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

source "$SETUP/venv311_fresh/bin/activate"

export PYTHONPATH="$WORK_DIR:$BOA_CLONE:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Load W&B key from .env if present.
if [ -f "$WORK_DIR/.env" ]; then
    set -a
    source "$WORK_DIR/.env"
    set +a
fi

# --- ROCm device selection ---
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

# atomic_numbers must be read after the venv is active (needs python).
if [ ! -f "$ATOMIC_NUMBERS_FILE" ]; then
    echo "ERROR: $ATOMIC_NUMBERS_FILE not found. Run boa_ft.preprocess first." >&2
    exit 2
fi
ATOMIC_NUMBERS=$(python -c "import json; print(json.load(open('$ATOMIC_NUMBERS_FILE')))")
TRAIN_CMD+=("data.basis_info.atomic_numbers=$ATOMIC_NUMBERS")

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Job dir: $WORK_DIR"
echo "BOA clone: $BOA_CLONE"
echo "atomic_numbers: $ATOMIC_NUMBERS"
rocm-smi || true

python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
"

cd "$WORK_DIR"

# --- Train ------------------------------------------------------------------
srun --kill-on-bad-exit=1 "${TRAIN_CMD[@]}"

echo "Done. Exit code: $?"
