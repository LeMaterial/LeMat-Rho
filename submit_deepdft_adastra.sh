#!/bin/bash
# DeepDFT training on Adastra (CINES, AMD MI250X), single-GPU paper-faithful.
#
# Faithful to peterbjorgensen/DeepDFT paper settings:
#   - 1 GCD (paper used 1x RTX 3090; we use 1x MI250X)
#   - batch=2 materials, train=1000 probes/material (same as upstream);
#     val=1000 probes/material over a 200-material seeded subsample
#     (upstream's val=5000 probes OOM-killed the 64 GB job 5004725: the
#     probe neighborlist grows quadratically in the probe count)
#   - cutoff=4 A, num_interactions=3, node_size=128, PaiNN model
#   - max_steps=10,000,000
#
# Single-GPU keeps the gradient-step semantics identical to the paper.
# DDP code paths in runner.py only fire when WORLD_SIZE>1 -- we leave them
# out here on purpose. If we ever want DDP for DeepDFT we'd also need to
# sweep the LR (effective batch grows with world_size).
#
# Env vars:
#   LEMATRHO_ADASTRA_SETUP    override $SETUP            (default: cad16353 scratch)
#   LEMATRHO_DEEPDFT_VARIANT  painn (default) | schnet
#   LEMATRHO_DRY_RUN          1 to print resolved cmd + exit
#
# Submit examples:
#   sbatch submit_deepdft_adastra.sh                                                # PaiNN
#   sbatch --export=ALL,LEMATRHO_DEEPDFT_VARIANT=schnet submit_deepdft_adastra.sh   # SchNet
#
#SBATCH --job-name=deepdft_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=c1816212
#SBATCH --constraint=MI250
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

# --- Paths ---
SETUP="${LEMATRHO_ADASTRA_SETUP:-/lus/scratch/CT10/cad16353/msiron/charge3net_setup}"
WORK_DIR="$SETUP/LeMat-Rho"
DATA_DIR="$SETUP/charge3net_data_15cube"
DEEPDFT_REPO="$SETUP/DeepDFT"

# --- Model variant ---
VARIANT="${LEMATRHO_DEEPDFT_VARIANT:-painn}"
case "$VARIANT" in
    painn)
        EXTRA_ARGS=(--use_painn_model)
        OUTPUT_DIR="$SETUP/deepdft_runs/painn"
        export WANDB_NAME="deepdft_painn"
        ;;
    schnet)
        EXTRA_ARGS=()  # SchNet is the default architecture, no flag needed
        OUTPUT_DIR="$SETUP/deepdft_runs/schnet"
        export WANDB_NAME="deepdft_schnet"
        ;;
    *)
        echo "ERROR: LEMATRHO_DEEPDFT_VARIANT must be 'painn' or 'schnet', got '$VARIANT'" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

# --- Build train command -----------------------------------------------------
# Hyperparameters lifted from pretrained_models/{nmc,qm9,ethylenecarbonate}_painn
# in the upstream DeepDFT repo. Same values across all three published checkpoints.
TRAIN_ARGS=(
    --dataset "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --cutoff 4
    --num_interactions 3
    --node_size 128
    --max_steps 10000000
    --device cuda
    --val-probes 1000
    --val-max-samples 200
    "${EXTRA_ARGS[@]}"
)
if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
    TRAIN_ARGS+=(--load_model "$OUTPUT_DIR/best_model.pth")
fi

if [ "${LEMATRHO_DRY_RUN:-0}" = "1" ]; then
    echo "WANDB_NAME=$WANDB_NAME"
    echo "VARIANT=$VARIANT"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    printf 'python -m deepdft_ft.runner'
    for arg in "${TRAIN_ARGS[@]}"; do
        printf ' %s' "$arg"
    done
    printf '\n'
    exit 0
fi

# --- Environment -------------------------------------------------------------
export HTTP_PROXY=http://proxy-l-adastra.cines.fr:3128
export HTTPS_PROXY=$HTTP_PROXY
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTP_PROXY

source "$SETUP/venv311_fresh/bin/activate"

export PYTHONPATH="$WORK_DIR:$DEEPDFT_REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

if [ -f "$WORK_DIR/.env" ]; then
    set -a
    source "$WORK_DIR/.env"
    set +a
fi

# Pin to GCD 0 (single-GPU paper-faithful). Do NOT set WORLD_SIZE so that
# runner.py's _setup_ddp returns the single-process tuple (0, 0, 1).
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Variant: $VARIANT (wandb name: $WANDB_NAME)"
echo "Output dir: $OUTPUT_DIR"
echo "Single-GPU mode (WORLD_SIZE unset)"
rocm-smi || true

python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
"

cd "$WORK_DIR"

# --- Train (single GPU, no srun) --------------------------------------------
python3 -m deepdft_ft.runner "${TRAIN_ARGS[@]}"

echo "Done. Exit code: $?"
