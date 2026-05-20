#!/bin/bash
# DeepDFT training on Adastra (CINES, AMD MI250X), half-node DDP.
#
# Comparison baseline for ChargE3Net. Uses PaiNN (the equivariant variant)
# for an apples-to-apples comparison since ChargE3Net is also equivariant.
#
# Env vars:
#   LEMATRHO_ADASTRA_SETUP   override $SETUP            (default: cad16353 scratch)
#   LEMATRHO_DEEPDFT_VARIANT painn (default) | schnet   (model architecture)
#   LEMATRHO_DRY_RUN         1 to print the resolved train command and exit
#
# Submit examples:
#   sbatch submit_deepdft_adastra.sh                                                              # PaiNN
#   sbatch --export=ALL,LEMATRHO_DEEPDFT_VARIANT=schnet submit_deepdft_adastra.sh                # SchNet
#
# Half-node resource layout (matches submit_charge3net_adastra.sh):
#   - 4 GCDs, 64 CPUs, 128 GB RAM
#   - 4 tasks, one per GCD, for torch DistributedDataParallel
#SBATCH --job-name=deepdft_ft
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
SETUP="${LEMATRHO_ADASTRA_SETUP:-/lus/scratch/CT10/cad16353/msiron/charge3net_setup}"
WORK_DIR="$SETUP/LeMat-Rho"
DATA_DIR="$SETUP/charge3net_data"
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
        EXTRA_ARGS=()
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
# DeepDFT runner reads --dataset; we point it at the LeMat-Rho parquet dir
# and let deepdft_ft/runner.py:_is_parquet_dir auto-route to our adapter.
TRAIN_ARGS=(
    --dataset "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --cutoff 4.0
    --num_interactions 3
    --node_size 128
    --max_steps 100000000
    --device cuda
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

source "$SETUP/venv311/bin/activate"

export PYTHONPATH="$WORK_DIR:$DEEPDFT_REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

if [ -f "$WORK_DIR/.env" ]; then
    set -a
    source "$WORK_DIR/.env"
    set +a
fi

# --- Distributed-training env vars ---
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29501  # different from charge3net (29500) so concurrent jobs don't collide

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Variant: $VARIANT (wandb name: $WANDB_NAME)"
echo "Output dir: $OUTPUT_DIR"
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
TRAIN_ARGS_QUOTED=""
for arg in "${TRAIN_ARGS[@]}"; do
    TRAIN_ARGS_QUOTED+=" $(printf '%q' "$arg")"
done
export TRAIN_ARGS_QUOTED

srun --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    echo "task RANK=$RANK LOCAL_RANK=$LOCAL_RANK on $(hostname) (will use cuda:$LOCAL_RANK)"
    eval "python3 -m deepdft_ft.runner $TRAIN_ARGS_QUOTED"
'

echo "Done. Exit code: $?"
