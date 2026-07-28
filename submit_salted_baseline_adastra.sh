#!/bin/bash
# Phase D6 (path B): train the SALTED baseline coefficient-prediction
# model on the D2 projected outputs.
#
# Single-GPU MI250X job. Dataset is the 65k r2SCAN structures with
# their pre-projected per-atom basis coefficients (from D2). Loss is
# MSE on the (n_atoms, 100) coefficient vectors. See
# salted_ft/train_baseline.py for the model architecture (SchNet-style
# invariant message passing, 2 cfconv layers).
#
# Env vars
#   LEMATRHO_ADASTRA_SETUP   override $SETUP                  (default: cad16353 scratch)
#   LEMATRHO_DRY_RUN         1 to print resolved cmd and exit
#
# Submit:
#   sbatch submit_salted_baseline_adastra.sh
#
#SBATCH --job-name=salted_baseline
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
#
# Resource sizing notes:
#  - Single GCD: the baseline model is tiny (~50k params) and
#    saturates the per-atom forward path; DDP across multiple GCDs
#    would only help if we batched many structures per step, which
#    the per-atom variable size makes awkward. Single-GPU is fine.
#  - 24h walltime: 10 epochs over 65k rows at ~0.1s/row =~ 2h, plus
#    margin for I/O and Adastra cold-start.

set -eo pipefail

SETUP="${LEMATRHO_ADASTRA_SETUP:-/lus/scratch/CT10/cad16353/msiron/charge3net_setup}"
WORK_DIR="$SETUP/LeMat-Rho"
SOURCE_DIR="$SETUP/charge3net_data"
COEFFS_DIR="$SETUP/salted_projected_coefficients"
OUTPUT_DIR="$SETUP/salted_baseline_runs"
mkdir -p "$OUTPUT_DIR"
CKPT="$OUTPUT_DIR/salted_baseline_${SLURM_JOB_ID:-local}.pt"

source "$SETUP/venv311/bin/activate"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# ROCm visibility (mirrors submit_deepdft_adastra.sh)
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

CMD=(python -m salted_ft.train_baseline
     --source-dir "$SOURCE_DIR"
     --coeffs-dir "$COEFFS_DIR"
     --output-ckpt "$CKPT"
     --n-epochs 10
     --batch-size 8
     --learning-rate 1e-3
     --device cuda)

if [[ "${LEMATRHO_DRY_RUN:-0}" == "1" ]]; then
    printf '%s ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Source dir:  $SOURCE_DIR"
echo "Coeffs dir:  $COEFFS_DIR"
echo "Ckpt out:    $CKPT"

cd "$WORK_DIR"

"${CMD[@]}"

echo "Done. Exit code: $?"
echo "Wrote: $CKPT"
ls -lh "$CKPT"
