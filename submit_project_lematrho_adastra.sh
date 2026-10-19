#!/bin/bash
# Phase D2: project the LeMat-Rho parquet dataset onto the SALTED basis.
#
# One-time CPU job. Reads $SETUP/charge3net_data/chunk_*.parquet,
# writes $SETUP/salted_projected_coefficients/chunk_*.parquet via
# salted_ft.project_dataset (one LSQR per row, ~75 ms per row).
#
# Adastra smoke test (1 chunk, 956 valid rows) timed at 71 s wall.
# Full dataset (69 chunks, ~65k rows) extrapolates to ~80 min.
# Budget 2 h with slack.
#
# Env vars
#   LEMATRHO_ADASTRA_SETUP  override $SETUP (default: cad16353 scratch)
#
#SBATCH --job-name=salted_project_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=c1816212
#SBATCH --constraint=GENOA
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Resource sizing notes (2026-05-22):
# - --partition=genoa-shared rejected by CINES policy ("You are not allowed
#   to ask for a partition"), same as --qos=debug. We use --constraint=GENOA
#   and let SLURM auto-route based on resource size.
# - Bumped --cpus-per-task from 16 to 4 so SLURM keeps us in genoa-shared
#   (it auto-routes to the shared partition for small CPU asks, exclusive
#   for larger ones). 4 CPUs is enough for our numpy-LSQR + BLAS thread
#   pool; the projection is ~1 min/chunk, single chunk is the bottleneck.

set -eo pipefail

SETUP="${LEMATRHO_ADASTRA_SETUP:-/lus/scratch/CT10/cad16353/msiron/charge3net_setup}"
WORK_DIR="$SETUP/LeMat-Rho"
INPUT_DIR="$SETUP/charge3net_data"
OUTPUT_DIR="$SETUP/salted_projected_coefficients"

mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

source "$SETUP/venv311/bin/activate"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# numpy / lstsq is already multi-threaded via BLAS; cap thread count
# to match the SLURM allocation so we do not oversubscribe the node.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_ON_NODE
export MKL_NUM_THREADS=$SLURM_CPUS_ON_NODE

echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "CPUs:   $SLURM_CPUS_ON_NODE"

cd "$WORK_DIR"

python -m salted_ft.project_dataset \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Done. Exit code: $?"
echo "Counting output chunks:"
ls "$OUTPUT_DIR"/chunk_*.parquet | wc -l
