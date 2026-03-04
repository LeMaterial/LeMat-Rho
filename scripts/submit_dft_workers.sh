#!/usr/bin/env bash
#SBATCH --job-name=lemat_rho_workers
#SBATCH --output=logs/lemat_rho_workers_%j.out
#SBATCH --error=logs/lemat_rho_workers_%j.err
#SBATCH --partition=PARTITION_NAME
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --ntasks=1
#
# Fireworks worker(s) for LeMat-Rho DFT (VASP) calculations.
# Jobs must already be in the LaunchPad (submitted via run_utilities.run_batch or run_calculation).
# This script runs rlaunch to consume the queue and execute VASP on this node.
#
# Before running:
#   1. Create logs/ if using the default --output/--error paths.
#   2. Set PARTITION_NAME (and optionally time, cpus-per-task, mem).
#   3. Set REPO_ROOT and activate your env + set VASP_CMD in the block below.
#   4. Set LAUNCHPAD_YAML to your LaunchPad config (or leave unset to use Fireworks default).
#   5. Submit: sbatch scripts/submit_dft_workers.sh

set -e

# --- Configure these (or override with env vars) ---
REPO_ROOT="${REPO_ROOT:-/path/to/LeMat-Rho}"
LAUNCHPAD_YAML="${LAUNCHPAD_YAML:-}"   # e.g. /path/to/lematrho_launchpad.yaml
WORKER_NAME="${WORKER_NAME:-lemat_rho_worker}"
NLAUNCHES="${NLAUNCHES:-multi}"         # "singles" = one firework per job; "multi" = run until queue empty

# --- Environment: Python + VASP ---
cd "${REPO_ROOT}"
# Activate conda or venv (uncomment and adjust one)
# source "${REPO_ROOT}/.venv/bin/activate"
# conda activate lemat_rho

# VASP and MPI (required by atomate2; adjust paths for your cluster)
# export PATH=/path/to/intel/mpi/bin:/path/to/vasp/bin:$PATH
# export VASP_CMD="mpirun -np ${SLURM_CPUS_PER_TASK} /path/to/vasp/bin/vasp_std"

# Optional: point Fireworks at your LaunchPad config
if [[ -n "${LAUNCHPAD_YAML}" ]]; then
  export FW_CONFIG_FILE="${LAUNCHPAD_YAML}"
fi

# Run Fireworks workers (they run VASP jobs from the LaunchPad queue).
# Use the same worker name in your queue adapter / LaunchPad config as in run_batch(..., worker=...).
rlaunch "${NLAUNCHES}"
