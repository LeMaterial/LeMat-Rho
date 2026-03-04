#!/usr/bin/env bash
#SBATCH --account=cad16353
#SBATCH --job-name=lemat_rho_workers
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/lemat_rho_workers_%j.out
#SBATCH --error=logs/lemat_rho_workers_%j.err
#
# Adastra: Fireworks workers for LeMat-Rho DFT (VASP).
# Jobs must already be in the LaunchPad (submitted via run_utilities.run_batch or run_calculation).
# This script runs rlaunch to consume the queue and execute VASP on this node.
#
# Before running: create logs/, set REPO_ROOT and LAUNCHPAD_YAML if needed, then:
#   sbatch scripts/submit_dft_workers.sh

set -e

module purge
module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-gnu
module load cray-hdf5 cray-fftw
module list

export OMP_PROC_BIND=CLOSE
export OMP_PLACES=THREADS
export OMP_NUM_THREADS=1

# VASP (required by atomate2; adjust path if needed)
export PATH=/lus/scratch/CT10/cad16353/mfranckel/vasp_src/2024_V6.5/vasp.6.5.0/bin:$PATH
export VASP_CMD="srun --ntasks-per-node=192 --cpus-per-task=${OMP_NUM_THREADS} --threads-per-core=1 vasp_std"

# Repo and Python env
REPO_ROOT="${REPO_ROOT:-/lus/scratch/CT10/cad16353/mfranckel/LeMat-Rho}"
cd "${REPO_ROOT}"
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

# LaunchPad config (set if not using default)
if [[ -n "${LAUNCHPAD_YAML}" ]]; then
  export FW_CONFIG_FILE="${LAUNCHPAD_YAML}"
fi

# Run Fireworks workers (same workflow as elsewhere; worker name must match run_batch(..., worker=...))
rlaunch "${NLAUNCHES:-multi}"
