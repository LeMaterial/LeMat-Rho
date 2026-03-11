#!/bin/bash
#SBATCH --account=cad16353
#SBATCH --job-name="test_vasp"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00

module purge

module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-gnu

module load cray-hdf5 cray-fftw

module list

# export OMP_DISPLAY_AFFINITY=TRUE
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=THREADS

export OMP_NUM_THREADS=1

export PATH=/lus/scratch/CT10/cad16353/mfranckel/vasp_src/2024_V6.5/vasp.6.5.0/bin:$PATH

srun --ntasks-per-node=192 --cpus-per-task="${OMP_NUM_THREADS}" \
     --threads-per-core=1 --label \
     vasp_std
~                
