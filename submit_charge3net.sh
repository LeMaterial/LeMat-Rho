#!/bin/bash
#SBATCH --job-name=charge3net_ft
#SBATCH --account=dqo@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

# --- Environment ---
source /etc/profile
eval "$($WORK/miniforge3/bin/conda shell.bash hook)"
conda activate uma12

cd $SCRATCH/LeMat-Rho

# Load W&B API key from .env
export $(grep -v '^#' .env | xargs)

# --- Train ---
python train.py \
    --parquet-dir $SCRATCH/charge3net_data/lematrho_full_10x10x10 \
    --ckpt-path $SCRATCH/charge3net/models/charge3net_mp.pt \
    --save-dir $SCRATCH/charge3net_checkpoints \
    --epochs 50 \
    --batch-size 4 \
    --lr 5e-4 \
    --train-probes 200 \
    --val-probes 1000 \
    --num-workers 8 \
    --wandb-project lemat-rho-charge3net \
    --wandb-entity dtts

echo "Done. Exit code: $?"
