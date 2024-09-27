#!/bin/bash
#SBATCH --job-name=train_adapter
#SBATCH --time=4-0:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=5



conda activate knobgen

echo "Starting accelerate..."
srun python3 train_adapter.py --config configs/multigen20k_adapter.yaml --launcher slurm 
