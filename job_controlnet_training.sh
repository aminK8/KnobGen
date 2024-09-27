#!/bin/bash
#SBATCH --job-name=train_controlnet
#SBATCH --time=4-0:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=5



conda activate knobgen

echo "Starting accelerate..."
srun python3 train_controlnet.py --config configs/controlnet.yaml --launcher slurm 
