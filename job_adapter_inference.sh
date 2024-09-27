#!/bin/bash
#SBATCH --job-name=inf_adapter
#SBATCH --time=1-05:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

conda activate knobgen

echo "Starting accelerate..."
srun python3 inference_adapter.py --config configs/multigen20k_adapter.yaml --launcher slurm 
