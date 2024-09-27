#!/bin/bash
#SBATCH --job-name=inf_controlnet
#SBATCH --time=1-20:40:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5


conda activate knobgen

echo "Starting accelerate..."
srun python3 inference_controlnet.py --config configs/controlnet.yamll --launcher slurm 
