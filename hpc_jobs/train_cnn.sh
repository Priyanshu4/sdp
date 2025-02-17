#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p general
#SBATCH --gres=gpu:1  # Request GPU
#SBATCH --mem=32G     # Request memory
#SBATCH --time=24:00:00  # Request 24 hours
#SBATCH -o cnn_training_%J.out
#SBATCH -e cnn_training_%J.err

module load python/3.8
module load cuda/11.7  # For GPU support

# Run training
cd ../xrd_ml
python CNN.py
