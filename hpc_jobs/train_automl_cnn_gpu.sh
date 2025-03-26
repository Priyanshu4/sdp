#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p general
#SBATCH --gres=gpu:1  # Request GPU
#SBATCH --mem=32G     # Request memory
#SBATCH --time=12:00:00 
#SBATCH -o automl_cnn_gpu_%J.out

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/automl_cnn.py