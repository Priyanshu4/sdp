#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p general
#SBATCH --gres=gpu:1  # Request GPU
#SBATCH --mem=32G     # Request memory
#SBATCH --time=24:00:00  # Request 24 hours
#SBATCH -o automl_cnn_gpu_%J.out
#SBATCH -e automl_cnn_gpu_%J.err

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/automl_cnn.py