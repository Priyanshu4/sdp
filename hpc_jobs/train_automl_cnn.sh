#!/bin/bash
#SBATCH --ntasks 16
#SBATCH -p general
#SBATCH --time=24:00:00  # Request 24 hours
#SBATCH -o automl_cnn_%J.out
#SBATCH -e automl_cnn_%J.err

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/automl_cnn.py