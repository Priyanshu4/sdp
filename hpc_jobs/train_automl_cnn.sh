#!/bin/bash
#SBATCH --ntasks 16
#SBATCH -p general
#SBATCH --time=12:00:00  
#SBATCH -o automl_cnn_%J.out

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/automl_cnn.py "$@"