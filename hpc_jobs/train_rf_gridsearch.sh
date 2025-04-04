#!/bin/bash

#SBATCH --ntasks 16
#SBATCH -o rf_gridsearch_%J.out

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/rf_gridsearch.py "$@"