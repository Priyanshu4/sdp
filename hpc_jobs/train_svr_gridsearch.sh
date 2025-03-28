#!/bin/bash

#SBATCH --ntasks 16
#SBATCH -o svr_gridsearch_%J.out

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/svr_gridsearch.py "$@"