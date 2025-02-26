#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/svr.py