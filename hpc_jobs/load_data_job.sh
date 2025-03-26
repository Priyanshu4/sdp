#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o load_data_%J.out

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/load_data.py