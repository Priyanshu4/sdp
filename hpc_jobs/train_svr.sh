#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks 4

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate xrd_ml

python ../xrd_ml/svr.py