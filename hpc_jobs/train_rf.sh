#!/bin/bash

#SBATCH --job-name=random_forest
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=8            
#SBATCH -p general                   
#SBATCH -o rf_training_%J.out        
#SBATCH -e rf_training_%J.err        

module purge                         

source ~/miniconda3/etc/profile.d/conda.sh  

conda activate xrd_ml

python ../xrd_ml/Random_Forest.py    