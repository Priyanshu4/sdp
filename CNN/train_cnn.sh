#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH -o cnn_training_%J.out
#SBATCH -e cnn_training_%J.err

module load python/3.8
python train_cnn.py
