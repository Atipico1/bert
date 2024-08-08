#!/bin/bash
#SBATCH -J preprocess
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:0
#SBATCH --mem=200G
#SBATCH -o log/preprocess.out
#SBATCH -e log/preprocess.err
#SBATCH --time 48:00:00

python preprocess_dataset.py