#!/bin/bash
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH -o log/train.out
#SBATCH -e log/train.err
#SBATCH --time 144:00:00

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train.py