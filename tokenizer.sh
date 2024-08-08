#!/bin/bash
#SBATCH -J tokenizer
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:0
#SBATCH --mem=200G
#SBATCH -o log/tokenizer.out
#SBATCH -e log/tokenizer.err
#SBATCH --time 48:00:00

python train_tokenizer.py