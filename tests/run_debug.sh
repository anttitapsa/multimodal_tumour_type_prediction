#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=5G
#SBATCH --partition=gpushort
#SBATCH --gres=gpu

python3 debugging_model.py