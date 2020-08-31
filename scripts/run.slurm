#!/bin/bash
#SBATCH --job-name=retention
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20g
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/fs/clip-quiz/shifeng/karl/
#SBATCH --exclude=materialgpu00

python -m karl.get_all_predictions
