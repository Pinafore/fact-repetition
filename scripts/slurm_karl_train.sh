#!/bin/bash
#SBATCH --qos=gpu-long
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=karl_bert
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20g
#SBATCH --chdir=/fs/clip-quiz/shifeng/karl
#SBATCH --output=/fs/www-users/shifeng/logs/%A.log
#SBATCH --error=/fs/www-users/shifeng/logs/%A.log

set -e
hostname
nvidia-smi
allennlp train \
    --include-package karl.retention \
    -s checkpoints/karl_bert \
    -f configs/karl_bert.config
