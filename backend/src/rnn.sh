#!/bin/bash
#SBATCH --qos=gpu-long
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=fact
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20g
#SBATCH --chdir=/fs/clip-quiz/entilzha/code/fact-repetition/backend/src
#SBATCH --output=/fs/www-users/entilzha/logs/%A.log
#SBATCH --error=/fs/www-users/entilzha/logs/%A.log

set -e
hostname
nvidia-smi
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate fact
allennlp train --include-package fact -s models/karl-bert -f configs/generated/rnn.jsonnet