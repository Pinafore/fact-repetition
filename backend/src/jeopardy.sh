#!/bin/bash
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --time=19:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=fact-rnn
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20g

set -e
hostname
nvidia-smi
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate fact
allennlp train --include-package fact -s models/karl-rnn-jeopardy -f configs/generated/jeopardy.jsonnet
