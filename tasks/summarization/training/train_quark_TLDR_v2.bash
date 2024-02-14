#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=48000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_TLDR_5q.out"
#SBATCH --open-mode=append

module load eth_proxy

python tasks/summarization/training/train_quark_TLDR.py --config tasks/summarization/training/train_quark_TLDR_config.yml
