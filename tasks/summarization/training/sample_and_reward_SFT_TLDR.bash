#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=4:00:00
#SBATCH --output="output/SFT_sampling_and_reward_TLDR.out"
#SBATCH --open-mode=append

python tasks/summarization/training/SFT_sampling.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml --split valid
python tasks/summarization/training/SFT_reward.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml --split valid
