#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=4:00:00
#SBATCH --output="output/quark_sampling_and_reward_TLDR_5q_valid_sampling_stage_1.out"
#SBATCH --open-mode=append

python tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml --first_iter False --split valid
python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml --first_iter False --split valid