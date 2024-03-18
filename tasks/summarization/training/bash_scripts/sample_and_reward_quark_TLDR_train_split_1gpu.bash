#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=24:00:00
#SBATCH --output="output/quark_sampling_and_rewarding_TLDR_5q_train_split_sampling_stage_2.out"
#SBATCH --open-mode=append

accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_1gpu.yaml tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/configs/quark_TLDR_config.yml --first_iter False --split train
python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/quark_TLDR_config.yml --split train