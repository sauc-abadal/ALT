#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_3090:2
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_test_rewarding_TLDR_5q_train_split_sampling_stage_2.out"
#SBATCH --open-mode=append

accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_sampling_ddp_2gpus.yaml tasks/summarization/training/quark_reward.py --config tasks/summarization/training/configs/quark_TLDR_config_debug.yaml --split train
