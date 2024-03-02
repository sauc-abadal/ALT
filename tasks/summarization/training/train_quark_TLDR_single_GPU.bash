#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=96000
#SBATCH --time=24:00:00
#SBATCH --output="output/quark_train_TLDR_5q_single_GPU_sampling_stage_3.out"
#SBATCH --open-mode=append

accelerate launch --config_file tasks/summarization/training/default_config_single_GPU.yaml tasks/summarization/training/quark_train_single_GPU.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml