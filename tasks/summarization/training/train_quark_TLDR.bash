#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100-pcie-40gb:4
#SBATCH --mem-per-cpu=48000
#SBATCH --time=4:00:00
#SBATCH --output="output/quark_train_TLDR_5q_sampling_stage_1.out"
#SBATCH --open-mode=append

python set_wandb_run_id.py --config tasks/summarization/training/train_quark_TLDR_config.yml

accelerate launch --config_file tasks/summarization/training/default_config.yaml tasks/summarization/training/quark_train.py --config tasks/summarization/training/train_quark_TLDR_config.yml
