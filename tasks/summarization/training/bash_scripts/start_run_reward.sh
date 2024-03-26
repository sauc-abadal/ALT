#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=160G
#SBATCH --time=2:00:00

python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/configs/quark_TLDR_config_debug.yaml --split train --split_number $1 --total_splits $2

