#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=160G
#SBATCH --time=2:00:00

python tasks/summarization/training/quark_reward.py --config $1 --input_sampling_file $2 --output_dir $3 --split_number $4 --total_splits $5

