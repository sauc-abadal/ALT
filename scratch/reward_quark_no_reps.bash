#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_reward_TLDR_5q_sampling_stage_2_no_reps.out"
#SBATCH --open-mode=append

python scratch/quark_reward_filtered.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml --root_path /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v2/sampling --input_json_file quark_sampling_data_train_stage_2_no_reps.json 
