#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=160G
#SBATCH --time=4:00:00


source path_to_anaconda/anaconda3/bin/activate gptj_training

python tasks/summarization/reward.py \
    --config "$1" \
    --input_sampling_file "$2" \
    --output_dir "$3" \
    --split_number "$4" \
    --total_splits "$5" \
    --num_generations "$6" \
    --ALT

