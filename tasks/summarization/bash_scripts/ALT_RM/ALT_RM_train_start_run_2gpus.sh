#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --mem-per-cpu=160000
#SBATCH --time=6:00:00

source path_to_anaconda/anaconda3/bin/activate gptj_training

accelerate launch --config_file "$1" tasks/summarization/ALT_RM_train_noKL.py \
    --config "$2" \
    --iteration "$3" \
    --input_sampling_file "$4" \
    --model_path "$5" \
    --ds_optimizer \
    --ds_scheduler