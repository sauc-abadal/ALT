#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=120G

source path_to_anaconda/anaconda3/bin/activate sample

python tasks/summarization/ALT_LMU_feedback.py \
    --config "$1" \
    --input_sampling_file "$2" \
    --output_dir "$3" \
    --split_number "$4" \
    --total_splits "$5" \
    --num_generations "$6" \
    --ALT
