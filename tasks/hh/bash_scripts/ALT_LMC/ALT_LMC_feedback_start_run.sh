#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=120G

source path_to_anaconda/anaconda3/bin/activate sample

python tasks/hh/ALT_LMC_feedback.py \
    --config "$1" \
    --input_sampling_file "$2" \
    --output_dir "$3" \
    --split_number "$4" \
    --total_splits "$5" \
    --num_generations "$6" \
