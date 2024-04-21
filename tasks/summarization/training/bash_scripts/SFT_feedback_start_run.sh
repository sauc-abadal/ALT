#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=96G
#SBATCH --output="/cluster/work/sachan/NLF/quark/slurm_output/SFT_gpt3.5_feedback_%j.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

echo "--config: $1"
echo "--input_sampling_file: $2"
echo "--output_dir: $3"
echo "--split_number: $4"
echo "--total_splits: $5"
echo "--num_generations: $6"

python tasks/summarization/training/feedback.py \
    --config "$1" \
    --input_sampling_file "$2" \
    --output_dir "$3" \
    --split_number "$4" \
    --total_splits "$5" \
    --num_generations "$6" \
