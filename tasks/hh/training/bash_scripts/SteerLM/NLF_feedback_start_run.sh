#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=120G
#SBATCH --output="/cluster/work/sachan/NLF/hh_SteerLM/slurm_output/gpt3.5_feedback_%j.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

echo "--config: $1"
echo "--input_sampling_file: $2"
echo "--output_dir: $3"
echo "--split_number: $4"
echo "--total_splits: $5"
echo "--num_generations: $6"

python tasks/hh/training/feedback_SteerLM.py \
    --config "$1" \
    --input_sampling_file "$2" \
    --output_dir "$3" \
    --split_number "$4" \
    --total_splits "$5" \
    --num_generations "$6" \
