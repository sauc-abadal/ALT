#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=160G
#SBATCH --time=4:00:00
#SBATCH --output="/cluster/work/sachan/NLF/quarkToNLF/slurm_output/rewarding_%j.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

echo "--config: $1"
echo "--input_sampling_file: $2"
echo "--output_dir: $3"
echo "--split_number: $4"
echo "--total_splits: $5"
echo "--num_generations: $6"

python tasks/summarization/training/reward.py \
    --config "$1" \
    --input_sampling_file "$2" \
    --output_dir "$3" \
    --split_number "$4" \
    --total_splits "$5" \
    --num_generations "$6"

