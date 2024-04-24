#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/quark_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/quark_sampling_data_valid_split_iter_10_feedback_subset_0_sentstripped.json"
output_dir="/cluster/work/sachan/NLF/"

num_generations=1

# Submit SLURM jobs and capture job IDs
feedback1=$(sbatch tasks/summarization/training/bash_scripts/quark_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 1 "$num_generations" | awk '{print $4}')

