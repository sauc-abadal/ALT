#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/NLF_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/nlf/output_iter_1/NLF_sampling_data_valid_split_iter_1_varied_feedback.json"
output_dir="/cluster/work/sachan/NLF/nlf/output_iter_1/"

num_generations=1

sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 4 "$num_generations"
sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 4 "$num_generations"
sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 4 "$num_generations"
sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 4 "$num_generations"
