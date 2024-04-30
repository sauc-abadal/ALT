#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/quark_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/CarperAI_test_prompts/quark_iter_10/quark_sampling_data_test_split_iter_10.json"
output_dir="/cluster/work/sachan/NLF/CarperAI_test_prompts/quark_iter_10"

num_generations=1

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/test_output_"{0..3}.json

# Submit SLURM jobs and capture job IDs
reward1=$(sbatch tasks/summarization/training/bash_scripts/quark/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 4 "$num_generations" | awk '{print $4}')
reward2=$(sbatch tasks/summarization/training/bash_scripts/quark/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 4 "$num_generations" | awk '{print $4}')
reward3=$(sbatch tasks/summarization/training/bash_scripts/quark/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 4 "$num_generations" | awk '{print $4}')
reward4=$(sbatch tasks/summarization/training/bash_scripts/quark/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 4 "$num_generations" | awk '{print $4}')

