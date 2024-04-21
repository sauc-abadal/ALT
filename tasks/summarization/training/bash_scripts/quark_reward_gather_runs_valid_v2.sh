#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/quark_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/quark/output_iter_1/quark_v2_sampling_data_valid_split_iter_1.json"
output_dir="/cluster/work/sachan/NLF/quark/output_iter_1/"

num_generations=1

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
# bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
#     "$input_sampling_file" \
#     "${output_dir}/valid_output_"{0..7}.json

# Submit SLURM jobs and capture job IDs
reward1=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 8 "$num_generations" | awk '{print $4}')
reward2=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 8 "$num_generations" | awk '{print $4}')
reward3=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 8 "$num_generations" | awk '{print $4}')
reward4=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 8 "$num_generations" | awk '{print $4}')
reward5=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 4 8 "$num_generations" | awk '{print $4}')
reward6=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 5 8 "$num_generations" | awk '{print $4}')
reward7=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 6 8 "$num_generations" | awk '{print $4}')
reward8=$(sbatch tasks/summarization/training/bash_scripts/quark_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 7 8 "$num_generations" | awk '{print $4}')
