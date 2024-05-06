#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/quarkToNLF_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/CarperAI_test_prompts/TLDR_test_split_prompts.json"
output_dir="/cluster/work/sachan/NLF/CarperAI_test_prompts/"

num_generations=1

# Submit SLURM jobs and capture job IDs
reward1=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 4 "$num_generations" | awk '{print $4}')
reward2=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 4 "$num_generations" | awk '{print $4}')
reward3=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 4 "$num_generations" | awk '{print $4}')
reward4=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 4 "$num_generations" | awk '{print $4}')

