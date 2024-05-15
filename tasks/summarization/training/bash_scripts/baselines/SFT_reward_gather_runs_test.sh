#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/quarkToNLF_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/CarperAI_test_prompts/SFT/SFT_sampling_data_test_split.json"
file_prefix="SFT_sampling_data_test_split"
output_dir="/cluster/work/sachan/NLF/CarperAI_test_prompts/SFT"

num_generations=1

# Submit SLURM jobs and capture job IDs
reward1=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 4 "$num_generations" | awk '{print $4}')
reward2=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 4 "$num_generations" | awk '{print $4}')
reward3=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 4 "$num_generations" | awk '{print $4}')
reward4=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 4 "$num_generations" | awk '{print $4}')

# concatenate 8 rewarded files (dependency on 'reward_v0..3') and capture job ID
reward_v=$(sbatch --dependency=afterok:$reward1:$reward2:$reward3:$reward4 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/${file_prefix}_reward_thread_"{0..3}.json | awk '{print $4}')