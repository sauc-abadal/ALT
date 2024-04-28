#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/quarkToNLF_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_1/quark_sampling_data_valid_split_iter_1.json"
output_dir="/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_1/"

num_generations=1

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/valid_output_"{0..7}.json

# Submit SLURM jobs and capture job IDs
reward1=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 8 "$num_generations" | awk '{print $4}')
reward2=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 8 "$num_generations" | awk '{print $4}')
reward3=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 8 "$num_generations" | awk '{print $4}')
reward4=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 8 "$num_generations" | awk '{print $4}')
reward5=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 4 8 "$num_generations" | awk '{print $4}')
reward6=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 5 8 "$num_generations" | awk '{print $4}')
reward7=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 6 8 "$num_generations" | awk '{print $4}')
reward8=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF_reward_start_run.sh "$config" "$input_sampling_file" "$output_dir" 7 8 "$num_generations" | awk '{print $4}')

# Submit reward_gather_runs_valid.sh after all jobs complete
sbatch --dependency=afterok:$reward1:$reward2:$reward3:$reward4:$reward5:$reward6:$reward7:$reward8 tasks/summarization/training/bash_scripts/quarkToNLF_perplexity.sh
