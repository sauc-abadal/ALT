#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

config="tasks/summarization/training/configs/NLF_TLDR_config.yaml"
input_sampling_file="/cluster/work/sachan/NLF/nlf/output_iter_2/NLF_sampling_data_train_split_iter_2.json"
output_dir="/cluster/work/sachan/NLF/nlf/output_iter_2/"

num_generations=10

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/train_output_"{0..7}.json

# Submit SLURM jobs and capture job IDs
feedback1=$(sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 0 6 "$num_generations" | awk '{print $4}')
feedback2=$(sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 1 6 "$num_generations" | awk '{print $4}')
feedback3=$(sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 2 6 "$num_generations" | awk '{print $4}')
feedback4=$(sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 3 6 "$num_generations" | awk '{print $4}')
feedback5=$(sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 4 6 "$num_generations" | awk '{print $4}')
feedback6=$(sbatch tasks/summarization/training/bash_scripts/NLF_feedback_start_run.sh "$config" "$input_sampling_file" "$output_dir" 5 6 "$num_generations" | awk '{print $4}')

# Submit reward_gather_runs_valid.sh after all jobs complete
sbatch --dependency=afterok:$feedback1:$feedback2:$feedback3:$feedback4:$feedback5:$feedback6 tasks/summarization/training/bash_scripts/NLF_train_TLDR_2gpus.sh