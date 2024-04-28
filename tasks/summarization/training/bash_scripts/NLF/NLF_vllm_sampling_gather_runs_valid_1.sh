#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

input_file=/cluster/work/sachan/NLF/nlf/output_iter_1/NLF_conditioned_prompts_valid_iter_1.json

output_dir=/cluster/work/sachan/NLF/nlf/output_iter_1
model_path=/cluster/work/sachan/NLF/nlf/model/iter_1/model_ckp_2560
tokenizer_path=/cluster/work/sachan/NLF/nlf/NLF_TLDR_tokenizer

data_split=valid

num_generations=1
temperature=0.0
top_p=1.0
max_new_tokens=64

# Submit SLURM jobs and capture job IDs
sample1=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 0 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample2=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 1 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample3=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 2 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample4=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 3 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample5=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 4 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample6=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 5 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample7=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 6 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample8=$(sbatch tasks/summarization/training/bash_scripts/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 7 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')

# individual jsonl files named saved in f"{args.output_dir}/{args.data_split}_output_{args.split_number}.json"

# the files can be concatenated without the need of adding a newline in between, as "\n" is already included 
# at the end of every line.

# Submit NLF_reward_gather_runs_valid.sh after all jobs complete
sbatch --dependency=afterok:$sample1:$sample2:$sample3:$sample4:$sample5:$sample6:$sample7:$sample8 tasks/summarization/training/bash_scripts/NLF_reward_gather_runs_valid_1.sh