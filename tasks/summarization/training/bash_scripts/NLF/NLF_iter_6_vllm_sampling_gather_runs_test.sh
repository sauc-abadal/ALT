#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

input_file=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_6/TLDR_test_split_prompts.json

output_dir=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_6
model_path=/cluster/work/sachan/NLF/nlf_v2/model/iter_6/model_ckp_6
tokenizer_path=/cluster/work/sachan/NLF/nlf/NLF_TLDR_tokenizer

data_split=test

num_generations=1
temperature=0.0
top_p=1.0
max_new_tokens=128

# Submit SLURM jobs and capture job IDs
sample1=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 0 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample2=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 1 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample3=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 2 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample4=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 3 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')

# individual jsonl files named saved in f"{args.output_dir}/{args.data_split}_output_{args.split_number}.json"

# the files can be concatenated without the need of adding a newline in between, as "\n" is already included 
# at the end of every line.

# Submit NLF_reward_gather_runs_valid.sh after all jobs complete
sbatch --dependency=afterok:$sample1:$sample2:$sample3:$sample4 tasks/summarization/training/bash_scripts/NLF/NLF_iter_6_reward_gather_runs_test.sh
