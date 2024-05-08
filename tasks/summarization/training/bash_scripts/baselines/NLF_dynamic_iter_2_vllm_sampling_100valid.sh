#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

input_file=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_2/NLF_dynamic_sampling_data_valid_split_100random_subset.json

output_dir=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_2
model_path="/cluster/work/sachan/NLF/nlf_v2/model/iter_2/model_ckp_2"
tokenizer_path=/cluster/work/sachan/NLF/nlf/NLF_TLDR_tokenizer

data_split=valid_dynamic

num_generations=1
temperature=0.0
top_p=1.0
max_new_tokens=128

# Submit SLURM jobs and capture job IDs
sample1=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 0 1 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
