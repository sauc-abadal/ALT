#!/bin/bash

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

input_file=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_1_extended/TLDR_test_split_prompts.json

input_sampling_file=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_1_extended/NLF_sampling_data_test_split_iter_1.json
output_dir=/cluster/work/sachan/NLF/CarperAI_test_prompts/nlf_iter_1_extended
model_path=/cluster/work/sachan/NLF/nlf_v3/model/iter_1/model_ckp_1
tokenizer_path=/cluster/work/sachan/NLF/nlf/NLF_TLDR_tokenizer

data_split=test

num_generations=1
temperature=0.0
top_p=1.0
max_new_tokens=128

# Submit SLURM jobs and capture job IDs
sample1=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 0 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample2=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 1 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample3=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 2 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample4=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 3 4 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')

# concatenate 8 sampled files (dependency on 'sample0..3') and capture job ID
sbatch --dependency=afterok:$sample1:$sample2:$sample3:$sample4 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/${data_split}_output_"{0..3}.json