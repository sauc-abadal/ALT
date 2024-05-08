og_input_file=/cluster/work/sachan/NLF/quarkToNLF_v2/all_prompts_train_conditioned.json
input_file=/cluster/work/sachan/NLF/quarkToNLF_v2/sampled_prompts_iter_11.json

shuf -n 2048 $og_input_file > $input_file


input_sampling_file_train=/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_11/quark_sampling_data_train_split_iter_11.json

output_dir=/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_11
model_path=/cluster/work/sachan/NLF/quarkToNLF_v2/model/iter_10/model_ckp_10
tokenizer_path=/cluster/work/sachan/NLF/nlf_v2/NLF_TLDR_tokenizer

data_split=train

num_generations=20
temperature=0.9
top_p=0.9
max_new_tokens=64

# Submit SLURM jobs and capture job IDs
sample1=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 0 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample2=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 1 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample3=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 2 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample4=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 3 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample5=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 4 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample6=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 5 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample7=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 6 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample8=$(sbatch tasks/summarization/training/bash_scripts/quarkToNLF/quarkToNLF_vllm_sampling_start_run.sh "$input_file" "$output_dir" 7 8 "$model_path" "$tokenizer_path" "$data_split" $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')

# concatenate 8 sampled files (dependency on 'sample0..7') and capture job ID
concat=$(sbatch --dependency=afterok:$sample1:$sample2:$sample3:$sample4:$sample5:$sample6:$sample7:$sample8 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${data_split}_output_"{0..7}.json | awk '{print $4}')