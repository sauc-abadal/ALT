config="tasks/hh/training/configs/NLF_HH_config.yaml"

# to be initialized and saved in this path!
tokenizer_path="/cluster/work/sachan/NLF/hh_nlf/tokenizer-pythia-2.8b-mitchell-sft_hh_rlhf"

num_generations_valid=1
data_split_valid=test_1000subset
temperature_valid=1.0
top_p_valid=0.9
max_new_tokens_valid=256

iteration=12

input_prompts_file_valid="/cluster/work/sachan/NLF/hh_nlf/HH_test_prompts_1000subset_conditioned.json"

output_dir="/cluster/work/sachan/NLF/hh_nlf/output_iter_${iteration}/"

input_sampling_file_valid="/cluster/work/sachan/NLF/hh_nlf/output_iter_${iteration}/NLF_sampling_data_test_1000subset_iter_${iteration}.json"
file_prefix_valid="NLF_sampling_data_test_1000subset_iter_${iteration}"

model_path="/cluster/work/sachan/NLF/hh_nlf/model/iter_${iteration}/model_ckp_${iteration}"

# 1. ---------------- SAMPLING (valid) ----------------
# Submit SLURM SAMPLE jobs (no dependency) and capture job IDs
sample_v1=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 0 2 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v2=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 1 2 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')

# concatenate 2 sampled files (dependency on 'sample_v0..1') and capture job ID
sample_v=$(sbatch --dependency=afterok:$sample_v1:$sample_v2 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${data_split_valid}_output_"{0..1}.json | awk '{print $4}')

# # 2. ---------------- FEEDBACK (valid) ----------------
# # Submit SLURM FEEDBACK jobs (dependency on 'sample_v') and capture job IDs
feedback_v1=$(sbatch --dependency=afterok:$sample_v tasks/hh/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 0 2 $num_generations_valid | awk '{print $4}')
feedback_v2=$(sbatch --dependency=afterok:$sample_v tasks/hh/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 1 2 $num_generations_valid | awk '{print $4}')

# concatenate 2 feedback files (dependency on 'feedback_v0..1') and capture job ID
feedback_v=$(sbatch --dependency=afterok:$feedback_v1:$feedback_v2 tasks/hh/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${file_prefix_valid}_feedback_subset_"{0..1}.json | awk '{print $4}')

# 3. ---------------- PERPLEXITY (valid) ----------------
# Submit SLURM PERPLEXITY job (dependency on 'feedback_v')
sbatch --dependency=afterok:$feedback_v tasks/hh/training/bash_scripts/NLF/NLF_perplexity.sh \
    "$config" "$output_dir" "${file_prefix_valid}.json"