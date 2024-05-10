config="tasks/summarization/training/configs/NLFv3_TLDR_config.yaml"
accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml
tokenizer_path="/cluster/work/sachan/NLF/nlf_v2/NLF_TLDR_tokenizer"

num_generations_valid=1
data_split_valid=valid
temperature_valid=0.0
top_p_valid=1.0
max_new_tokens_valid=128

iteration=1
input_prompts_file_valid="/cluster/work/sachan/NLF/nlf_v3/NLF_conditioned_prompts_valid_varied_feedbacks.json"
output_dir="/cluster/work/sachan/NLF/nlf_v3/output_iter_${iteration}/"

input_sampling_file_valid="/cluster/work/sachan/NLF/nlf_v3/output_iter_${iteration}/NLF_sampling_data_valid_split_iter_${iteration}.json"
file_prefix_valid="NLF_sampling_data_valid_split_iter_${iteration}"

# 4. ---------------- SAMPLING (valid) ----------------
# Submit SLURM SAMPLE jobs (dependency on 'train') and capture job IDs

model_path="/cluster/work/sachan/NLF/nlf_v3/model/iter_${iteration}/model_ckp_$((iteration))"

sample_v1=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 0 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v2=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 1 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v3=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 2 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v4=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 3 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v5=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 4 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v6=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 5 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v7=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 6 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v8=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 7 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')

# concatenate 8 sampled files (dependency on 'sample_v0..7') and capture job ID
sample_v=$(sbatch --dependency=afterok:$sample_v1:$sample_v2:$sample_v3:$sample_v4:$sample_v5:$sample_v6:$sample_v7:$sample_v8 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${data_split_valid}_output_"{0..7}.json | awk '{print $4}')

# 2. ---------------- FEEDBACK (train) ----------------
# Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_v1=$(sbatch --dependency=afterok:$sample_v tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 0 4 $num_generations_valid | awk '{print $4}')
feedback_v2=$(sbatch --dependency=afterok:$sample_v tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 1 4 $num_generations_valid | awk '{print $4}')
feedback_v3=$(sbatch --dependency=afterok:$sample_v tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 2 4 $num_generations_valid | awk '{print $4}')
feedback_v4=$(sbatch --dependency=afterok:$sample_v tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 3 4 $num_generations_valid | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..3') and capture job ID
feedback_v=$(sbatch --dependency=afterok:$feedback_v1:$feedback_v2:$feedback_v3:$feedback_v4 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${file_prefix_valid}_feedback_subset_"{0..3}.json | awk '{print $4}')

# 5. ---------------- REWARDING (valid) ----------------
# Submit SLURM REWARD jobs (dependency on 'sample_v') and capture job IDs
reward_v1=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 0 8 $num_generations_valid | awk '{print $4}')
reward_v2=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 1 8 $num_generations_valid | awk '{print $4}')
reward_v3=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 2 8 $num_generations_valid | awk '{print $4}')
reward_v4=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 3 8 $num_generations_valid | awk '{print $4}')
reward_v5=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 4 8 $num_generations_valid | awk '{print $4}')
reward_v6=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 5 8 $num_generations_valid | awk '{print $4}')
reward_v7=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 6 8 $num_generations_valid | awk '{print $4}')
reward_v8=$(sbatch --dependency=afterok:$feedback_v tasks/summarization/training/bash_scripts/NLF/NLF_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 7 8 $num_generations_valid | awk '{print $4}')

# concatenate 8 rewarded files (dependency on 'reward_v0..7') and capture job ID
reward_v=$(sbatch --dependency=afterok:$reward_v1:$reward_v2:$reward_v3:$reward_v4:$reward_v5:$reward_v6:$reward_v7:$reward_v8 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${file_prefix_valid}_reward_thread_"{0..7}.json | awk '{print $4}')

# 6. ---------------- PERPLEXITY (valid) ----------------
# Submit SLURM PERPLEXITY job (dependency on 'reward_v')
sbatch --dependency=afterok:$reward_v tasks/summarization/training/bash_scripts/NLF/NLF_perplexity.sh \
    "$config" "$output_dir" "${file_prefix_valid}.json"