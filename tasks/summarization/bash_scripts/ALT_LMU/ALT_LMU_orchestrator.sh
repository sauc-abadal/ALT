config="tasks/summarization/configs/ALT_LMU_TLDR_config.yaml"
accelerate_config="tasks/summarization/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml"

# to be downloaded, initialize its attributes, and saved in this path!
tokenizer_path="path_to_tokenizer"

num_generations_train=20
data_split_train=train
temperature_train=0.9
top_p_train=0.9
max_new_tokens_train=64

num_generations_valid=1
data_split_valid=valid
temperature_valid=0.0
top_p_valid=1.0
max_new_tokens_valid=64

# to be manually increased at every iteration
iteration=1

# train prompts for iter 1 are unconditioned, but for iter 2 onwards they should be conditioned on the exemplar feedback, e.g., "feedback: {feedback}. input: {prompt}", where {feedback} mirrors GPT-3.5 feedback on aligned samples
input_prompts_file_train="path_to_train_prompts/sampled_prompts_iter_${iteration}.json"
# valid prompt should be conditioned on the exemplar feedback, e.g., "feedback: {feedback}. input: {prompt}", where {feedback} mirrors GPT-3.5 feedback on aligned samples
input_prompts_file_valid="path_to_valid_prompts/conditioned_prompts_valid.json"

output_dir="save_path/output_iter_${iteration}/"

input_sampling_file_train="save_path/output_iter_${iteration}/ALT_LMU_sampling_data_train_split_iter_${iteration}.json"
input_sampling_file_valid="save_path/output_iter_${iteration}/ALT_LMU_sampling_data_valid_split_iter_${iteration}.json"

file_prefix_train="ALT_LMU_sampling_data_train_split_iter_${iteration}"
file_prefix_valid="ALT_LMU_sampling_data_valid_split_iter_${iteration}"

if [ "$iteration" -eq 1 ]; then
    model_path="CarperAI/openai_summarize_tldr_sft"
else
    model_path="save_path/model/iter_$((iteration-1))/model_ckp_$((iteration-1))"
fi

# 1. ---------------- SAMPLING (train) ----------------
# Submit SLURM SAMPLE jobs (no dependency) and capture job IDs
sample_t1=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 0 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t2=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 1 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t3=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 2 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t4=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 3 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t5=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 4 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t6=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 5 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t7=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 6 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t8=$(sbatch tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 7 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')

# concatenate 8 sampled files (dependency on 'sample_t0..7') and capture job ID
sample_t=$(sbatch --dependency=afterok:$sample_t1:$sample_t2:$sample_t3:$sample_t4:$sample_t5:$sample_t6:$sample_t7:$sample_t8 tasks/summarization/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${data_split_train}_output_"{0..7}.json | awk '{print $4}')

# # 2. ---------------- FEEDBACK (train) ----------------
# # Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_t1=$(sbatch --dependency=afterok:$sample_t tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 0 4 $num_generations_train | awk '{print $4}')
feedback_t2=$(sbatch --dependency=afterok:$sample_t tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 1 4 $num_generations_train | awk '{print $4}')
feedback_t3=$(sbatch --dependency=afterok:$sample_t tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 2 4 $num_generations_train | awk '{print $4}')
feedback_t4=$(sbatch --dependency=afterok:$sample_t tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 3 4 $num_generations_train | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..3') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4 tasks/summarization/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..3}.json | awk '{print $4}')

# 3. ---------------- TRAINING ----------------
# Submit SLURM TRAIN job (dependency on 'reward_t') and capture job ID
train=$(sbatch --dependency=afterok:$feedback_t tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_train_start_run_2gpus.sh \
    "$accelerate_config" "$config" "$iteration" "$input_sampling_file_train" "$model_path" | awk '{print $4}')

# 4. ---------------- SAMPLING (valid) ----------------
# Submit SLURM SAMPLE jobs (dependency on 'train') and capture job IDs

model_path="save_path/model/iter_${iteration}/model_ckp_$((iteration))"

sample_v1=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 0 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v2=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 1 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v3=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 2 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v4=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 3 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v5=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 4 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v6=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 5 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v7=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 6 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')
sample_v8=$(sbatch --dependency=afterok:$train tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_vllm_sampling_start_run.sh "$input_prompts_file_valid" "$output_dir" 7 8 "$model_path" "$tokenizer_path" "$data_split_valid" $num_generations_valid $temperature_valid $top_p_valid $max_new_tokens_valid | awk '{print $4}')

# concatenate 8 sampled files (dependency on 'sample_v0..7') and capture job ID
sample_v=$(sbatch --dependency=afterok:$sample_v1:$sample_v2:$sample_v3:$sample_v4:$sample_v5:$sample_v6:$sample_v7:$sample_v8 tasks/summarization/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${data_split_valid}_output_"{0..7}.json | awk '{print $4}')

# 5. ---------------- REWARDING (valid) ----------------
# Submit SLURM REWARD jobs (dependency on 'sample_v') and capture job IDs
reward_v1=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 0 8 $num_generations_valid | awk '{print $4}')
reward_v2=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 1 8 $num_generations_valid | awk '{print $4}')
reward_v3=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 2 8 $num_generations_valid | awk '{print $4}')
reward_v4=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 3 8 $num_generations_valid | awk '{print $4}')
reward_v5=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 4 8 $num_generations_valid | awk '{print $4}')
reward_v6=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 5 8 $num_generations_valid | awk '{print $4}')
reward_v7=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 6 8 $num_generations_valid | awk '{print $4}')
reward_v8=$(sbatch --dependency=afterok:$sample_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_reward_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 7 8 $num_generations_valid | awk '{print $4}')

# concatenate 8 rewarded files (dependency on 'reward_v0..7') and capture job ID
reward_v=$(sbatch --dependency=afterok:$reward_v1:$reward_v2:$reward_v3:$reward_v4:$reward_v5:$reward_v6:$reward_v7:$reward_v8 tasks/summarization/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${file_prefix_valid}_reward_thread_"{0..7}.json | awk '{print $4}')

# 6. ---------------- PERPLEXITY (valid) ----------------
# Submit SLURM PERPLEXITY job (dependency on 'reward_v')
sbatch --dependency=afterok:$reward_v tasks/summarization/bash_scripts/ALT_LMU/ALT_LMU_perplexity.sh \
    "$config" "$output_dir" "${file_prefix_valid}.json"