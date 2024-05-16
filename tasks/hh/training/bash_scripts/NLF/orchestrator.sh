config="tasks/hh/training/configs/NLF_HH_config.yaml"
accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/hh/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml

# to be initialized and saved in this path!
tokenizer_path="/cluster/work/sachan/NLF/hh_nlf/tokenizer-pythia-2.8b-mitchell-sft_hh_rlhf"

num_generations_train=20
data_split_train=train
temperature_train=1.0
top_p_train=0.9
max_new_tokens_train=256

num_generations_valid=1
data_split_valid=valid
temperature_valid=1.0
top_p_valid=0.9
max_new_tokens_valid=256

iteration=3

if [ "$iteration" -eq 1 ]; then
    og_input_prompts_file="/cluster/work/sachan/NLF/hh_nlf/HH_train_prompts.json"
else
    og_input_prompts_file="/cluster/work/sachan/NLF/hh_nlf/HH_train_prompts_conditioned.json"
fi

input_prompts_file_train=/cluster/work/sachan/NLF/hh_nlf/sampled_prompts_iter_${iteration}.json
shuf -n 2048 $og_input_prompts_file > $input_prompts_file_train

input_prompts_file_valid="/cluster/work/sachan/NLF/hh_nlf/HH_test_prompts_1000subset.json"

output_dir="/cluster/work/sachan/NLF/hh_nlf/output_iter_${iteration}/"

input_sampling_file_train="/cluster/work/sachan/NLF/hh_nlf/output_iter_${iteration}/NLF_sampling_data_train_split_iter_${iteration}.json"
input_sampling_file_valid="/cluster/work/sachan/NLF/hh_nlf/output_iter_${iteration}/NLF_sampling_data_valid_split_iter_${iteration}.json"
file_prefix_train="NLF_sampling_data_train_split_iter_${iteration}"
file_prefix_valid="NLF_sampling_data_valid_split_iter_${iteration}"

if [ "$iteration" -eq 1 ]; then
    model_path="mnoukhov/pythia-2.8b-sft_hh_rlhf"
else
    model_path="/cluster/work/sachan/NLF/hh_nlf/model/iter_$((iteration-1))/model_ckp_$((iteration-1))"
fi

# 1. ---------------- SAMPLING (train) ----------------
# Submit SLURM SAMPLE jobs (no dependency) and capture job IDs
sample_t1=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 0 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t2=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 1 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t3=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 2 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t4=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 3 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t5=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 4 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t6=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 5 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t7=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 6 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')
sample_t8=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_vllm_sampling_start_run.sh "$input_prompts_file_train" "$output_dir" 7 8 "$model_path" "$tokenizer_path" "$data_split_train" $num_generations_train $temperature_train $top_p_train $max_new_tokens_train | awk '{print $4}')

# concatenate 8 sampled files (dependency on 'sample_t0..7') and capture job ID
sample_t=$(sbatch --dependency=afterok:$sample_t1:$sample_t2:$sample_t3:$sample_t4:$sample_t5:$sample_t6:$sample_t7:$sample_t8 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${data_split_train}_output_"{0..7}.json | awk '{print $4}')

# # 2. ---------------- FEEDBACK (train) ----------------
# # Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_t1=$(sbatch --dependency=afterok:$sample_t tasks/hh/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 0 4 $num_generations_train | awk '{print $4}')
feedback_t2=$(sbatch --dependency=afterok:$sample_t tasks/hh/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 1 4 $num_generations_train | awk '{print $4}')
feedback_t3=$(sbatch --dependency=afterok:$sample_t tasks/hh/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 2 4 $num_generations_train | awk '{print $4}')
feedback_t4=$(sbatch --dependency=afterok:$sample_t tasks/hh/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 3 4 $num_generations_train | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..3') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4 tasks/hh/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..3}.json | awk '{print $4}')

# 3. ---------------- TRAINING ----------------
# Submit SLURM TRAIN job (dependency on 'reward_t') and capture job ID
train=$(sbatch --dependency=afterok:$feedback_t tasks/hh/training/bash_scripts/NLF/NLF_train_start_run_2gpus.sh \
    "$accelerate_config" "$config" "$iteration" "$input_sampling_file_train" "$model_path" | awk '{print $4}')
