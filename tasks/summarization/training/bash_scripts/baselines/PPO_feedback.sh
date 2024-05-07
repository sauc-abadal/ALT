config="tasks/summarization/training/configs/NLF_TLDR_config.yaml"

num_generations_train=1

output_dir="/cluster/work/sachan/NLF/CarperAI_test_prompts/PPO/"
input_sampling_file_train="/cluster/work/sachan/NLF/CarperAI_test_prompts/PPO/PPO_sampling_data_valid_split_100random_subset.json"
file_prefix_train="PPO_sampling_data_valid_split_100random_subset"

# ---------------- FEEDBACK (train) ----------------
# Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_t1=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 0 4 $num_generations_train | awk '{print $4}')
feedback_t2=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 1 4 $num_generations_train | awk '{print $4}')
feedback_t3=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 2 4 $num_generations_train | awk '{print $4}')
feedback_t4=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 3 4 $num_generations_train | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..7') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..3}.json | awk '{print $4}')