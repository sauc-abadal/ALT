config="tasks/summarization/training/configs/NLFv3_TLDR_config.yaml"
input_sampling_file_train=/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_11/quark_sampling_data_train_split_iter_11.json
output_dir=/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_11
num_generations_train=20

file_prefix_train="quark_sampling_data_train_split_iter_11"

# 2. ---------------- FEEDBACK (train) ----------------
# Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_t1=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 0 4 $num_generations_train | awk '{print $4}')
feedback_t2=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 1 4 $num_generations_train | awk '{print $4}')
feedback_t3=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 2 4 $num_generations_train | awk '{print $4}')
feedback_t4=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 3 4 $num_generations_train | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..3') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..3}.json | awk '{print $4}')
