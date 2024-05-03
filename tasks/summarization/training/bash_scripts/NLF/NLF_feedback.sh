config="tasks/summarization/training/configs/NLF_TLDR_config.yaml"

num_generations_train=10

iteration=1
output_dir="/cluster/work/sachan/NLF/nlf_v2/output_iter_${iteration}/"
input_sampling_file_train="/cluster/work/sachan/NLF/nlf_v2/output_iter_${iteration}/NLF_sampling_data_train_split_iter_${iteration}.json"
file_prefix_train="NLF_sampling_data_train_split_iter_${iteration}"

# ---------------- FEEDBACK (train) ----------------
# Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_t1=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 0 8 $num_generations_train | awk '{print $4}')
feedback_t2=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 1 8 $num_generations_train | awk '{print $4}')
feedback_t3=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 2 8 $num_generations_train | awk '{print $4}')
feedback_t4=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 3 8 $num_generations_train | awk '{print $4}')
feedback_t5=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 4 8 $num_generations_train | awk '{print $4}')
feedback_t6=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 5 8 $num_generations_train | awk '{print $4}')
feedback_t7=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 6 8 $num_generations_train | awk '{print $4}')
feedback_t8=$(sbatch tasks/summarization/training/bash_scripts/NLF/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 7 8 $num_generations_train | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..7') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4:$feedback_t5:$feedback_t6:$feedback_t7:$feedback_t8 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..7}.json | awk '{print $4}')