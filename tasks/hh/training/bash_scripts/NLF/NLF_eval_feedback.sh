config="tasks/hh/training/configs/NLF_HH_config.yaml"

num_generations_valid=1
data_split_valid=test_1000subset
temperature_valid=1.0
top_p_valid=0.9
max_new_tokens_valid=256


output_dir="/cluster/work/sachan/NLF/hh_nlf/output_iter_20/"
input_sampling_file_valid="/cluster/work/sachan/NLF/hh_nlf/output_iter_20/NLF_sampling_data_test_1000subset_iter_20.json"
file_prefix_valid="NLF_sampling_data_test_1000subset_iter_20"


# 1. ---------------- EVAL FEEDBACK (valid) ----------------
# Submit SLURM FEEDBACK jobs (dependency on 'sample_v') and capture job IDs
feedback_v1=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_eval_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 0 2 $num_generations_valid | awk '{print $4}')
feedback_v2=$(sbatch tasks/hh/training/bash_scripts/NLF/NLF_eval_feedback_start_run.sh "$config" "$input_sampling_file_valid" "$output_dir" 1 2 $num_generations_valid | awk '{print $4}')

# concatenate 2 feedback files (dependency on 'feedback_v0..1') and capture job ID
feedback_v=$(sbatch --dependency=afterok:$feedback_v1:$feedback_v2 tasks/hh/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_valid" \
    "${output_dir}/${file_prefix_valid}_eval_feedback_subset_"{0..1}.json | awk '{print $4}')
