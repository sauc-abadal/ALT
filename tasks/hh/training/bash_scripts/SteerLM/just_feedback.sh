config="tasks/hh/training/configs/SteerLM_HH_config.yaml"
accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/hh/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml

# to be initialized and saved in this path!
tokenizer_path="/cluster/work/sachan/NLF/hh_nlf/tokenizer-pythia-2.8b-mitchell-sft_hh_rlhf"

num_generations_train=20
data_split_train=train
temperature_train=1.0
top_p_train=0.9
max_new_tokens_train=256

iteration=1

if [ "$iteration" -eq 1 ]; then
    og_input_prompts_file="/cluster/work/sachan/NLF/hh_SteerLM/HH_train_prompts.json"
else
    og_input_prompts_file="/cluster/work/sachan/NLF/hh_SteerLM/HH_train_prompts_conditioned.json"
fi

input_prompts_file_train=/cluster/work/sachan/NLF/hh_SteerLM/sampled_prompts_iter_${iteration}.json
shuf -n 2048 $og_input_prompts_file > $input_prompts_file_train

output_dir="/cluster/work/sachan/NLF/hh_SteerLM/output_iter_${iteration}/"

input_sampling_file_train="/cluster/work/sachan/NLF/hh_SteerLM/output_iter_${iteration}/SteerLM_sampling_data_train_split_iter_${iteration}.json"
file_prefix_train="SteerLM_sampling_data_train_split_iter_${iteration}"

if [ "$iteration" -eq 1 ]; then
    model_path="mnoukhov/pythia-2.8b-sft_hh_rlhf"
else
    model_path="/cluster/work/sachan/NLF/hh_SteerLM/model/iter_$((iteration-1))/model_ckp_$((iteration-1))"
fi

# 1. ---------------- FEEDBACK (train) ----------------
# Submit SLURM FEEDBACK jobs (dependency on 'sample_t') and capture job IDs
feedback_t1=$(sbatch tasks/hh/training/bash_scripts/SteerLM/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 0 4 $num_generations_train | awk '{print $4}')
feedback_t2=$(sbatch tasks/hh/training/bash_scripts/SteerLM/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 1 4 $num_generations_train | awk '{print $4}')
feedback_t3=$(sbatch tasks/hh/training/bash_scripts/SteerLM/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 2 4 $num_generations_train | awk '{print $4}')
feedback_t4=$(sbatch tasks/hh/training/bash_scripts/SteerLM/NLF_feedback_start_run.sh "$config" "$input_sampling_file_train" "$output_dir" 3 4 $num_generations_train | awk '{print $4}')

# concatenate 8 feedback files (dependency on 'feedback_t0..3') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4 tasks/hh/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..3}.json | awk '{print $4}')
