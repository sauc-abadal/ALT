#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=6:00:00

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

######## DEFINITION OF ALL VARIABLES ########

# to be manually increased at every iteration
iteration=1

# set to the path where you cloned the git repository
project_dir_path="/cluster/project/sachan/sauc/nlf"
# set to the same path as in NLF_TLDR_keerthi_config_yaml --> logging: save_dir
save_dir_path="/cluster/work/sachan/NLF/nlf"

output_dir="${save_dir_path}/output_iter_${iteration}/"
input_sampling_file_train="${save_dir_path}/output_iter_${iteration}/NLF_sampling_data_train_split_iter_${iteration}.json"
file_prefix_train=NLF_sampling_data_train_split_iter_${iteration}
num_generations_train=48

######## SCRIPT LAUNCHING ########

# concatenate 8 feedback files (dependency on 'feedback_t0..3') and capture job ID
feedback_t=$(sbatch --dependency=afterok:$feedback_t1:$feedback_t2:$feedback_t3:$feedback_t4 tasks/summarization/training/bash_scripts/sbatch_concatenate_jsonl.sh \
    "$input_sampling_file_train" \
    "${output_dir}/${file_prefix_train}_feedback_subset_"{0..3}.json | awk '{print $4}')

# you may need to modifiy the feedback_keerthi.py script
python tasks/summarization/training/feedback_keerthi.py \
    --config "$config" \
    --input_sampling_file "$input_sampling_file_train" \
    --output_dir "$output_dir" \
    --split_number 0 \
    --total_splits 1 \
    --num_generations "$num_generations_train" \
    --NLF

mv "${output_dir}/${file_prefix_train}_feedback_subset_0.json" "$input_sampling_file_train"
