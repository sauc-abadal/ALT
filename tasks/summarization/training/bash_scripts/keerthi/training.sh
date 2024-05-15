#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=6:00:00

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

######## DEFINITION OF ALL VARIABLES ########

# to be manually increased at every iteration
iteration=1

# By setting this variable, the path to the correct accelerate_config file will be automatically changed
num_gpus=2

# set to the path where you cloned the git repository
project_dir_path="/cluster/project/sachan/sauc/nlf"
# set to the same path as in NLF_TLDR_keerthy_config_yaml --> logging: save_dir
save_dir_path="/cluster/work/sachan/NLF/nlf"

config="${project_dir_path}/tasks/summarization/training/bash_scripts/keerthi/NLF_TLDR_keerthi_config.yaml"
# now it's pointing to a 2GPUs DeepSpeed config
accelerate_config="${project_dir_path}/tasks/summarization/training/bash_scripts/keerthi/accelerate_config_ds_${num_gpus}gpu_ds_opt_ds_sch_cpu_off.yaml"

output_dir="${save_dir_path}/output_iter_${iteration}/"
input_sampling_file_train="${save_dir_path}/output_iter_${iteration}/NLF_sampling_data_train_split_iter_${iteration}.json"

if [ "$iteration" -eq 1 ]; then
    model_path="CarperAI/openai_summarize_tldr_sft"
else
    model_path="${save_dir_path}/model/iter_$((iteration-1))/model_ckp_$((iteration-1))"
fi

accelerate launch --config_file "$accelerate_config" tasks/summarization/training/NLF_train_noKL.py \
    --config "$config" \
    --iteration "$iteration" \
    --input_sampling_file "$input_sampling_file_train" \
    --model_path "$model_path" \
    --ds_optimizer \
    --ds_scheduler