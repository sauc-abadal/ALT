#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=4:00:00

# You may increase the number of GPUs and vLLM will automatically handle it (no config needed)

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

######## DEFINITION OF ALL VARIABLES ########

# to be manually increased at every iteration
iteration=1

# set to the path where you cloned the git repository
project_dir_path="/cluster/project/sachan/sauc/nlf"
# set to the same path as in NLF_TLDR_keerthi_config_yaml --> logging: save_dir
save_dir_path="/cluster/work/sachan/NLF/nlf"

output_dir="${save_dir_path}/output_iter_${iteration}/"
# place a priori the prompts json files (for each iteration) in save_dir_path 
input_prompts_file_train="${save_dir_path}/sampled_prompts_iter_${iteration}.json"
input_sampling_file_train="${save_dir_path}/output_iter_${iteration}/NLF_sampling_data_train_split_iter_${iteration}.json"

tokenizer_path="${save_dir_path}/NLF_TLDR_tokenizer"

num_generations_train=20
data_split_train=train
temperature_train=0.9
top_p_train=0.9
max_new_tokens_train=64

if [ "$iteration" -eq 1 ]; then
    model_path="CarperAI/openai_summarize_tldr_sft"
else
    model_path="${save_dir_path}/model/iter_$((iteration-1))/model_ckp_$((iteration-1))"
fi

######## SCRIPT LAUNCHING ########

python tasks/summarization/training/vllm_sampling.py \
    --input_file "$input_prompts_file_train" \
    --output_dir "$output_dir" \
    --split_number 0 \
    --total_splits 1 \
    --model_path "$model_path" \
    --tokenizer_path "$tokenizer_path" \
    --data_split "$data_split_train" \
    --num_generations "$num_generations_train" \
    --temperature "$temperature_train" \
    --top_p "${top_p_train}" \
    --max_new_tokens "${max_new_tokens_train}"

mv "${output_dir}/${data_split_train}_output_0.json" "$input_sampling_file_train"




