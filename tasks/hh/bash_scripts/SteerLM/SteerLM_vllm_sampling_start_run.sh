#!/bin/bash

#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=160G

source path_to_anaconda/anaconda3/bin/activate sample

python alt/vllm_sampling.py \
    --input_file "$1" \
    --output_dir "$2" \
    --split_number "$3" \
    --total_splits "$4" \
    --model_path "$5" \
    --tokenizer_path "$6" \
    --data_split "$7" \
    --num_generations "$8" \
    --temperature "$9" \
    --top_p "${10}" \
    --max_new_tokens "${11}"