#!/bin/bash

#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=160G
#SBATCH --output="/cluster/work/sachan/NLF/quark/slurm_output/vllm_sampling_%j.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

echo "--input_file: $1"
echo "--output_dir: $2"
echo "--split_number: $3"
echo "--total_splits: $4"
echo "--model_path: $5"
echo "--tokenizer_path: $6"
echo "--data_split: $7"
echo "--num_generations: $8"
echo "--temperature: $9"
echo "--top_p: ${10}"
echo "--max_new_tokens: ${11}"

python tasks/summarization/training/vllm_sampling.py \
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