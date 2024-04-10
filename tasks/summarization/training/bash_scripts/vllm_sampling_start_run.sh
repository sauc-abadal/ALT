#!/bin/bash

#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=160G
#SBATCH --output="/cluster/work/sachan/NLF/slurm_output/vllm_sampling_%j.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

# python tasks/summarization/training/vllm_sampling.py --input_file "$1" --output_dir "$2" --split_number "$3" --total_splits "$4" --model_path "$5" --tokenizer_path "$6" --data_split "$7" --num_generations "$8" --temperature "$9" --top_p "$10" --max_new_tokens "$11"

python tasks/summarization/training/vllm_sampling.py --input_file "/cluster/work/sachan/NLF/sampled_prompts_iter_3.json" --output_dir "/cluster/work/sachan/NLF/output_iter_3" --split_number 0 --total_splits 8 --model_path "/cluster/work/sachan/NLF/model/iter_2/model_ckp_5120" --tokenizer_path "/cluster/work/sachan/NLF/quark_TLDR_5q_tokenizer" --data_split "train" --num_generations 96 --temperature 0.9 --top_p 0.9 --max_new_tokens 64

