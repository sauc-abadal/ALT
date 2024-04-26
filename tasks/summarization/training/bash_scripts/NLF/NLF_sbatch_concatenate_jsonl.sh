#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --time=1:00:00
#SBATCH --output="/cluster/work/sachan/NLF/nlf/slurm_output/vllm_sampling_train_jsonl_cat.out"
#SBATCH --open-mode=append

input_sampling_file="/cluster/work/sachan/NLF/nlf/output_iter_2/NLF_sampling_data_train_split_iter_2.json"
output_dir="/cluster/work/sachan/NLF/nlf/output_iter_2/"

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/train_output_"{0..7}.json


