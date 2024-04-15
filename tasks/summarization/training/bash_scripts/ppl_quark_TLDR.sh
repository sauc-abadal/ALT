#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="/cluster/work/sachan/NLF/slurm_output/ppl_quark_iter_4.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

iteration=3
input_sampling_file=/cluster/work/sachan/NLF/output_iter_4/quark_sampling_data_valid_split_iter_4.json
output_dir=/cluster/work/sachan/NLF/output_iter_4/
file_prefix=quark_sampling_data_valid_split_iter_4

echo "--iteration: $iteration"
echo "--input_sampling_file: $input_sampling_file"
echo "--output_dir: $output_dir"

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/${file_prefix}_reward_thread_"{0..7}.json

# compute perplexities
python tasks/summarization/training/quark_eval.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration $iteration
