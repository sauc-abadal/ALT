#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="/cluster/work/sachan/NLF/quarkToNLF/slurm_output/quarkToNLF_ppl_iter_1.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

output_dir=/cluster/work/sachan/NLF/quarkToNLF/output_iter_1/
input_sampling_file=quakr_sampling_data_valid_split_iter_1.json
file_prefix=quark_sampling_data_valid_split_iter_1

echo "--iteration: $iteration"
echo "--input_sampling_file: $input_sampling_file"
echo "--output_dir: $output_dir"

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "${output_dir}/${input_sampling_file}" \
    "${output_dir}/${file_prefix}_reward_thread_"{0..7}.json

# compute perplexities
python tasks/summarization/training/perplexity.py \
    --config tasks/summarization/training/configs/quarkToNLF_TLDR_config.yaml \
    --out_dir $output_dir \
    --input_file $input_sampling_file