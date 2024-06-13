#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=4:00:00
#SBATCH --output="/cluster/work/sachan/NLF/hh_SteerLM/slurm_output/NLF_ppl.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

config="tasks/hh/training/configs/SteerLM_HH_config.yaml"

for iteration in {3..20}
do

	output_dir="/cluster/work/sachan/NLF/hh_SteerLM/output_iter_${iteration}/"
	file_prefix_valid="SteerLM_sampling_data_test_1000subset_iter_${iteration}"

	python tasks/summarization/training/perplexity.py \
    		--config "$config" \
    		--out_dir "$output_dir" \
    		--input_file "${file_prefix_valid}.json"
done

