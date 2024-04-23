#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --exclude eu-ts-02
#SBATCH --mem-per-cpu=160000
#SBATCH --time=12:00:00
#SBATCH --output="/cluster/work/sachan/NLF/nlf/slurm_output/NLF_training_TLDR_v1_noKL_2gpu_iter_2.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml
yaml_config=tasks/summarization/training/configs/NLF_TLDR_config.yaml

model_path=/cluster/work/sachan/NLF/nlf/model/iter_1/model_ckp_2560

iteration=2
input_sampling_file=/cluster/work/sachan/NLF/nlf/output_iter_2/NLF_sampling_data_train_split_iter_2.json
output_dir=/cluster/work/sachan/NLF/nlf/output_iter_2/

echo "--iteration: $iteration"
echo "--input_sampling_file: $input_sampling_file"
echo "--output_dir: $output_dir"

# concatenate previously sampled jsonl files (6 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/${file_prefix}_feedback_subset_"{0..5}.json

# launch training
accelerate launch --config_file $accelerate_config tasks/summarization/training/NLF_train_noKL.py \
    --config $yaml_config \
    --iteration $iteration \
    --input_sampling_file $input_sampling_file \
    --model_path $model_path \
    --ds_optimizer \
    --ds_scheduler

# launch evaluation 
bash tasks/summarization/training/bash_scripts/NLF_vllm_sampling_gather_runs_valid.sh