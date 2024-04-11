#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --mem-per-cpu=160000
#SBATCH --time=12:00:00
#SBATCH --output="/cluster/work/sachan/NLF/slurm_output/quark_training_TLDR_5q_v6_noKL_2gpu_iter_4.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml
yaml_config=tasks/summarization/training/configs/quark_TLDR_config.yaml

model_path=/cluster/work/sachan/NLF/model/iter_3/model_ckp_7680

iteration=3
input_sampling_file=/cluster/work/sachan/NLF/output_iter_4/quark_sampling_data_train_split_iter_4.json
output_dir=/cluster/work/sachan/NLF/output_iter_4/
file_prefix=quark_sampling_data_train_split_iter_4

echo "--iteration: $iteration"
echo "--input_sampling_file: $input_sampling_file"
echo "--output_dir: $output_dir"

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh \
    "$input_sampling_file" \
    "${output_dir}/${file_prefix}_reward_thread_"{0..7}.json

# launch training
accelerate launch --config_file $accelerate_config tasks/summarization/training/quark_train_noKL.py \
    --config $yaml_config \
    --iteration $iteration \
    --input_sampling_file $input_sampling_file \
    --model_path $model_path \
    --ds_optimizer \
    --ds_scheduler

# launch evaluation 
bash tasks/summarization/training/bash_scripts/vllm_sampling_gather_runs_valid.sh