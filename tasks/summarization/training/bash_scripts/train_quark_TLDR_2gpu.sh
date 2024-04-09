#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --mem-per-cpu=160000
#SBATCH --time=12:00:00
#SBATCH --output="output/quark_training_TLDR_5q_v6_noKL_2gpu_iter_3.out"
#SBATCH --open-mode=append

conda activate nlf_gptj

accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml
yaml_config=tasks/summarization/training/configs/quark_TLDR_config.yaml
iteration=3
input_sampling_file=/cluster/work/sachan/NLF/output_iter_3/quark_sampling_data_train_split_iter_3.json
model_path=/cluster/work/sachan/NLF/model/iter_2/model_ckp_5120

output_dir=/cluster/work/sachan/NLF/output_iter_3/

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh $input_sampling_file $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_0.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_1.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_2.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_3.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_4.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_5.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_6.json $output_dir/quark_sampling_data_train_split_iter_3_reward_thread_7.json 

accelerate launch --config_file $accelerate_config tasks/summarization/training/quark_train_noKL.py --config $yaml_config --iteration $iteration --input_sampling_file $input_sampling_file --model_path $model_path --ds_optimizer --ds_scheduler
