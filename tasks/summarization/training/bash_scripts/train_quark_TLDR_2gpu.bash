#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --mem-per-cpu=160000
#SBATCH --time=12:00:00
#SBATCH --output="output/quark_training_TLDR_5q_v6_noKL_2gpu_iter_3.out"
#SBATCH --open-mode=append
#SBATCH --mail-type='END'

accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml tasks/summarization/training/quark_train_noKL.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration 3 --input_sampling_file /cluster/work/sachan/NLF/output_iter_3/quark_sampling_data_train_split_iter_3.json --model_path /cluster/work/sachan/NLF/model/iter_2/model_ckp_5120 --ds_optimizer --ds_scheduler
