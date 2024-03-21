#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=96000
#SBATCH --time=24:00:00
#SBATCH --output="output/quark_training_TLDR_5q_1gpu_sampling_stage_1_debug.out"
#SBATCH --open-mode=append

accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_1gpu_ds_opt_ds_sch_cpu_off.yaml tasks/summarization/training/quark_train_debug.py --config tasks/summarization/training/configs/quark_TLDR_config_debug.yaml --sampling_stage 1 --step_num 0 --ds_optimizer --ds_scheduler
accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_1gpu_ds_opt_ds_sch_cpu_off.yaml tasks/summarization/training/quark_train_debug.py --config tasks/summarization/training/configs/quark_TLDR_config_debug.yaml --sampling_stage 2 --step_num 1500 --ds_optimizer --ds_scheduler
