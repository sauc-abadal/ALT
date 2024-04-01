#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=96000
#SBATCH --time=24:00:00
#SBATCH --output="output/quark_training_TLDR_5q_1gpu_sampling_stage_1.out"
#SBATCH --open-mode=append

accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_1gpu_ds_opt_ds_sch_cpu_off.yaml tasks/summarization/training/quark_train.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --sampling_stage 1 --ds_optimizer --ds_scheduler
