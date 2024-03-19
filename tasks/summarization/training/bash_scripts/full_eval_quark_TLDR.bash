#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_full_eval_TLDR_5q_sampling_stage_1.out"
#SBATCH --open-mode=append

mv /cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/quark_sampling_data_valid_stage_1_worker_0.json /cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/quark_sampling_data_valid_stage_1.json
python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --split valid
python tasks/summarization/training/quark_eval.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml
