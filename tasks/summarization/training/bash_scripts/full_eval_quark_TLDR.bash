#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_full_eval_TLDR_5q_v6_iter_1.out"
#SBATCH --open-mode=append

accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_1gpu.yaml tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration 1 --split valid --model_path /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/model/model_ckp_2560

mv /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/sampling/iter_1/quark_sampling_data_valid_split_iter_1_worker_0.json /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/sampling/iter_1/quark_sampling_data_valid_split_iter_1.json
python tasks/summarization/training/quark_reward_1gen.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --input_sampling_file /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/sampling/iter_1/quark_sampling_data_valid_split_iter_1.json --output_dir /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/sampling/iter_1 --split_number 0 --total_splits 1

mv /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/sampling/iter_1/quark_sampling_data_valid_split_iter_1_worker_0.json /cluster/work/sachan/sauc/nlf/quark_TLDR_5q_v6/sampling/iter_1/quark_sampling_data_valid_split_iter_1.json
python tasks/summarization/training/quark_eval.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration 1
