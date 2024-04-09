#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_full_eval_TLDR_5q_v6_iter_2.out"
#SBATCH --open-mode=append

# accelerate launch --config_file /cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_1gpu.yaml tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration 2 --split valid --model_path /cluster/work/sachan/NLF/model/iter_2/model_ckp_5120

# mv /cluster/work/sachan/NLF/output_iter_2/quark_sampling_data_valid_split_iter_2_worker_0.json /cluster/work/sachan/NLF/output_iter_2/quark_sampling_data_valid_split_iter_2.json
# python tasks/summarization/training/reward.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --input_sampling_file /cluster/work/sachan/NLF/output_iter_2/quark_sampling_data_valid_split_iter_2.json --output_dir /cluster/work/sachan/NLF/output_iter_2 --split_number 0 --total_splits 1 --num_generations 1

mv /cluster/work/sachan/NLF/output_iter_2/quark_sampling_data_valid_split_iter_2_reward_thread_0.json /cluster/work/sachan/NLF/output_iter_2/quark_sampling_data_valid_split_iter_2.json
python tasks/summarization/training/quark_eval.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration 2
