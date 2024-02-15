#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:48g
#SBATCH --mem-per-cpu=48000
#SBATCH --time=2:00:00
#SBATCH --output="output/quark_TLDR_5q.out"
#SBATCH --open-mode=append

module load eth_proxy

# for i in range(1, freq_exploration):
python tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter True --split train
python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/train_quark_TLDR_config.yml --split train
python tasks/summarization/training/quark_train.py --config tasks/summarization/training/train_quark_TLDR_config.yml
# eval 
python tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter False --split valid
python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/train_quark_TLDR_config.yml --split valid
python tasks/summarization/training/eval.py --config tasks/summarization/training/train_quark_TLDR_config.yml
