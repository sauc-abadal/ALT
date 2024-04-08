#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=1:00:00
#SBATCH --output="output/quark_full_eval_TLDR_5q_v6_iter_1.out"
#SBATCH --open-mode=append

python tasks/summarization/training/quark_eval.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml --iteration 1
