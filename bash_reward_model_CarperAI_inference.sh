#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:48g
#SBATCH --mem-per-cpu=48000
#SBATCH --time=2:00:00
#SBATCH --output="output/rm_CarperAI_inference.out"
#SBATCH --open-mode=append

module load eth_proxy

python -m tasks.summarization.models.test_reward_CarperAI_inference
