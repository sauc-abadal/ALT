#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:48000
#SBATCH --mem-per-cpu=48g
#SBATCH --time=4:00:00
#SBATCH --output="output/sample_and_loss.out"
#SBATCH --open-mode=append

module load eth_proxy

python -m tasks.summarization.models.test_sampling_and_loss
