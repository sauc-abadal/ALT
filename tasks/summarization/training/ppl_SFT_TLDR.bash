#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=1:00:00
#SBATCH --output="output/SFT_ppl_TLDR.out"
#SBATCH --open-mode=append

python tasks/summarization/training/ref_or_SFT_ppl.py --config tasks/summarization/training/train_quark_TLDR_single_GPU_config.yml --references False
