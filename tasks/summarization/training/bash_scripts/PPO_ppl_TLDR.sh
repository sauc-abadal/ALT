#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="/cluster/work/sachan/NLF/slurm_output/ppl_PPO.out"
#SBATCH --open-mode=append

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

# compute perplexities
python tasks/summarization/training/PPO_eval.py --config tasks/summarization/training/configs/quark_TLDR_config.yaml
