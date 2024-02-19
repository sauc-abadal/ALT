#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=48000
#SBATCH --time=4:00:00
#SBATCH --output="output/quark_TLDR_5q.out"
#SBATCH --open-mode=append

module load eth_proxy


# Specify the number of iterations
freq_exploration=2 

python set_wandb_run_id.py --config tasks/summarization/training/train_quark_TLDR_config.yml

for i in $(seq 1 $freq_exploration); do

  # Conditional argument for first iteration
  if [[ $i -eq 1 ]]; then
    python tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter True --split train
  else
    python tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter False --split train
  fi

  python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter True --split train
  accelerate launch --config_file tasks/summarization/training/default_accelerate_config.yaml tasks/summarization/training/quark_train.py --config tasks/summarization/training/train_quark_TLDR_config.yml

  # Evaluation (unchanged)
  python tasks/summarization/training/quark_sampling.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter False --split valid
  python tasks/summarization/training/quark_reward.py --config tasks/summarization/training/train_quark_TLDR_config.yml --first_iter False --split valid
  python tasks/summarization/training/quark_eval.py --config tasks/summarization/training/train_quark_TLDR_config.yml
done
