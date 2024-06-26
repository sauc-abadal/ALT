#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00

source path_to_anaconda/anaconda3/bin/activate sample

python alt/perplexity.py \
    --config "$1" \
    --out_dir "$2" \
    --input_file "$3"
