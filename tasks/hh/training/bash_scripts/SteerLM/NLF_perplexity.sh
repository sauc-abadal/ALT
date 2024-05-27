#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --exclude eu-ts-02
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=96000
#SBATCH --time=2:00:00
#SBATCH --output="/cluster/work/sachan/NLF/hh_SteerLM/slurm_output/NLF_ppl.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

echo "--config: $1"
echo "--output_dir: $2"
echo "--input_sampling_file: $3"

# compute perplexities
python tasks/summarization/training/perplexity.py \
    --config "$1" \
    --out_dir "$2" \
    --input_file "$3"
