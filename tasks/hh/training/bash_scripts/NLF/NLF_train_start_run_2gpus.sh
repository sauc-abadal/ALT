#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --exclude eu-ts-02
#SBATCH --mem-per-cpu=160000
#SBATCH --time=4:00:00
#SBATCH --output="/cluster/work/sachan/NLF/hh_nlf/slurm_output/NLF_training_HH_v0_noKL_2gpu.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate sample

echo "--accelerate_config: $1"
echo "--yaml_config: $2"
echo "--iteration: $3"
echo "--input_sampling_file: $4"
echo "--model_path: $5"

accelerate launch --config_file "$1" tasks/hh/training/NLF_train_noKL.py \
    --config "$2" \
    --iteration "$3" \
    --input_sampling_file "$4" \
    --model_path "$5" \
    --ds_optimizer \
    --ds_scheduler
