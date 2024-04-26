#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_80gb:2
#SBATCH --exclude eu-ts-02
#SBATCH --mem-per-cpu=160000
#SBATCH --time=6:00:00
#SBATCH --output="/cluster/work/sachan/NLF/quarkToNLF_v2/slurm_output/quarkToNLF_training_TLDR_v1_noKL_2gpu.out"
#SBATCH --open-mode=append
#SBATCH --mail-type=END

source /cluster/project/sachan/sauc/anaconda3/bin/activate nlf_gptj

accelerate_config=/cluster/project/sachan/sauc/nlf/tasks/summarization/training/configs/accelerate_config_ds_2gpu_ds_opt_ds_sch_cpu_off.yaml
yaml_config=tasks/summarization/training/configs/quarkToNLF_TLDR_config.yaml

model_path=CarperAI/openai_summarize_tldr_sft

iteration=1
input_sampling_file=/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_1/quark_sampling_data_train_split_iter_1.json
output_dir=/cluster/work/sachan/NLF/quarkToNLF_v2/output_iter_1/

echo "--accelerate_config: $1"
echo "--yaml_config: $2"
echo "--iteration: $3"
echo "--input_sampling_file: $4"
echo "--model_path: $5"

accelerate launch --config_file "$1" tasks/summarization/training/quarkToNLF_train_noKL.py \
    --config "$2" \
    --iteration "$3" \
    --input_sampling_file "$4" \
    --model_path "$5" \
    --ds_optimizer \
    --ds_scheduler