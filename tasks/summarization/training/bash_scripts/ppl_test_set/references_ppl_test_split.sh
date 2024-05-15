#!/bin/bash

config="tasks/summarization/training/configs/NLF_TLDR_config.yaml"
output_dir="/cluster/work/sachan/NLF/CarperAI_test_prompts/"
test_file="TLDR_test_split_prompts.json"

sbatch tasks/summarization/training/bash_scripts/NLF/NLF_perplexity.sh \
    "$config" "$output_dir" "${test_file}"