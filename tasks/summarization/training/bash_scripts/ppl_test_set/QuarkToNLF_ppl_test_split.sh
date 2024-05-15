#!/bin/bash

config="tasks/summarization/training/configs/NLF_TLDR_config.yaml"
output_dir="/cluster/work/sachan/NLF/CarperAI_test_prompts/quarkToNLF_v2_iter_10"
test_file="quark_sampling_data_test_split_iter_10.json"

sbatch tasks/summarization/training/bash_scripts/NLF/NLF_perplexity.sh \
    "$config" "$output_dir" "${test_file}"