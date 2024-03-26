config=tasks/summarization/training/configs/quark_TLDR_config.yaml
input_sampling_file=/cluster/project/sachan/shehzaad/sampling/output/output_0.json
output_dir=/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/stage_1

sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 0 1

