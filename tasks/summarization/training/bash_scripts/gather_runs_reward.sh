config=tasks/summarization/training/configs/quark_TLDR_config_debug.yaml
input_sampling_file=/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/stage_2/quark_sampling_data_train_stage_2.json
output_dir=/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/stage_2

sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 0 2
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 1 2

