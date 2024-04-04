config=tasks/summarization/training/configs/quark_TLDR_config.yaml
input_sampling_file=/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/quark_sampling_data_train_split_stage_1_chunk_0_subset.json
output_dir=/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/

sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 0 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 1 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 2 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 3 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 4 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 5 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 6 8
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 7 8

