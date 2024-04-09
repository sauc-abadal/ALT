config=tasks/summarization/training/configs/quark_TLDR_config.yaml
input_sampling_file=/cluster/work/sachan/NLF/output_iter_3/quark_sampling_data_train_split_iter_3.json
output_dir=/cluster/work/sachan/NLF/output_iter_3/

num_generations=96

sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 0 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 1 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 2 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 3 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 4 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 5 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 6 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 7 8 $num_generations

