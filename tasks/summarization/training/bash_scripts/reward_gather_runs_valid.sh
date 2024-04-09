config=tasks/summarization/training/configs/quark_TLDR_config.yaml
input_sampling_file=/cluster/work/sachan/NLF/output_iter_2/quark_sampling_data_valid_split_iter_2.json
output_dir=/cluster/work/sachan/NLF/output_iter_2/

num_generations=1

# concatenate previously sampled jsonl files (8 threads) into a single jsonl file
bash tasks/summarization/training/bash_scripts/concatenate_jsonl.sh $input_sampling_file $output_dir/valid_output_0.json $output_dir/valid_output_1.json $output_dir/valid_output_2.json $output_dir/valid_output_3.json $output_dir/valid_output_4.json $output_dir/valid_output_5.json $output_dir/valid_output_6.json $output_dir/valid_output_7.json 

sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 0 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 1 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 2 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 3 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 4 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 5 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 6 8 $num_generations
sbatch tasks/summarization/training/bash_scripts/start_run_reward.sh $config $input_sampling_file $output_dir 7 8 $num_generations

