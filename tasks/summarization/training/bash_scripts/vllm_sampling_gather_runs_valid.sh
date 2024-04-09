input_file=/cluster/work/sachan/NLF/Q0_conditioned_prompts_valid.json

output_dir=/cluster/work/sachan/NLF/output_iter_2
model_path=/cluster/work/sachan/NLF/model/iter_2/model_ckp_5120
tokenizer_path=/cluster/work/sachan/NLF/quark_TLDR_5q_tokenizer

data_split=valid

num_generations=1
temperature=0.0
top_p=1.0
max_new_tokens=64

# Submit SLURM jobs and capture job IDs
sample1=$(sbatch start_run.sh $input_file $output_dir 0 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample2=$(sbatch start_run.sh $input_file $output_dir 1 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample3=$(sbatch start_run.sh $input_file $output_dir 2 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample4=$(sbatch start_run.sh $input_file $output_dir 3 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample5=$(sbatch start_run.sh $input_file $output_dir 4 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample6=$(sbatch start_run.sh $input_file $output_dir 5 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample7=$(sbatch start_run.sh $input_file $output_dir 6 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')
sample8=$(sbatch start_run.sh $input_file $output_dir 7 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens | awk '{print $4}')

# individual jsonl files named saved in f"{args.output_dir}/{args.data_split}_output_{args.split_number}.json"

# the files can be concatenated without the need of adding a newline in between, as "\n" is already included 
# at the end of every line.

# Submit reward_gather_runs_valid.sh after all jobs complete
sbatch --dependency=afterok:$sample1:$sample2:$sample3:$sample4:$sample5:$sample6:$sample7:$sample8 reward_gather_runs_valid.sh
