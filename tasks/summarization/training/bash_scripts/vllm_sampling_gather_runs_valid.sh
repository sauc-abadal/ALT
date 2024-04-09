input_file=/cluster/work/sachan/NLF/Q0_conditioned_prompts_valid.json

output_dir=/cluster/work/sachan/NLF/output_iter_2
model_path=/cluster/work/sachan/NLF/model/iter_2/model_ckp_5120
tokenizer_path=/cluster/work/sachan/NLF/quark_TLDR_5q_tokenizer

data_split=valid

num_generations=1
temperature=0.0
top_p=1.0
max_new_tokens=64

# split input file into several files
sbatch start_run.sh $input_file $output_dir 0 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 1 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 2 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 3 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 4 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 5 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 6 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens
sbatch start_run.sh $input_file $output_dir 7 8 $model_path $tokenizer_path $data_split $num_generations $temperature $top_p $max_new_tokens

# individual jsonl files named saved in f"{args.output_dir}/{args.data_split}_output_{args.split_number}.json"

# the files can be concatenated without the need of adding a newline in between, as "\n" is already included 
# at the end of every line.