from vllm import LLM, SamplingParams
import argparse
from collections import defaultdict
import json

from utils import ensure_dir

#args
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_file', type=str, help='input file path')
parser.add_argument('--output_dir', type=str, help='output directory path')
parser.add_argument('--split_number', type=int, help='split number')
parser.add_argument('--total_splits', type=int, help='split number')
parser.add_argument('--model_path', type=str, help='path to model', default="/cluster/work/sachan/NLF/model/iter_1/model_ckp_2560")
parser.add_argument('--tokenizer_path', type=str, help='path to tokenizer', default="/cluster/work/sachan/NLF/quark_TLDR_5q_tokenizer")
parser.add_argument('--data_split', type=str, help='data split', default="train")
parser.add_argument('--num_generations', required=True, type=int, help='number of generations per prompt')
parser.add_argument('--temperature', required=True, type=float, help='temperature')
parser.add_argument('--top_p', required=True, type=float, help='top_p')
parser.add_argument('--max_new_tokens', required=True, type=int, help='max_new_tokens')
args = parser.parse_args()

ensure_dir(args.output_dir)

data = []
with open(args.input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))
data = [d['prompt'] for d in data]

# Split the data into chunks.
chunk_size = len(data) // args.total_splits
start = (args.split_number) * chunk_size
end = min((args.split_number + 1) * chunk_size, len(data))
data = data[start:end]

# Define the prompts.
sampling_params = SamplingParams(
    n=args.num_generations, 
    temperature=args.temperature, 
    top_p=args.top_p, 
    max_tokens=args.max_new_tokens
)

llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path)

outputs = llm.generate(data, sampling_params)
all_outputs = defaultdict(list)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = [op.text for op in output.outputs]
    all_outputs[prompt].append(generated_text)

## save outputs and prompts as json
with open(f"{args.output_dir}/{args.data_split}_output_{args.split_number}.json", "w") as f:
    for prompt, generations in all_outputs.items():
        op = {"prompt": prompt, "generations": generations[0]}
        f.write(json.dumps(op) + "\n")
