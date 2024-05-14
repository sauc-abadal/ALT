import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")

import os
import argparse
import yaml
import json
import gc

from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

from utils import distinctness, set_seed, ensure_dir, reduce_mean
from tasks.summarization.models.policy import Policy

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--out_dir', required=True, help='path to output save directory also containing the input json file')
parser.add_argument('--input_file', required=True, help='name of the input json file contained in output dir, with prompts and generations to evaluate')
args = parser.parse_args()
out_dir = args.out_dir
input_file = args.input_file

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['out_dir'] = out_dir
    args['input_file'] = input_file

class PerplexityEvaluator:
    def __init__(self,
                 params: dict,
                 ref_policy: Policy,
                 ) -> None:
        
        self.params = params
        self.ref_policy = ref_policy
        self.ref_policy.model.eval()

        self.right_tokenizer = AutoTokenizer.from_pretrained(
            args['model']['tokenizer']['name_or_path'],
            padding_side='right', # right padding
            model_max_length=args['train']['max_input_length']
        )
        if not self.right_tokenizer.pad_token:
            self.right_tokenizer.pad_token = self.right_tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
            self.right_tokenizer.pad_token_id = self.right_tokenizer.eos_token_id

    def perplexity(self) -> None:
        prompts, generations = [], []
        with open(f"{self.params['out_dir']}/{self.params['input_file']}", 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                entry = json.loads(line)
                prompt = entry['prompt']
                generations_ = entry['generations']
                prompts.append(prompt)
                generations.extend(generations_)

        batch_size = self.params['train']['training_batch_size_per_card']

        perplexities = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Computing perplexity"):
            batch_prompts = prompts[i:i + batch_size]
            batch_generations = generations[i:i + batch_size]

            # get ref_logprobs to compute perplexity
            with torch.no_grad():
                input_dict = self.ref_policy.tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
                input_ids = input_dict["input_ids"]
                attention_mask = input_dict["attention_mask"]

                output_dict = self.right_tokenizer(batch_generations, padding=True, truncation=True, return_tensors="pt")
                generations_input_ids = output_dict["input_ids"]
                generations_attention_mask = output_dict["attention_mask"]

                masks = generations_attention_mask.to(self.ref_policy.device)

                ref_outputs = self.ref_policy.forward_pass(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generated_input_ids=generations_input_ids,
                    generated_attention_mask=generations_attention_mask
                )

                ref_logprobs = ref_outputs['generated_logprobs']
                perplexity = torch.exp(-1 * reduce_mean(ref_logprobs, masks.float(), axis=1))
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())
    
        avg_ppl = np.nanmean(perplexities)
        dist_1, dist_2, dist_3 = distinctness(generations)
        print(f"Perplexity: {avg_ppl:+.2f}")
        print(f"dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}")

        # Adding the perplexity scores to each dictionary
        for i, line in enumerate(lines):
            data = json.loads(line)
            data['perplexity'] = perplexities[i]
            lines[i] = json.dumps(data)

        # Write the modified dictionaries with rewards to the sampling JSONL file
        with open(f"{self.params['out_dir']}/{self.params['input_file']}", 'w') as out_file:
            out_file.write('\n'.join(lines))

        with open(f"{self.params['out_dir']}/eval_metrics.txt", 'w') as f:
            f.write(f"Perplexity: {avg_ppl:+.2f}\n")
            f.write(f"dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}")

def main():

    ################################################################
    # -------------------- Set up Environment -------------------- #
    ################################################################
    gc.collect()
    torch.cuda.empty_cache()

    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    print("############### perplexity.py ###############")

    # Set GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Writing sampling (eval) data to reward_file: {args['out_dir']}/{args['input_file']}")

    print(f'Initializing models ...')

    ################################################################
    # ------------------- Initialize Tokenizer ------------------- #
    ################################################################

    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['tokenizer']['name_or_path'],
        padding_side=args['model']['policy_model']['input_padding_side'], # left padding
        model_max_length=args['train']['max_input_length']
    ) # GPT2Tokenizer -> vocab_size 50257 (id from 0 to 50256) + extra_tokens for efficiency (id from 50257 to 50399) -> 50400 total vocabulary 
    
    if not tokenizer.pad_token:
        print("Setting PAD token to EOS token for open-ended generation.")
        tokenizer.pad_token = tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ################################################################
    # --------------- Initialize Reference Policy ---------------- #
    ################################################################

    ref_policy = Policy(
        model_checkpoint_name=args['model']['ref_policy']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    print(f"Reference policy loaded to {device}.")
    
    ################################################################
    # --------------------- Set up Evaluator --------------------- #
    ################################################################

    evaluator = PerplexityEvaluator(
        params=args,
        ref_policy=ref_policy,
    )     

    print("\n--------------------- STARTING EVALUATING! ---------------------\n")
    evaluator.perplexity()
    print("\n--------------------- EVALUATING COMPLETED ---------------------\n")

if __name__ == "__main__":
    main()