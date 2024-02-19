import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")
print(sys.path)

import os
import argparse
import yaml
import json
from typing import Dict, List, Tuple, Optional, Union
import gc

from transformers import AutoTokenizer
import torch
import numpy as np
import wandb

from utils import distinctness, set_seed, ensure_dir, reduce_mean, WANDB_API_KEY
from tasks.summarization.models.policy import Policy
from state import load_state

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

class QuarkEvaluator:
    def __init__(self,
                 params: dict,
                 ref_policy: Policy,
                 quantile_tokens: List[str],
                 reward_file: str,
                 ) -> None:
        
        self.params = params
        self.num_quantiles = params['train']['num_quantiles']
        self.ref_policy = ref_policy
        self.ref_policy.model.eval()
        self.reward_file = reward_file

        self.quantile_tokens = quantile_tokens
        self.best_quantile_token = self.quantile_tokens[0]
        self.best_quantile_id = self.ref_policy.tokenizer.convert_tokens_to_ids(self.best_quantile_token)

    def remove_quantile_from_prompt_input_ids(self,
                                              input_ids: torch.Tensor,
                                              attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            input_ids: tensor of shape (batch_size, seq_length) with left-padding and a prepended reward quantile token.
                e.g., [50256, 50256, 50256,    0,  35,  43,  96, 115] -> 0 to be removed
                      [50256, 50256,     0, 3445, 245,  15, 4900, 86] -> 0 to be removed
                      [    0, 1105,     24, 1111,  25, 902, 1500, 10] -> 0 to be removed  
                    input_ids.shape = [3, 8] -> [3, 7]
            attention_mask: tensor of shape (batch_size, seq_length) with left-padding and a prepended reward quantile token attention.
                e.g., [0, 0, 0, 1, 1, 1, 1, 1] -> first 1 to be removed
                      [0, 0, 1, 1, 1, 1, 1, 1] -> first 1 to be removed
                      [1, 1, 1, 1, 1, 1, 1, 1] -> first 1 to be removed   
                      attention_mask.shape = [3, 8] -> [3, 7]
        """
        batch_size, seq_length = input_ids.shape
        first_att_idxs = torch.argmax(attention_mask, dim=1).unsqueeze(1) # shape (batch_size, 1)
        # define boolean masking
        mask = torch.arange(seq_length).unsqueeze(0) != first_att_idxs # shape (batch_size, seq_length)
        # e.g., [True,  True,  True, False, True, True, True, True]
        #       [True,  True, False,  True, True, True, True, True]
        #       [False, True,  True,  True, True, True, True, True]
        input_ids = input_ids[mask].reshape(batch_size, -1)
        attention_mask = attention_mask[mask].reshape(batch_size, -1)
        return (input_ids, attention_mask)
        
    def decode(self, 
               tokenizer: AutoTokenizer,
               prompt_input_ids: torch.Tensor,
               generation_input_ids: Optional[torch.Tensor] = None,
               skip_special_tokens=True) -> Union[List[str], Tuple[List[str], List[str]]]:
        
        prompts = tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
        if generation_input_ids is None:
            return prompts
        
        generations = tokenizer.batch_decode(generation_input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
        return (prompts, generations)

    def eval(self, sampling_stage, step_num) -> None:
        print(f"[Sampling stage {sampling_stage}] | Evaluating on the dev set ...")

        prompts, generations, rewards = [], [], []
        with open(self.reward_file, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                entry = json.loads(line)
                prompt = entry['prompt']
                generation = entry['generation']
                reward = entry['reward']
                prompts.append(prompt)
                generations.append(generation)
                rewards.append(reward)

        perplexities = []
        # get ref_logprobs to compute perplexity
        with torch.no_grad():
            input_dict = self.ref_policy.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            input_ids = input_dict["input_ids"]
            attention_mask = input_dict["attention_mask"]

            output_dict = self.ref_policy.tokenizer(generations, padding=True, truncation=True, return_tensors="pt")
            generations_input_ids = output_dict["input_ids"]
            generations_attention_mask = output_dict["attention_mask"]

            ref_outputs = self.ref_policy.forward_pass(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generated_input_ids=generations_input_ids,
                generated_attention_mask=generations_attention_mask
            )

            ref_logprobs = ref_outputs['generated_logprobs']
            perplexity = torch.exp(-1 * reduce_mean(ref_logprobs, generations_attention_mask.float(), axis=1), dim=1)
            perplexities.extend(perplexity.cpu().detach().numpy().tolist())

        rewards = np.array(rewards)
        avg_ppl, avg_reward = np.nanmean(perplexities), np.mean(rewards)
        dist_1, dist_2, dist_3 = distinctness(generations)
        print(f"Perplexity: {avg_ppl:+.2f}")
        print(f"Avg. Reward: {avg_reward:.2f}")
        print(f"dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}")
        if self.params['logging']['wandb_log']:
            wandb.log({f'Evaluation/perplexity': avg_ppl}, step=step_num)
            wandb.log({f'Evaluation/reward': avg_reward}, step=step_num)
            wandb.log({f'Evaluation/Dist-1': dist_1}, step=step_num)
            wandb.log({f'Evaluation/Dist-2': dist_2}, step=step_num)
            wandb.log({f'Evaluation/Dist-3': dist_3}, step=step_num)

        # Adding the perplexity scores to each dictionary
        for i, line in enumerate(lines):
            data = json.loads(line)
            data['perplexity'] = perplexities[i]
            lines[i] = json.dumps(data)

        # Write the modified dictionaries with rewards to the sampling JSONL file
        with open(self.reward_file, 'w') as out_file:
            out_file.write('\n'.join(lines))

def main():
    print("############### quark_eval.py ###############")
    gc.collect()
    torch.cuda.empty_cache()

    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    # Set GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Set wandb logging
    wandb_log = args['logging']['wandb_log']
    if wandb_log:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            entity=args['logging']['wandb_entity'],
            project=args['logging']['wandb_project'],
            name=f"{args['logging']['run_name']}",
            id=f"{args['logging']['run_id']}"
        )

    # Load the state
    state_file_path = args['train']['state_file_path'] 
    state_dict = load_state(state_file_path)
    sampling_stage = state_dict["sampling_stage"] - 1
    step_num = state_dict["step_num"]

    # Set saving directories
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], 'sampling')
    ensure_dir(args['sampling_dir'])
    print(f"Writing sampling (eval) data to output directory: {args['sampling_dir']}")

    # Save the config file
    with open(os.path.join(args['save_dir'], 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    print(f'Initializing models ...')

    # -------------- Initialize Tokenizer --------------
    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['tokenizer']['name_or_path'],
        padding_side=args['model']['policy_model']['input_padding_side'], # left padding
        model_max_length=args['train']['max_input_length']) # GPT2Tokenizer -> vocab_size 50257 (id from 0 to 50256) + extra_tokens for efficiency (id from 50257 to 50399) -> 50400 total vocabulary 
    
    if tokenizer.pad_token is None:
        print("Setting PAD token to EOS token for open-ended generation.")
        tokenizer.pad_token = tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    num_quantiles = args['train']['num_quantiles']
    quantile_tokens =  [f"_QUANTILE_TOKEN_{str(quantile_idx)}" for quantile_idx in range(num_quantiles)]

    # add special reward quantile tokens to the tokenizer
    tokenizer.add_tokens(quantile_tokens, special_tokens=True)

    # -------------- Initialize Reference Policy --------------
    ref_policy = Policy(
        model_checkpoint_name=args['model']['ref_policy']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    print(f"Reference policy loaded to {device}.")
    reward_file = f"{args['sampling_dir']}/quark_sampling_data_valid_stage_{sampling_stage}.json"
    
    # -------------- Set up trainer --------------
    evaluator = QuarkEvaluator(
        params=args,
        ref_policy=ref_policy,
        quantile_tokens=quantile_tokens,
        reward_file=reward_file
    )     
    evaluator.eval(sampling_stage, step_num)

if __name__ == "__main__":
    main()