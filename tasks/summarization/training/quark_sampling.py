import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")
print(sys.path)

import os
import argparse
import yaml
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler, GenerationConfig
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from utils import set_seed, ensure_dir, WANDB_API_KEY
from tasks.summarization.models.policy import Policy
from tasks.summarization.datasets.sampling_dataset_and_collator import TLDRSamplingDataset, QuarkTLDRSamplingPromptCollatorWithPadding
from state import load_state, save_state

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

class QuarkSampler:
    def __init__(self,
                 params: dict,
                 policy: Policy,
                 quantile_tokens: List[str],
                 sampling_train_dataloader: DataLoader,
                 train_generation_config: GenerationConfig,
                 ) -> None:
        
        self.params = params
        self.num_quantiles = params['train']['num_quantiles']
        
        self.policy = policy
        self.sampling_train_dataloader = sampling_train_dataloader
        self.train_generation_config = train_generation_config

        self.quantile_tokens = quantile_tokens
        self.best_quantile_token = self.quantile_tokens[0]
        self.best_quantile_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_quantile_token)

        self.sampling_prompt_collator = QuarkTLDRSamplingPromptCollatorWithPadding(tokenizer=self.policy.tokenizer, quantile_tokens=self.quantile_tokens)

    def collate_fn_wrapper(self, batch, best_quantile=True, conditioning=True):
        return self.sampling_prompt_collator(batch, best_quantile=best_quantile, conditioning=conditioning)

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
        attention_mask = attention_mask.reshape(batch_size, -1)
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

    def sample(self, sampling_stage) -> None:        
        print(f"[Sampling stage {sampling_stage}] Sampling ...")

        if sampling_stage == 1:
            # in the 1st sampling phase, use collate_fn that collated batches of data without reward quantile tokens
            collate_fn = lambda batch: self.collate_fn_wrapper(batch, best_quantile=True, conditioning=False)     
        else:
            # in subsequent sampling phases, use collate_fn that collates batches of data with reward quantile tokens
            collate_fn = lambda batch: self.collate_fn_wrapper(batch, best_quantile=True, conditioning=True)
        
        self.sampling_train_dataloader.collate_fn = collate_fn

        prompts, prompts_quantile, generations = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.sampling_train_dataloader, total=len(self.sampling_train_dataloader), desc='Sampling from current policy')):
                input_ids, attention_mask = batch["inputs"]
                prompts_batch = batch["prompts"]

                rollouts = self.policy.sample(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    generation_config=self.train_generation_config)
                
                prompts_quantile_batch = self.decode(tokenizer=self.policy.tokenizer, query_input_ids=input_ids, skip_special_tokens=False)
                generations_batch = rollouts["generated_text"]
            
                prompts.extend(prompts_batch)
                generations.extend(generations_batch)
                prompts_quantile.extend(prompts_quantile_batch)

        # save sampling data in a json file 
        sampling_file = Path(self.params['sampling_dir']) / f"quark_sampling_data_stage_{sampling_stage}.json"
        with sampling_file.open('w') as f:
            for (prompt_quantile_data, prompt_data, generation_data) in zip(prompts_quantile, prompts, generations):
                response_dict = {
                    'prompt_quantile': prompt_quantile_data,
                    'prompt': prompt_data,
                    'generation': generation_data,
                }
                json.dump(response_dict, f)
                f.write('\n')

def main():
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
            name=f"{args['logging']['run_name']}"
        )

    # Set saving directories
    args['sampling_dir'] = os.path.join(args['save_dir'], 'sampling')
    args['model_dir'] = os.path.join(args['save_dir'], 'model')
    print(f"Writing sampling data to output directory: {args['sampling_dir']}")
    print(f"Loading policy model from directory: {args['model_dir']}")
    for dir in [args['sampling_dir'], args['model_dir']]:
        ensure_dir(dir)

    # Save the config file
    with open(os.path.join(args['save_dir'], 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    # Load the state
    state_file_path = args['train']['state_file_path'] 
    load_state(state_file_path)
    if not state_dict["sampling_stage"]:
        state_dict["sampling_stage"] = 1
    sampling_stage = state_dict["sampling_stage"]
    
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
    bad_words_ids = [tokenizer.convert_tokens_to_ids(quantile_token) for quantile_token in quantile_tokens]

    # -------------- Initialize Policy to be finetuned --------------
    policy = Policy(
        model_checkpoint_name=args['model']['policy_model']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    print(f"Policy correctly loaded to {device}.")
    # resize token_embeddings associated to the newly added tokens
    weights = policy.model.get_input_embeddings().weight.detach().cpu().numpy()
    mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
    new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in quantile_tokens])

    policy.model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        new_inits = torch.tensor(new_inits)
        policy.model.get_input_embeddings().weight[-len(quantile_tokens):, :] = new_inits

    if sampling_stage > 1:
        last_ckpt = state_dict["last_ckpt"]
        last_ckpt_path = f"{args['model_dir']}/ckpt_{last_ckpt}.pth"
        print(f"Loading Policy satate_dict from {last_ckpt_path}.")
        state_dict = torch.load(last_ckpt_path)
        policy.model.load_state_dict(state_dict["policy_model"])
        print(f"Policy satate_dict correctly loaded from {last_ckpt_path}.")

    train_generation_config = GenerationConfig(
        max_length = args["model"]["policy_model"]["train_generation_kwargs"]["max_length"],
        max_new_tokens = args["model"]["policy_model"]["train_generation_kwargs"]["max_new_tokens"],
        do_sample = args["model"]["policy_model"]["train_generation_kwargs"]["do_sample"], # False means greedy decoding
        num_beams = args["model"]["policy_model"]["train_generation_kwargs"]["num_beams"], # no beam search
        temperature = args["model"]["policy_model"]["train_generation_kwargs"]["temperature"], 
        top_k = args["model"]["policy_model"]["train_generation_kwargs"]["top_k"], # number of highest prob. vocabulary tokens to keep for top-k filtering
        top_p = args["model"]["policy_model"]["train_generation_kwargs"]["top_p"], # if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top-P or higher are kept for generation
        bad_words_ids = bad_words_ids, # List[List[int]] -> useful for Quark-based to avoid sampling of newly added tokens | list of list of tokens ids that are not allowed to be generated
        num_return_sequences = args["model"]["policy_model"]["train_generation_kwargs"]["num_return_sequences"], # may be interesting to sample many completions for which to collect feedback    
        return_dict_in_generate = True,
        pad_token_id = tokenizer.pad_token_id, # error if not passed...
    )

    # -------------- Load Sampling datasets and dataloaders --------------
    print(f'Loading data ...')
    splits = []
    if args['data']['train_split_name']:
        splits.append(args['data']['train_split_name'])

    sampling_datasets = TLDRSamplingDataset(
        local_or_remote_path=args['data']['name_or_path'],
        tokenizer=tokenizer,
        data_format=None,
        splits=splits,
        remote=args['data']['remote'])
    
    sampling_train_dataset = sampling_datasets[args['data']['train_split_name']]

    prompt_collator = QuarkTLDRSamplingPromptCollatorWithPadding(tokenizer=tokenizer, quantile_tokens=quantile_tokens)
    def collate_fn_wrapper(batch, best_quantile=True, conditioning=True):
        return prompt_collator(batch, best_quantile=best_quantile, conditioning=conditioning)
    
    sampling_train_dataloader = DataLoader(
        dataset=sampling_train_dataset,
        batch_size=args['train']['sampling_batch_size_per_card'],
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: collate_fn_wrapper(batch, best_quantile=True, conditioning=False)
    )
    print(f"Sampling Train dataset loaded with {len(sampling_train_dataset)} samples | Sampling Train dataloader with {len(sampling_train_dataloader)} batches")

    # -------------- Set up Sampler --------------
    trainer = QuarkSampler(
        params=args,
        policy=policy,
        quantile_tokens=quantile_tokens,
        sampling_train_dataloader=sampling_train_dataloader,
        train_generation_config=train_generation_config,
    )

    trainer.sample(sampling_stage)

    state_dict["sampling_stage"] += 1
    # Save the state
    save_state(state_dict, state_file_path)

if __name__ == "__main__":
    main()