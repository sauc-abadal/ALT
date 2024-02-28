import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")

import os
import argparse
import yaml
import json
import gc

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import wandb

from utils import set_seed, ensure_dir, WANDB_API_KEY
from state import load_state, save_state
from data_pool import QuarkDataPool
from tasks.summarization.models.reward import GPTRewardModel, MyRMDataCollator, MyRMDataset

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--split', required=True, help='sampling on train/valid split')
args = parser.parse_args()
split = args.split

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['split'] = split

class ReferencesRewarder:
    def __init__(self,
                 params: dict,
                 reward_model: GPTRewardModel,
                 reward_tokenizer: AutoTokenizer,
                 sampling_file: str,
                 ) -> None:
        
        self.params = params
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.rm_collator = MyRMDataCollator(tokenizer=self.reward_tokenizer, max_length=self.reward_tokenizer.max_length)
        self.sampling_file = sampling_file

    def get_rewards(self) -> None:
        print(f"[{self.params['split']}] Computing rewards ...")

        samples = []
        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                entry = json.loads(line)
                # rewards are computed on prompts without reward quantile tokens
                prompt = entry['prompt']
                ref_summary = entry['summary']
                sample = prompt + ref_summary
                samples.append(sample)

        rm_dataset = MyRMDataset(samples=samples)
        rm_dataloader = DataLoader(
            rm_dataset, 
            shuffle=False, 
            drop_last=False,
            batch_size=self.params['reward']['batch_size'], 
            collate_fn=self.rm_collator)

        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(rm_dataloader), total=len(rm_dataloader)):
                for x in rm_batch:
                    rm_batch[x] = rm_batch[x].cuda()
                rewards_batch = self.reward_model.get_reward(**rm_batch)
                rewards.extend(rewards_batch)

        # Adding the scores to each dictionary
        for i, line in enumerate(lines):
            data = json.loads(line)
            data['reward'] = rewards[i]
            lines[i] = json.dumps(data)

        # Write the modified dictionaries with rewards to the sampling JSONL file
        with open(self.sampling_file, 'w') as out_file:
            out_file.write('\n'.join(lines))

def main():
    print(f"############### ({args['split']}) references_reward.py ###############")
    gc.collect()
    torch.cuda.empty_cache()

    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    # Set GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')

    # Set saving directories
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], 'sampling')
    ensure_dir(args['sampling_dir'])
    print(f"Writing reward data to output directory: {args['sampling_dir']}")
    
    print(f'Initializing models ...')
    
    # -------------- Initialize Reward Model --------------
    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    if args['reward']['load_state_dict']:
        rm_state_dict = torch.load(args['reward']['state_dict_path'])
        reward_model.load_state_dict(rm_state_dict)
        print("Reward Model correctly loaded!")

    max_length = args['reward']['max_length']
    reward_tokenizer = AutoTokenizer.from_pretrained(args['model']['tokenizer']['name_or_path'], padding_side="right")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.max_length = max_length
    
    reward_model.cuda()
    reward_model.eval()
    if args['reward']['half']:
        reward_model.half()

    sampling_file = f"{args['sampling_dir']}/references_data_{args['split']}.json"
    print(f"Reading sampling_file from: {sampling_file}")

    # -------------- Set up Rewarder --------------
    rewarder = ReferencesRewarder(
        params=args,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        sampling_file=sampling_file,
    )

    rewarder.get_rewards()

if __name__ == "__main__":
    main()