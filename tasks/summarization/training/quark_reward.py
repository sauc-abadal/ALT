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
from accelerate import Accelerator

from utils import set_seed, ensure_dir
from state import load_state
from tasks.summarization.models.reward import GPTRewardModel, MyRMDataCollator, MyRMDataset

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--split', required=True, help='sampling on train/valid split')
parser.add_argument('--split_number', required=True, type=int, help='thread number / split number of the data file')
parser.add_argument('--total_splits', required=True, type=int, help='total number of threads / splits of the data file')
args = parser.parse_args()
split = args.split
split_number = args.split_number
total_splits = args.total_splits

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['split'] = split
    args['split_number'] = split_number
    args['total_splits'] = total_splits

class QuarkRewarder:
    def __init__(self,
                 params: dict,
                 reward_model: GPTRewardModel,
                 reward_dataloader: DataLoader,
                 sampling_file: str,
                 ) -> None:
        
        self.params = params
        self.reward_model = reward_model
        self.reward_dataloader = reward_dataloader
        self.sampling_file = sampling_file

    def get_rewards(self, sampling_stage) -> None:
        print(f"[Sampling stage {sampling_stage} ({self.params['split']})] Computing rewards ...")

        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(self.reward_dataloader), total=len(self.reward_dataloader), disable=not self.accelerator.is_main_process):
                
                for x in rm_batch:
                    rm_batch[x] = rm_batch[x].cuda()
                rewards_batch = self.reward_model.get_reward(**rm_batch)
                rewards.extend(rewards_batch)

        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()

        # Adding the scores to each dictionary
        for i, line in enumerate(lines):
            if i == 10:
                break
            data = json.loads(line)
            data['reward'] = rewards[i]
            lines[i] = json.dumps(data)

        # Write the modified dictionaries with rewards to the sampling JSONL file
        with open(self.sampling_file, 'w') as out_file:
            out_file.write('\n'.join(lines))

def main():

    ################################################################
    # -------------------- Set up Environment -------------------- #
    ################################################################
    gc.collect()
    torch.cuda.empty_cache()
    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic']
    )
    print(f"############### ({args['split']}) quark_reward.py ###############")
    
    # Set GPU device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    """
    # Load the state from the state_dict
    state_file_path = args['train']['state_file_path'] 
    state_dict = load_state(state_file_path)
    sampling_stage = state_dict["sampling_stage"] - 1 # sampling_stage variable increased after sampling, but rewarding takes place in the current iteration
    print(f"state_dict loaded: {state_dict}")

    # Set saving directories
    """
    sampling_stage = 2
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], f'sampling/stage_{sampling_stage}')
    ensure_dir(args['sampling_dir'])
    
    sampling_file = f"{args['sampling_dir']}/quark_sampling_data_{args['split']}_stage_{sampling_stage}.json"
    print(f"Reading/Writing reward data from/to sampling_file: {sampling_file}")
    
    print(f'Initializing models ...')
    
    ################################################################
    # ------------------ Initialize Reward Model ----------------- #
    ################################################################

    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    print("base Reward Model correctly loaded!")
    if args['reward']['load_state_dict']:
        print("Attempting to load Reward Model checkpoint...")
        rm_state_dict = torch.load(args['reward']['state_dict_path'], map_location=torch.device('cpu'))
        reward_model.load_state_dict(rm_state_dict)
        print("Reward Model checkpoint correctly loaded!")

    max_length = args['reward']['max_length']
    reward_tokenizer = AutoTokenizer.from_pretrained(args['model']['tokenizer']['name_or_path'], padding_side="right")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.max_length = max_length
    
    if args['reward']['half']:
        reward_model.half()
    reward_model.to(device)
    reward_model.eval()

    ################################################################
    # -------------- Rewarding Dataset / Dataloader -------------- #
    ################################################################

    samples = []
    with open(sampling_file, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            entry = json.loads(line)
            # rewards are computed on prompts without reward quantile tokens
            prompt = entry['prompt']
            generation = entry['generation']
            sample = prompt + generation
            samples.append(sample)
    
    # Split the data into chunks.
    chunk_size = len(samples) // args["total_splits"] + 1
    start = (args["split_number"]) * chunk_size
    end = min((args["split_number"] + 1) * chunk_size, len(samples))
    samples = samples[start:end]

    # Save chunk of sampling data into json for writing the reward scores afterward
    lines = lines[start:end]
    new_sampling_file = f"{sampling_file}_thread_{args['split_number']}"
    with open(new_sampling_file, 'w') as output_file:
        output_file.write('\n'.join(lines))

    rm_dataset = MyRMDataset(samples=samples[:10])
    rm_collator = MyRMDataCollator(tokenizer=reward_tokenizer, max_length=reward_tokenizer.max_length)
    rm_dataloader = DataLoader(
        rm_dataset, 
        shuffle=False, 
        drop_last=False,
        batch_size=args['reward']['batch_size'], 
        collate_fn=rm_collator)
    
    ################################################################
    # ---------------------- Set up Rewarder --------------------- #
    ################################################################
        
    rewarder = QuarkRewarder(
        params=args,
        reward_model=reward_model,
        reward_dataloader=rm_dataloader,
        sampling_file=new_sampling_file,
    )

    print("\n--------------------- STARTING REWARDING! ---------------------\n")
    rewarder.get_rewards(sampling_stage)
    print("\n--------------------- REWARDING COMPLETED ---------------------\n")
    
if __name__ == "__main__":
    main()