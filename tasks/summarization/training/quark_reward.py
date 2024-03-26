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

from utils import set_seed, ensure_dir
from state import load_state
from tasks.summarization.models.reward import GPTRewardModel, MyRMDataCollatorMultipleGenerations, MyRMDatasetMultipleGenerations

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--input_sampling_file', required=True, type=str, help='path to input sampling file')
parser.add_argument('--output_dir', required=True, type=str, help='otuput dir where to save sampling file with rewards')
parser.add_argument('--split_number', required=True, type=int, help='thread number / split number of the data file')
parser.add_argument('--total_splits', required=True, type=int, help='total number of threads / splits of the data file')
args = parser.parse_args()
input_sampling_file = args.input_sampling_file
output_dir = args.output_dir
split_number = args.split_number
total_splits = args.total_splits

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['input_sampling_file'] = input_sampling_file
    args['output_dir'] = output_dir
    args['split_number'] = split_number
    args['total_splits'] = total_splits

class QuarkRewarder:
    def __init__(self,
                 params: dict,
                 reward_model: GPTRewardModel,
                 reward_dataloader: DataLoader,
                 sampling_file: str,
                 num_generations: int
                 ) -> None:
        
        self.params = params
        self.reward_model = reward_model
        self.reward_dataloader = reward_dataloader
        self.sampling_file = sampling_file
        self.num_generations = num_generations

    def get_rewards(self) -> None:
        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(self.reward_dataloader), total=len(self.reward_dataloader)):
                
                for x in rm_batch:
                    rm_batch[x] = rm_batch[x].cuda()
                rewards_batch = self.reward_model.get_reward(**rm_batch)
                rewards.extend(rewards_batch)

        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()

        # Adding the scores to each dictionary
        for i, line in enumerate(lines):
            data = json.loads(line)
            start_idx = i * self.num_generations
            end_idx = i * self.num_generations + self.num_generations
            data['rewards'] = rewards[start_idx:end_idx]
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
    print(f"############### quark_reward.py ###############")
    
    # Set GPU device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    sampling_file = args['input_sampling_file']
    print(f"Reading sampled data from sampling_file: {sampling_file}")
    save_dir = args['output_dir']
    print(f"Writing reward data to: {save_dir}")
    ensure_dir(save_dir)
    
    ################################################################
    # ------------------ Initialize Reward Model ----------------- #
    ################################################################
    print('Initializing base Reward Model ...')
    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    print("Base Reward Model correctly loaded!")
    if args['reward']['load_state_dict']:
        print("Attempting to load Reward Model checkpoint ...")
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

    num_generations = args['reward']['num_generations']

    with open(sampling_file, 'r') as f:
        data = json.loads(f.read())

    data_jsonl = []
    all_samples = []
    for key, value in data.items():
        prompt = key
        generations = value[0]
        assert len(generations) == num_generations
        my_dict = {
            "prompt": prompt, 
            "generations": generations
        }
        data_jsonl.append(my_dict)
        samples = [prompt + generation for generation in generations]
        assert len(samples) == num_generations
        all_samples.append(samples)

    print(f"Sampling file is a json with {len(data_jsonl)} prompts, each prompts has {len(data_jsonl[0])} generations.")
    
    # Split the data into chunks.
    chunk_size = len(samples) // args["total_splits"] + 1
    start = (args["split_number"]) * chunk_size
    end = min((args["split_number"] + 1) * chunk_size, len(samples))
    samples = samples[start:end]

    # Save chunk of sampling data into json for writing the reward scores afterward
    data_jsonl = data_jsonl[start:end]
    data_jsonl = [json.dumps(sample) for sample in data_jsonl]
    new_sampling_file = f"{save_dir}/{sampling_file.split('.')[0].split('/')[-1]}_thread_{args['split_number']}.json"
    with open(new_sampling_file, 'w') as output_file:
        output_file.write('\n'.join(data_jsonl))

    rm_dataset = MyRMDatasetMultipleGenerations(samples=samples)
    rm_collator = MyRMDataCollatorMultipleGenerations(tokenizer=reward_tokenizer, max_length=reward_tokenizer.max_length)
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
        num_generations=num_generations
    )

    print("\n--------------------- STARTING REWARDING! ---------------------\n")
    rewarder.get_rewards()
    print("\n--------------------- REWARDING COMPLETED ---------------------\n")
    
if __name__ == "__main__":
    main()
