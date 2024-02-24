import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")
print(sys.path)

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
from tasks.summarization.training.state import load_state, save_state
from data_pool import QuarkDataPool
from tasks.summarization.models.reward import GPTRewardModel, MyRMDataCollator, MyRMDataset

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

class QuarkRewarder:
    def __init__(self,
                 params: dict,
                 sampling_file: str,
                 data_pool: QuarkDataPool,
                 ) -> None:
        
        self.params = params
        self.sampling_file = sampling_file
        self.data_pool = data_pool
    
    def update_DataPool(self, sampling_stage) -> QuarkDataPool:
        print(f"[Sampling stage {sampling_stage} (train)] Updating DataPool ...")
        prompts, generations, rewards = [], [], []
        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                entry = json.loads(line)
                prompt = entry['prompt']
                generation = entry['generation']
                reward = entry['reward']
                prompts.append(prompt)
                generations.append(generation)
                rewards.append(reward)
        
        print(len(prompts))
        print(len(generations))
        print(len(rewards))

        # sampling data on the current sampling stage is added to the data_pool,
        # all the data in the data_pool is sorted by reward scores and assigned
        # to a reward quantile token
        self.data_pool.add(prompts=prompts, responses=generations, scores=rewards)

        # save training data in training_file (reward quantile tokens used during training)
        self.data_pool.save_data_for_training_in_json(self.params['sampling_dir'], sampling_stage)
        return self.data_pool

def main():

    # Load the state
    state_file_path = args['train']['state_file_path'] 
    state_dict = load_state(state_file_path)
    sampling_stage = state_dict["sampling_stage"] - 1
    print(f"state_dict loaded: {state_dict}")

    # Set saving directories
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], 'sampling')

    print(f'Initializing models ...')
    
    num_quantiles = args['train']['num_quantiles']
    quantile_tokens =  [f"_QUANTILE_TOKEN_{str(quantile_idx)}_" for quantile_idx in range(num_quantiles)]


    sampling_file = f"{args['sampling_dir']}/quark_sampling_data_train_stage_{sampling_stage}.json"
    print(f"Reading sampling_file from: {sampling_file}")

    # -------------- Initialize DataPool --------------

    # Initialize new DataPool
    data_pool = QuarkDataPool(
        reward_quantile_tokens=quantile_tokens, num_quantiles=num_quantiles
    )
    print("New data_pool initialized.")


    # -------------- Set up Rewarder --------------
    rewarder = QuarkRewarder(
        params=args,
        sampling_file=sampling_file,
        data_pool=data_pool,
    )

    data_pool = rewarder.update_DataPool(sampling_stage)
    print("data_pool correctly updated!")
    datapool_save_dict = data_pool.serialize_to_dict(args['save_dir'])
    print("data_pool correctly serialized!")
    state_dict["data_pool"] = datapool_save_dict
    # Save the state
    save_state(state_dict, state_file_path)
    print(f"state_dict saved: {state_dict}")

if __name__ == "__main__":
    main()
