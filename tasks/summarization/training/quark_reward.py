import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")
print(sys.path)

import os
import argparse
import yaml
import json

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
parser.add_argument('--first_iter', required=True, help='whether or not is the first sampling iteration')
parser.add_argument('--split', required=True, help='sampling on train/valid split')
args = parser.parse_args()
first_iter = bool(args.first_iter)
split = args.split

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['first_iter'] = first_iter
    args['split'] = split

class QuarkRewarder:
    def __init__(self,
                 params: dict,
                 reward_model: GPTRewardModel,
                 reward_tokenizer: AutoTokenizer,
                 sampling_file: str,
                 data_pool: QuarkDataPool,
                 ) -> None:
        
        self.params = params
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.rm_collator = MyRMDataCollator(tokenizer=self.reward_tokenizer, max_length=self.reward_tokenizer.max_length)
        self.sampling_file = sampling_file
        self.data_pool = data_pool

    def get_rewards(self, sampling_stage) -> None:
        print(f"[Sampling stage {sampling_stage} ({self.params['split']})] Computing rewards ...")

        samples = []
        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                entry = json.loads(line)
                # rewards are computed on prompts without reward quantile tokens
                prompt = entry['prompt']
                generation = entry['generation']
                sample = prompt + generation
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
    
    def update_DataPool(self, sampling_stage) -> QuarkDataPool:
        print(f"[Sampling stage {sampling_stage} ({self.params['split']})] Updating DataPool ...")
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
        
        # sampling data on the current sampling stage is added to the data_pool,
        # all the data in the data_pool is sorted by reward scores and assigned
        # to a reward quantile token
        self.data_pool.add(prompts=prompts, responses=generations, scores=rewards)

        # save training data in training_file (reward quantile tokens used during training)
        self.data_pool.save_data_for_training_in_json(self.params['sampling_dir'], sampling_stage)
        return self.data_pool

def main():
    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    # Set GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')
    
    # Set wandb logging
    wandb_log = args['logging']['wandb_log']
    if wandb_log:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            entity=args['logging']['wandb_entity'],
            project=args['logging']['wandb_project'],
            name=f"{args['logging']['run_name']}"
        )

    # Load the state
    state_file_path = args['train']['state_file_path'] 
    state_dict = load_state(state_file_path)
    sampling_stage = state_dict["sampling_stage"] - 1

    # Set saving directories
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], 'sampling')
    ensure_dir(args['sampling_dir'])
    print(f"Writing reward data to output directory: {args['sampling_dir']}")
        
    # Save the config file
    with open(os.path.join(args['save_dir'], 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)
    
    print(f'Initializing models ...')
    
    num_quantiles = args['train']['num_quantiles']
    quantile_tokens =  [f"_QUANTILE_TOKEN_{str(quantile_idx)}" for quantile_idx in range(num_quantiles)]

    # -------------- Initialize Reward Model --------------
    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    if args['reward']['load_state_dict']:
        state_dict = torch.load(args['reward']['state_dict_path'])
        reward_model.load_state_dict(state_dict)
        print("Reward Model correctly loaded!")

    max_length = args['reward']['max_length']
    reward_tokenizer = AutoTokenizer.from_pretrained(args['model']['tokenizer']['name_or_path'], padding_side="right")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.max_length = max_length
    
    reward_model.cuda()
    reward_model.eval()
    if args['reward']['half']:
        reward_model.half()

    sampling_file = f"{args['sampling_dir']}/quark_sampling_data_{args['split']}_stage_{sampling_stage}.json"

    if args['split'] == 'train':
        # -------------- Initialize DataPool --------------
        if args['first_iter']:
            # Initialize new DataPool
            data_pool = QuarkDataPool(
                reward_quantile_tokens=quantile_tokens, num_quantiles=num_quantiles
            )
        else:
            # Load existing DataPool
            datapool_load_dict = state_dict["data_pool"]
            data_pool = QuarkDataPool(
                reward_quantile_tokens=quantile_tokens, num_quantiles=num_quantiles
            )
            data_pool.load_from_dict(datapool_load_dict)
    else:
        data_pool = None

    # -------------- Set up Rewarder --------------
    rewarder = QuarkRewarder(
        params=args,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        sampling_file=sampling_file,
        data_pool=data_pool,
    )

    rewarder.get_rewards(sampling_stage)
    if args['split'] == 'train':
        data_pool = rewarder.update_DataPool(sampling_stage)
        import pdb
        pdb.set_trace()
        datapool_save_dict = data_pool.serialize_to_dict(args['save_dir'])
        state_dict["data_pool"] = datapool_save_dict
        # Save the state
        save_state(state_dict, state_file_path)

if __name__ == "__main__":
    main()