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
args = parser.parse_args()
split = args.split

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['split'] = split

class QuarkRewarder:
    def __init__(self,
                 params: dict,
                 accelerator: Accelerator,
                 reward_model: GPTRewardModel,
                 reward_dataloader: DataLoader,
                 sampling_file: str,
                 batch_size: int
                 ) -> None:
        
        self.params = params
        self.accelerator = accelerator
        self.reward_model = reward_model
        self.reward_dataloader = reward_dataloader
        self.sampling_file = sampling_file
        self.batch_size = batch_size

    def get_rewards(self, sampling_stage) -> None:
        self.accelerator.print(f"[Sampling stage {sampling_stage} ({self.params['split']})] Computing rewards ...")
        self.accelerator.wait_for_everyone()
        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(self.reward_dataloader), total=len(self.reward_dataloader), disable=not self.accelerator.is_main_process):
                
                print(f"Thread {self.accelerator.local_process_index} - Batch: {rm_batch}")
                
                for x in rm_batch:
                    rm_batch[x] = rm_batch[x].to(self.reward_model.device)
                rewards_batch = self.reward_model.get_reward(**rm_batch)
                rewards.extend(rewards_batch)

        print(f"Thread {self.accelerator.local_process_index} - Number of rewards computed: {len(rewards)}")

        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()
        indices = list(range(self.accelerator.local_process_index*self.batch_size, len(lines), self.accelerator.num_processes*self.batch_size))
        new_lines = []
        # Adding the scores to each dictionary
        for i, index in enumerate(indices):
            for sub_index in range(self.batch_size):
                index_ = index + sub_index 
                if index_ >= len(lines):
                    continue
                
                line = lines[index_]
                data = json.loads(line)
                data['reward'] = rewards[i]
                new_lines.append(json.dumps(data))

        # Write the modified dictionaries with rewards to the sampling JSONL file
        with open(f"/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/test_sampling_file_thread_{self.accelerator.local_process_index}.json", 'w') as out_file:
            out_file.write('\n'.join(new_lines))

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
    accelerator = Accelerator(even_batches=False)
    accelerator.print(f"############### ({args['split']}) quark_reward.py ###############")
    
    # Set GPUs
    num_gpus = accelerator.num_processes
    accelerator.print(f'Using {num_gpus} GPUS')
    device = accelerator.device
    
    """
    # Load the state from the state_dict
    state_file_path = args['train']['state_file_path'] 
    state_dict = load_state(state_file_path)
    sampling_stage = state_dict["sampling_stage"] - 1 # sampling_stage variable increased after sampling, but rewarding takes place in the current iteration
    accelerator.print(f"state_dict loaded: {state_dict}")

    # Set saving directories
    """
    sampling_stage = 2
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], f'sampling/stage_{sampling_stage}')
    if accelerator.is_main_process:
        ensure_dir(args['sampling_dir'])
    accelerator.wait_for_everyone()
    
    sampling_file = f"{args['sampling_dir']}/quark_sampling_data_{args['split']}_stage_{sampling_stage}.json"
    accelerator.print(f"Reading/Writing reward data from/to sampling_file: {sampling_file}")
    
    accelerator.print(f'Initializing models ...')
    
    ################################################################
    # ------------------ Initialize Reward Model ----------------- #
    ################################################################

    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    accelerator.print("base Reward Model correctly loaded!")
    if args['reward']['load_state_dict']:
        accelerator.print("Attempting to load Reward Model checkpoint...")
        rm_state_dict = torch.load(args['reward']['state_dict_path'], map_location=torch.device('cpu'))
        reward_model.load_state_dict(rm_state_dict)
        accelerator.print("Reward Model checkpoint correctly loaded!")

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

    rm_dataset = MyRMDataset(samples=samples[:50])
    rm_collator = MyRMDataCollator(tokenizer=reward_tokenizer, max_length=reward_tokenizer.max_length)
    rm_dataloader = DataLoader(
        rm_dataset, 
        shuffle=False, 
        drop_last=False,
        batch_size=args['reward']['batch_size'], 
        collate_fn=rm_collator)
    
    ################################################################
    # ---------------------- Set up Accelerator ------------------ #
    ################################################################

    accelerator.print("Preparing Reward dataloader for DDP...")
    rm_dataloader= accelerator.prepare(
        rm_dataloader
    )
    accelerator.print("Dataloader correctly prepared!")
    accelerator.print(f"After .prepare(): rm_dataloader has {len(rm_dataloader)} batches.")
    
    ################################################################
    # ---------------------- Set up Rewarder --------------------- #
    ################################################################
        
    rewarder = QuarkRewarder(
        params=args,
        accelerator=accelerator,
        reward_model=reward_model,
        reward_dataloader=rm_dataloader,
        sampling_file=sampling_file,
        batch_size=args['reward']['batch_size']
    )

    accelerator.print("\n--------------------- STARTING REWARDING! ---------------------\n")
    rewarder.get_rewards(sampling_stage)
    accelerator.print("\n--------------------- REWARDING COMPLETED ---------------------\n")
    
if __name__ == "__main__":
    main()