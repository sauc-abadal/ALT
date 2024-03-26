import sys
from typing import List
sys.path.append("/cluster/project/sachan/sauc/nlf")

import os
import argparse
import yaml
import json
import gc

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from utils import set_seed, ensure_dir
from state import load_state
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
                 accelerator: Accelerator,
                 reward_dataloader: DataLoader,
                 sampling_file: str,
                 batch_size: InterruptedError
                 ) -> None:
        
        self.params = params
        self.accelerator = accelerator
        self.reward_dataloader = reward_dataloader
        self.sampling_file = sampling_file
        self.batch_size = batch_size

    def get_rewards(self) -> None:
        self.accelerator.print(f"Computing rewards ...")
        self.accelerator.wait_for_everyone()
        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(self.reward_dataloader), total=len(self.reward_dataloader), disable=not self.accelerator.is_main_process):
                print(f"Thread {self.accelerator.local_process_index} - Batch: {rm_batch}")
                rewards_batch = [self.accelerator.process_index]*len(rm_batch)
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
    accelerator.print(f"############### quark_reward.py ###############")
    
    # Set GPUs
    num_gpus = accelerator.num_processes
    accelerator.print(f'Using {num_gpus} GPUS')
    device = accelerator.device
    
    ################################################################
    # -------------- Rewarding Dataset / Dataloader -------------- #
    ################################################################

    sampling_file = "/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/test_sampling_file.json"

    samples = []
    with open(sampling_file, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            entry = json.loads(line)
            # rewards are computed on prompts without reward quantile tokens
            sample = entry['prompt']
            samples.append(sample)

    class MyDataset(Dataset):
        def __init__(self, samples: List[str]):
            self.samples = [int(x) for x in samples]

        def __getitem__(self, index):
            return self.samples[index]

        def __len__(self):
            return len(self.samples)
    
    rm_dataset = MyDataset(samples=samples)
    rm_dataloader = DataLoader(
        rm_dataset, 
        shuffle=False, 
        drop_last=False,
        batch_size=4)
    
    ################################################################
    # ---------------------- Set up Accelerator ------------------ #
    ################################################################

    accelerator.print("Preparing Reward dataloader for DDP...")
    rm_dataloader= accelerator.prepare(rm_dataloader)
    accelerator.print("Model and dataloader correctly prepared!")
    accelerator.print(f"After .prepare(): rm_dataloader has {len(rm_dataloader)} batches.")

    
    ################################################################
    # ---------------------- Set up Rewarder --------------------- #
    ################################################################
        
    rewarder = QuarkRewarder(
        params=args,
        accelerator=accelerator,
        reward_dataloader=rm_dataloader,
        sampling_file=sampling_file,
        batch_size=4
    )

    accelerator.print("\n--------------------- STARTING REWARDING! ---------------------\n")
    rewarder.get_rewards()
    accelerator.print("\n--------------------- REWARDING COMPLETED ---------------------\n")
    
if __name__ == "__main__":
    main()