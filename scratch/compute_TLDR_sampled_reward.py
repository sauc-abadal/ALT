import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

from tasks.summarization.models.reward import GPTRewardModel

def compute_rewards_and_save(jsonl_file):
    samples = []

    # Read the JSONL file, excluding the last line
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()[:-1]  # Exclude the last line
        for line in lines:
            entry = json.loads(line)
            prompt = entry['prompt']
            generation = entry['generation']
            sample = prompt + generation
            samples.append(sample)

    # Tokenize and format samples according to the expected input format for the CarperAI RM
    class MyDataset(Dataset):
        def __init__(self, samples: List[str]):

            self.samples = ["<|startoftext|>" + sample.split("TL;DR:")[0].strip() + "\n" + "TL;DR: " + sample.split("TL;DR:")[1].strip() + "<|endoftext|>" for sample in samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    class MyDataCollator:
        def __init__(self, tokenizer: AutoTokenizer, max_length: int):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, data: List[str]):
            batch = {}
            encodings_dict = tokenizer(
                data,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            batch["input_ids"] = encodings_dict["input_ids"]
            batch["attention_mask"] = encodings_dict["attention_mask"]
            return batch

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    max_length = 550

    dataset = MyDataset(samples)
    collator_fn = MyDataCollator(tokenizer, max_length)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=12, collate_fn=collator_fn)

    # Load CarperAI RM
    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft")
    state_dict = torch.load("/cluster/work/sachan/sauc/summarize_from_feedback/reward_model/rm_checkpoint/pytorch_model.bin")
    model.load_state_dict(state_dict) 
    model.cuda()
    model.eval()
    model.half()

    rewards = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            rewards_batch = model.get_reward(**batch)
            rewards.extend(rewards_batch)
    
    # Read the JSONL file, excluding the last line
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()[:-1]  # Exclude the last line

    # Adding the scores to each dictionary
    for i, line in enumerate(lines):
        data = json.loads(line)
        data['reward'] = rewards[i]
        lines[i] = json.dumps(data)

    # Write the modified dictionaries with rewards to a new JSONL file
    with open(jsonl_file, 'w') as out_file:
        out_file.write('\n'.join(lines))

compute_rewards_and_save(
    jsonl_file='/cluster/project/sachan/sauc/nlf/output/TLDR_SFT_greedy_decoding_valid.jsonl')

