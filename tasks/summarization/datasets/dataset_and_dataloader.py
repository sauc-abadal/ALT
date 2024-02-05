from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch

from typing import Union, List, Dict, Optional, Tuple
import os

class TLDRDataset():
    def __init__(self, 
                 local_or_remote_path: Union[str, os.PathLike],
                 tokenizer: AutoTokenizer,
                 data_format: Optional[str] = None, 
                 splits: Optional[Union[str, List[str]]] = None,
                 remote: bool = True):
        """
        Initializes de TLDRDataset with the TL;DR raw data stored either locally or in HF Hub.
        Args:

            local_or_remote_path: the HF dataset name if remote=True, or the path to the root 
            local directory containing the data files for the different splits if remote=False.
            In that second case, if no splits are provided, the path should point at an
            individual data file ending with one of the following extensions: ["csv", "txt", 
            "json", "jsonl", "pkl"].

            tokenizer: the tokenizer to be employed for tokenization

            data_format: applies if remote=False; it can be one of the following data formats
            ["csv", "text", "json", "pandas"].

            splits: splits to load, e.g., ["train", "valid"].

            remote: whether to load the dataset from the HB Hub or from local disk.
        """

        self.tokenizer = tokenizer

        # loading the dataset from the HuggingFace Hub
        if remote:
            # download specific splits
            if splits:
                if isinstance(splits, str):
                    splits = [splits]

                raw_datasets = [load_dataset(path=local_or_remote_path, 
                                            split=split) for split in splits]
                
                raw_datasets = DatasetDict(dict(zip(splits, raw_datasets)))

            # download all splits 
            else:
                raw_datasets = load_dataset(local_or_remote_path)

        # loading the dataset from local drive
        else:
            # path to root folder, data splits ending with "{split}.{data_format}"
            if splits:
                if isinstance(splits, str):
                    splits = [splits]

                data_paths = [f'{local_or_remote_path}/*{split}.{data_format}'
                               for split in splits]
                data_files = dict(zip(splits, data_paths))

            # single path to specific data file to load
            else:
                data_files = local_or_remote_path
            raw_datasets = load_dataset(data_format, data_files=data_files)

        self.datasets = raw_datasets.map(self.remove_leading_and_trailing_spaces, batched=False)
        self.datasets = self.datasets.map(self.tokenize_function, batched=True)
    
    def remove_leading_and_trailing_spaces(self, example):
        prompt = example["prompt"].strip()
        return {"prompt": prompt,              
                "summary": example["label"]}

    def tokenize_function(self, example):
        prompt = example["prompt"]
        prompt_dict = self.tokenizer(prompt, truncation=True)
        return {"prompt": prompt,              
                "prompt_input_ids": prompt_dict["input_ids"],
                "prompt_attention_mask": prompt_dict["attention_mask"],
                "summary": example["label"]}

class PromptCollatorWithPadding(object):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
    
    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, Union[List[str], Dict[str, torch.Tensor]]]:
        """
        Collate prompts for LLM input by padding already tokenized prompts to the maximum sequence 
        length in a batch of examples.
        Args:
            examples: A list of examples, each represented as a dictionary with keys: "prompt" containing the prompt text,
            "prompt_input_ids" and "prompt_attention_mask" with the tokenized prompt text and attention mask respectively, 
            and "summary" containing the human-written summary text.

        Returns:
            Dictionary with keys "inputs", "prompts", and "summaries". "inputs" contains 
            a dictionary with keys "input_ids" and "attention_mask", 
            and values padded tensors of shape (B, S), being S the length of the longest sequence in the batch.
        """
        prompts = [example["prompt"] for example in examples]
        summaries = [example["summary"] for example in examples]

        desired_keys = ["prompt_input_ids", "prompt_attention_mask"]
        renamed_keys = ["input_ids", "attention_mask"]
        inputs = [{renamed_keys[i]: example[key] for i, key in enumerate(desired_keys)} for example in examples]
        inputs = self.data_collator(inputs)

        return {"inputs": inputs, "prompts": prompts, "summaries": summaries}

