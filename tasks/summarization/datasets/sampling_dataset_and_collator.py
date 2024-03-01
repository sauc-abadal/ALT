import random
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch

from typing import Union, List, Dict, Optional, Tuple
import os

class TLDRSamplingDataset():
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

class NLFTLDRSamplingPromptCollatorWithPadding(object):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
    
    def get_feedback_tokens(self) -> List[int]:
        pass

    def __call__(
            self, 
            examples: List[Dict[str, Union[str, List[int]]]],
            conditioning=True) -> Dict[str, Union[List[str], Dict[str, torch.Tensor]]]:
        """
        Collate prompts for LLM input by padding already tokenized prompts to the maximum sequence 
        length in a batch of examples. Also prepend "feedback" tokens to prompt input_ids.
        Args:
            examples: A list of examples, each represented as a dictionary with keys: "prompt" containing the prompt text,
            "prompt_input_ids" and "prompt_attention_mask" with the tokenized prompt text and attention mask respectively, 
            and "summary" containing the human-written summary text.
            conditioning: boolean indicating whether feedback tokens must be prepended to prompt input_ids or not
            (in the first sampling step we sample unconditioned).

        Returns:
            Dictionary with keys "inputs", "prompts", and "summaries". "inputs" contains 
            a dictionary with keys "input_ids" and "attention_mask", 
            and values padded tensors of shape (B, S), being S the length of the longest sequence in the batch,
            where each element in the batch has been prepended with a feedback token if conditioning=True.
        """
        prompts = [example["prompt"] for example in examples]
        summaries = [example["summary"] for example in examples]

        desired_keys = ["prompt_input_ids", "prompt_attention_mask"]
        renamed_keys = ["input_ids", "attention_mask"]
        inputs = [{renamed_keys[i]: example[key] for i, key in enumerate(desired_keys)} for example in examples]

        if conditioning:
            input_ids = [input_ids_batch["input_ids"] for input_ids_batch in inputs]
            attention_mask = [attention_mask_batch["attention_mask"] for attention_mask_batch in inputs]

            # preprend feedback tokens to prompt input_ids before left-padding
            feedback_tokens = self.get_feedback_tokens()
            input_ids = [feedback_tokens + input_ids_batch for input_ids_batch in input_ids]
            attention_mask = [[1]*len(feedback_tokens) + attention_mask_batch for attention_mask_batch in attention_mask]
 
            inputs = [
                {
                    "input_ids": input_ids_batch,
                    "attention_mask": attention_mask_batch,
                }
            for input_ids_batch, attention_mask_batch in zip(input_ids, attention_mask)]

        inputs = self.data_collator(inputs)
        return {"inputs": inputs, "prompts": prompts, "summaries": summaries}

class QuarkTLDRSamplingPromptCollatorWithPadding(object):
    def __init__(self, tokenizer: AutoTokenizer, quantile_tokens: List[str]):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

        self.quantile_tokens = quantile_tokens # List[str]
        self.quantile_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.quantile_tokens) # List[int]
        self.best_quantile_token = self.quantile_tokens[0] # str
        self.best_quantile_id = self.tokenizer.convert_tokens_to_ids(self.best_quantile_token) # int
        self.num_quantiles = len(self.quantile_tokens)

    def __call__(
        self, 
        examples: List[Dict[str, Union[str, List[int]]]],
        best_quantile=True,
        conditioning=True) -> Dict[str, Union[List[str], Dict[str, torch.Tensor]]]:
        """
        Collate prompts for LLM input by padding already tokenized prompts to the maximum sequence 
        length in a batch of examples. Also prepend reward quantile token to prompt input_ids.
        Args:
            examples: A list of examples, each represented as a dictionary with keys: "prompt" containing the prompt text,
            "prompt_input_ids" and "prompt_attention_mask" with the tokenized prompt text and attention mask respectively, 
            and "summary" containing the human-written summary text.
            best_quantile: boolean indicating whether to condition on the best reward quantile token during sampling, or
            condition on randomly drawn quantiles.
            conditioning: boolean indicating whether reward quantile tokens must be prepended to prompt input_ids or not
            (in the first sampling step we sample unconditioned).

        Returns:
            Dictionary with keys "inputs", "prompts", and "summaries". "inputs" contains 
            a dictionary with keys "input_ids" and "attention_mask", 
            and values padded tensors of shape (B, S), being S the length of the longest sequence in the batch,
            where each element in the batch has been prepended with a reward quantile token if conditioning=True.
        """
        prompts = [example["prompt"] for example in examples]
        summaries = [example["summary"] for example in examples]

        desired_keys = ["prompt_input_ids", "prompt_attention_mask"]
        renamed_keys = ["input_ids", "attention_mask"]
        inputs = [{renamed_keys[i]: example[key] for i, key in enumerate(desired_keys)} for example in examples]
        
        if conditioning:
            input_ids = [input_ids_batch["input_ids"] for input_ids_batch in inputs]
            attention_mask = [attention_mask_batch["attention_mask"] for attention_mask_batch in inputs]
            # preprend reward quantile token to prompt input_ids before left-padding
            if best_quantile:
                input_ids = [[self.best_quantile_id] + input_ids_batch for input_ids_batch in input_ids]
                attention_mask = [[1] + attention_mask_batch for attention_mask_batch in attention_mask]
            else:
                batch_size = len(input_ids)
                quantiles_idx = random.choices(range(self.num_quantiles), k=batch_size)
                input_ids = [[self.quantile_tokens_ids[quantiles_idx[i]]] + input_ids[i] for i in range(batch_size)]
                attention_mask = [[1] + attention_mask_batch for attention_mask_batch in attention_mask]

            inputs = [
                {
                    "input_ids": input_ids_batch,
                    "attention_mask": attention_mask_batch,
                }
            for input_ids_batch, attention_mask_batch in zip(input_ids, attention_mask)]

        inputs = self.data_collator(inputs)
        return {"inputs": inputs, "prompts": prompts, "summaries": summaries}

