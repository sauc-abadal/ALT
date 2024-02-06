from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch

from typing import Union, List, Dict, Optional, Tuple
import os

from data_pool import QuarkDataPool, NLFDataPool

class NLFTrainingDataset():
    """
    Dataset for handling sequences with Natural Language feedback tokens prepended before the prompt.

    Args:
        data_pool (DataPool): An instance of the DataPool class containing the data.
    """
    def __init__(self, data_pool: NLFDataPool):
        self.queries, self.responses, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        """
        Get the total number of sequences in the dataset.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.queries)

    def __getitem__(self, idx):
        """
        Get a sequence at the specified index.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            dict: A dictionary containing the query, response, and associated control tokens.
        """
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx]
                }

class TLDRSamplingPromptCollatorWithPadding(object):
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

