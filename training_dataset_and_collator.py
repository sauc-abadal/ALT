from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch

from typing import Union, List, Dict, Optional, Tuple
import os
from copy import deepcopy

from data_pool import QuarkDataPool, NLFDataPool

class NLFTrainingDataset():
    def __init__(
        self, 
        data_pool: NLFDataPool, 
        tokenizer: AutoTokenizer,
        feedback_prefix: Optional[str] = "feedback:",
        prompt_prefix: Optional[str] = "input:"):
        """
        Initalizes a Dataset for handling sequences with Natural Language feedback tokens prepended before the prompt.

        Args:
            data_pool (NLFDataPool): An instance of the NLFDataPool class containing the data.
        """

        prompts, responses, feedbacks = data_pool.get_data()
        data_dict = {
            "prompt": prompts,
            "response": responses,
            "feedback": feedbacks
        }
        train_dataset = Dataset.from_dict(data_dict)
        raw_dataset = DatasetDict({"train:": train_dataset}) 
        self.tokenizer = tokenizer
        self.feedback_prefix = feedback_prefix
        self.prompt_prefix = prompt_prefix

        self.dataset = raw_dataset.map(self.remove_leading_and_trailing_spaces, batched=False)
        self.dataset = self.dataset.map(self.compose_NLF_sequence, batched=False)
        # dataset is not pre-tokenized as it can be very large, may be more efficient to tokenize each batch on the fly
        # (every sampling stage new samples are added into the data pool)
    
    def remove_leading_and_trailing_spaces(self, example):
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        feedback = example["feedback"].strip()
        return {"prompt": prompt,              
                "response": response,
                "feedback": feedback}
    
    def compose_NLF_sequence(self, example):
        prompt = example["prompt"]
        response = example["response"]
        feedback = example["feedback"]
        input_seq = self.feedback_prefix + " " + feedback + " " + self.prompt_prefix + " "  + prompt
        output_seq = " " + response
        return {"prompt": prompt,               
                "input_seq": input_seq,
                "output_seq": output_seq}

class NLFTrainingSequenceCollatorWithPadding(object):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer_left = tokenizer
        self.tokenizer_right = deepcopy(tokenizer)
        self.tokenizer_right.padding_side = "right"
    
    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, Union[List[str], Dict[str, torch.Tensor]]]:
        """
        Collate input and output sequences for LLM log-probs computation by tokenizing, padding, and truncating
        sequences to the maximum sequence length in a batch of examples. input_sequences are left padded and output_sequences 
        are right padded.
        Args:
            examples: A list of examples, each represented as a dictionary with keys: "prompt", "input_seq", "output_seq".
            where "input_seq" contains the concatentation of feedback_prefix + feedback + prompt_prefix + prompt,
            and "output_seq" contains the response.

        Returns:
            Dictionary with keys "inputs", "outputs", "prompts", "input_seqs" and "output_seqs".
            "inputs" contains a dictionary with keys "input_ids" and "attention_mask", 
            and values left-padded tensors of shape (B, S), being S the length of the longest sequence in the batch,
            and "outputs" contains a dictionary with keys "input_ids" and "attention_mask", 
            and values right-padded tensors of shape (B, S), being S the length of the longest sequence in the batch.
        """
        prompts = [example["prompt"] for example in examples]
        input_seqs = [example["input_seq"] for example in examples]
        output_seqs = [example["output_seq"] for example in examples]

        input_seqs_dict = self.tokenizer_left(input_seqs, padding=True, truncation=True, return_tensors="pt")
        output_seqs_dict = self.tokenizer_right(output_seqs, padding=True, truncation=True, return_tensors="pt")

        return {
            "inputs": input_seqs_dict, 
            "outputs": output_seqs_dict,
            "prompts": prompts, 
            "input_seqs": input_seqs,
            "output_seqs": output_seqs}

class QuarkTrainingDataset():
    def __init__(
        self, 
        data_pool: QuarkDataPool, 
        tokenizer: AutoTokenizer):
        """
        Initalizes a Dataset for handling sequences with reward quantile tokens prepended before the prompt.

        Args:
            data_pool (QuarkDataPool): An instance of the QuarkDataPool class containing the data.
        """

        prompts, responses, quantiles = data_pool.get_data()
        data_dict = {
            "prompt": prompts,
            "response": responses,
            "quantile_token": quantiles
        }
        train_dataset = Dataset.from_dict(data_dict)
        raw_dataset = DatasetDict({"train": train_dataset}) 
        self.tokenizer = tokenizer

        self.dataset = raw_dataset.map(self.remove_leading_and_trailing_spaces, batched=False)
        self.dataset = self.dataset.map(self.compose_Quark_sequence, batched=False)
        # dataset is not pre-tokenized as it can be very large, may be more efficient to tokenize each batch on the fly
        # (every sampling stage new samples are added into the data pool)
    
    def remove_leading_and_trailing_spaces(self, example):
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        quantile = example["quantile_token"].strip()
        return {"prompt": prompt,              
                "response": response,
                "quantile_token": quantile}
    
    def compose_Quark_sequence(self, example):
        prompt = example["prompt"]
        response = example["response"]
        quantile = example["quantile_token"]
        input_seq = quantile + prompt
        output_seq = " " + response
        return {"prompt": prompt,               
                "input_seq": input_seq,
                "output_seq": output_seq}

class QuarkTrainingSequenceCollatorWithPadding(object):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer_left = tokenizer
        self.tokenizer_right = deepcopy(tokenizer)
        self.tokenizer_right.padding_side = "right"
    
    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, Union[List[str], Dict[str, torch.Tensor]]]:
        """
        Collate input and output sequences for LLM log-probs computation by tokenizing, padding, and truncating
        sequences to the maximum sequence length in a batch of examples. input_sequences are left padded and output_sequences 
        are right padded.
        Args:
            examples: A list of examples, each represented as a dictionary with keys: "prompt", "input_seq", "output_seq".
            where "input_seq" contains the concatentation of quantile_token + prompt,
            and "output_seq" contains the response.

        Returns:
            Dictionary with keys "inputs", "outputs", "prompts", "input_seqs" and "output_seqs".
            "inputs" contains a dictionary with keys "input_ids" and "attention_mask", 
            and values left-padded tensors of shape (B, S), being S the length of the longest sequence in the batch,
            and "outputs" contains a dictionary with keys "input_ids" and "attention_mask", 
            and values right-padded tensors of shape (B, S), being S the length of the longest sequence in the batch.
        """
        prompts = [example["prompt"] for example in examples]
        input_seqs = [example["input_seq"] for example in examples]
        output_seqs = [example["output_seq"] for example in examples]

        input_seqs_dict = self.tokenizer_left(input_seqs, padding=True, truncation=True, return_tensors="pt")
        output_seqs_dict = self.tokenizer_right(output_seqs, padding=True, truncation=True, return_tensors="pt")

        return {
            "inputs": input_seqs_dict, 
            "outputs": output_seqs_dict,
            "prompts": prompts, 
            "input_seqs": input_seqs,
            "output_seqs": output_seqs}
