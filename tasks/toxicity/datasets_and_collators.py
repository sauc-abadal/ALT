import json

import torch
from torch.utils.data import Dataset

from data_pool import DataPool

class PromptDataset(Dataset):
    """
    PyTorch Dataset for handling prompts.

    This dataset is designed to work with prompt texts. It encapsulates the text of prompts for language generation tasks.

    Args:
        path (str): The path to a file containing prompt data in a specific format (e.g., JSON).
    """
    def __init__(self, path):
        self.prompts = [json.loads(s.strip())["prompt"]["text"].strip() for s in open(path, 'r').readlines()]

    def __len__(self):
        """
        Get the total number of prompts in the dataset.

        Returns:
            int: The number of prompts in the dataset.
        """
        return len(self.prompts)

    def __getitem__(self, idx):
        """
        Get a prompt at the specified index.

        Args:
            idx (int): The index of the prompt to retrieve.

        Returns:
            dict: A dictionary containing the prompt text.
        """
        return {'prompt': self.prompts[idx]}


class PromptCollator(object):
    def __init__(self, tokenizer):
        """
        Initialize the PromptCollator with a tokenizer.

        Args:
            tokenizer: The tokenizer used to process the input prompts.
        """
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        """
        Collate prompts for language model input, including tokenization and padding.

        Args:
            sequences (List[dict]): A list of sequences, each represented as a dictionary with a 'prompt' key containing the prompt text.

        Returns:
            torch.Tensor: Padded and tokenized input IDs for the prompts.
            torch.Tensor: Prompt input attention mask.

        Note:
            - Sequences are padded with the tokenizer's pad_token_id.
            - Attention masks are generated to indicate which tokens to attend to and which are padding.
        """
        prompts = [sequence['prompt'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for handling sequences with control tokens.

    This dataset is designed to work with sequences that have control tokens for conditioning. It encapsulates query, response,
    and associated control tokens data.

    Args:
        data_pool (DataPool): An instance of the DataPool class containing the organized data.
    """
    def __init__(self, data_pool: DataPool):
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


class SequenceCollator(object):
    def __init__(self, tokenizer):
        """
        Initialize the SequenceCollator with a tokenizer.

        Args:
            tokenizer: The tokenizer used to process the input sequences.
        """
        self.tokenizer = tokenizer

    # MODIFIED
    def __call__(self, sequences):
        """
        Collate sequences for language model input, including control tokens, padding, separator token, and attention masking.

        Args:
            sequences (List[dict]): A list of sequences, each represented as a dictionary with keys 'query', 'response', and 'cat_tokens'.

        Returns:
            torch.Tensor: Padded and tokenized query input IDs with control tokens prepended, and separator token in between.
            torch.Tensor: Query input attention mask with control and separator tokens accounted for.
            torch.Tensor: Padded and tokenized response input IDs.
            torch.Tensor: Response input attention mask.

        Note:
            - Control tokens are prepended to each input query in the batch, and padding is added to the left of the sequences.
            - A separator token is placed between the control tokens and the original tokens.
            - The 'cat_tokens' contain natural language tokens associated with different quantiles.
            - Sequences are padded with the tokenizer's pad_token_id and separated by the tokenizer's sep_token_id.
            - Attention masks are generated to indicate which tokens to attend to and which are padding.
        """
        queries = [sequence['query'] for sequence in sequences]
        responses = [sequence['response'] for sequence in sequences]
        cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]

        # Given a list of sequences, as the Natural Language Feedback tokens might be of different sizes, 
        # e.g., 
        #    ['Low', 'est', 'ĠT', 'oxicity'] -> [20535, 395, 309, 44086] -> 4 tokens
        #    ['Low', '-', 'Mod', 'erate', 'ĠT', 'oxicity'] -> [20535, 12, 5841, 21620, 309, 44086] -> 6 tokens
        #    ['Mod', 'erate', 'ĠT', 'oxicity'] -> [5841, 21620, 309, 44086] -> 4 tokens
        #    ['High', '-', 'Mod', 'erate', 'ĠT', 'oxicity'] -> [11922, 12, 5841, 21620, 309, 44086] -> 6 tokens
        #    ['Maximum', 'ĠT', 'oxicity'] -> [40541, 309, 44086] -> 3 tokens
        # 
        # we should pad them with the tokenizer.pad_token_id (I opted for padding on the left) to make them equal size and be able to
        # pass a batch of inputs to the LLM.
        # We also need to create a masking tensor associated to these tokens so as to not attend to the padding tokens.

        padding_token_id = self.tokenizer.pad_token_id
        separator_token_id = self.tokenizer.sep_token_id

        cat_max_num_tokens = max([len(cat_tokens) for cat_tokens in cat_ids])
        cat_ids_padded_left = [[padding_token_id]*(cat_max_num_tokens - len(cat_tokens)) + cat_tokens for cat_tokens in cat_ids]
        cat_ids_mask = [[1 if id != padding_token_id else 0 for id in cat_ids] for cat_ids in cat_ids_padded_left]

        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        query_input_ids = torch.cat([query_input_ids.new(cat_ids_padded_left), 
                                     query_input_ids.new([[separator_token_id]]*len(query_input_ids)),
                                     query_input_ids], dim=1)
        
        query_mask = torch.cat([query_mask.new(cat_ids_mask), 
                                query_mask.new([[1]]*len(query_mask)),
                                query_mask], dim=1)

        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask