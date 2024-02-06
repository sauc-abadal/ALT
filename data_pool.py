from typing import List
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import json

class QuarkDataPool:
    def __init__(self, reward_quantile_tokens: List[str], num_quantiles: int):
        """
        Initialize a data pool for organizing and managing data into quantiles.

        Args:
            reward_quantile_tokens (List[str]): A list of possible reward quantile tokens associated with each quantile.
            num_quantiles (int): The number of quantiles to divide the data pool into.

        Example:
            num_quantiles = 5
            reward_quantile_tokens = [_TREE_TOKEN_0_0, _TREE_TOKEN_0_1, _TREE_TOKEN_0_2, _TREE_TOKEN_0_3, _TREE_TOKEN_0_4]
            
        """
        self.reward_quantile_tokens = reward_quantile_tokens
        self.num_quantiles = num_quantiles

        self.scores_pool = []
        self.prompts_pool, self.responses_pool, self.quantiles_pool = [], [], []

    def add(self, prompts: List[str], responses: List[str], scores: List[float]):
        """
        Add data to the data pool, sort it from highest to lowest reward, and divide it into equally-sized quantiles.

        Args:
            prompts (List[str]): A list of input prompts.
            responses (List[str]): A list of response sequences.
            scores (List[float]): A list of reward scores

        Note:
            - Data is sorted by reward scores, from highest to lowest reward, and reward quantile tokens are assigned to samples based on quantile ranking.
            - e.g., Quantile 0 is associated with highest reward and Quantile 4 is associated with lowest reward
        """
        self.prompts_pool.extend(prompts)
        self.responses_pool.extend(responses)
        self.scores_pool.extend(scores)
        
        # quantiles_pool restarted every time we add new data to the data_pool (after sampling) as data will be associated to different quantiles
        self.quantiles_pool = ["" for _ in range(len(self.prompts_pool))]

        # sort data iteratively according to one attribute score at a time
        for attr_type in range(self.num_attributes):
            data = zip(self.prompt_pool, self.response_pool, self.feedback_pool, self.score_pool[f"attr_{str(attr_type)}"])
            data = [x for x in data if x[-1] is not None]

            # get the sorting indices corresponding to sorting the data according to current attr_type
            sorted_indices = [i for i, x in sorted(enumerate(data), key=lambda x: x[1][-1], reverse=True)]

            # update pool of prompts, responses, feedback, and scores for current attr_type, according to current attr_type
            sorted_data = [data[i] for i in sorted_indices]
            self.prompt_pool, self.response_pool, self.feedback_pool, self.score_pool[f"attr_{str(attr_type)}"] = [list(x) for x in list(zip(*sorted_data))]
            # update pool of scores for all other attr_types, according to current_attr_type
            for j in range(self.num_attributes):
              if j != attr_type:
                self.score_pool[f"attr_{str(j)}"] = [self.score_pool[f"attr_{str(j)}"][i] for i in sorted_indices]
            
            # divide data pool into quantiles of roughly equal size (last quantile will be larger if the length of the data is not 
            # divisible by the desired number of quantiles), and obtain the associated quantile index to each sample in the data pool
            quantile_idx = [[i] * (len(sorted_data) // self.num_quantiles) for i in range(self.num_quantiles)]
            quantile_idx = [y for x in quantile_idx for y in x] # unfold list of lists into a single list
            quantile_idx = quantile_idx + [self.num_quantiles - 1] * (len(sorted_data) - len(quantile_idx)) # append indices for the last quantile
            # e.g., quantile_idx will be [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4] if currently the data pool has length 14 and we want to use 5 quantiles (the last four '4's are added as 14 % 5 != 0)
            
            # --- QUARK-based ---
            if not self.nlf_cond:
                self.feedback_pool = [(self.feedback_pool[i] + self.reward_quantile_tokens[attr_type][idx]).strip() for i, idx in enumerate(quantile_idx)]

            # --- CTG NLF ---
            else:
                if attr_type == 0: # empty feedack_pool
                    self.feedback_pool = [(self.feedback_pool[i] + " " + self.reward_quantile_tokens[attr_type][idx]).strip() for i, idx in enumerate(quantile_idx)] 
                else:
                    self.feedback_pool = [(self.feedback_pool[i] + ", and " + self.reward_quantile_tokens[attr_type][idx]).strip() for i, idx in enumerate(quantile_idx)]

    def get_data(self):
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[str]: A tuple containing the input prompts, response sequences,
            and feedback associated with quantiles.

        """
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.feedback_pool)

    def save_data_for_training_in_json(self, save_path, step_num):
        # save tuples of (prompt_feedback, promp, response, score) in reward_file
        reward_file = Path(save_path) / f"multitask_train_data_{step_num}.json"
        score_pool = self.score_pool
        with reward_file.open('a') as f:
            for idx, (prompt_feedback_data, prompt_data, response_data) in enumerate(zip(self.feedback_pool, self.prompt_pool, self.response_pool)):
                response_dict = {
                    'prompt_feedback': prompt_feedback_data,
                    'prompt': prompt_data,
                    'response': response_data,
                    'scores': [score_pool[f"attr_{str(attr)}"][idx] for attr in range(self.num_attributes)] # i.e., for each sample [rel, fact, comp]
                }
                json.dump(response_dict, f)
                f.write('\n')