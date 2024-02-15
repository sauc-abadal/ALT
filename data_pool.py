from typing import List, Tuple
from copy import deepcopy
from pathlib import Path
import json

class NLFDataPool:
    def __init__(self, num_feedback_labels: int):
        """
        Initialize a data pool for organizing and managing data.

        The argument 'num_feedback_labels' might be helpful for categorizing the different feedbacks into
        what extent specific samples are regarded as aligned. It might be understood as a score, e.g., 1-5,
        which can be computed by the feedback provider (human or AI-written) along with the NL feedback.

        It may help us to carry out some analysis afterward, and also it might be employed to train a 
        supervised policy just on the "best" feedback samples, i.e., label 5.
        """

        self.num_feedback_labels = num_feedback_labels
        self.feedbacks_labels_pool = []
        self.prompts_pool, self.responses_pool, self.feedbacks_pool = [], [], []

    def add(self, prompts: List[str], responses: List[str], feedbacks: List[str], feedbacks_labels: List[str]):
        """
        Add data to the data pool.

        Args:
            prompts (List[str]): A list of input prompts.
            responses (List[str]): A list of response sequences.
            feedbacks (List[str]): A list of natural language feedbacks (human or AI-written).
            feedbacks_labels (List[int]): A list of labels specifying each feedback category.

        """
        self.prompts_pool.extend(prompts)
        self.responses_pool.extend(responses)
        self.feedbacks_labels.extend(feedbacks)
        self.feedbacks_labels_pool.extend(feedbacks_labels)
        
    def get_data(self):
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[str]: A tuple containing the input prompts, response sequences,
            and associated NL feedbacks.

        """
        return deepcopy(self.prompts_pool), deepcopy(self.responses_pool), deepcopy(self.feedbacks_pool)

    def save_data_for_training_in_json(self, save_path, sampling_stage):
        # save tuples of (quantile_token, promp, response, score) in reward_file
        reward_file = Path(save_path) / f"nlf_training_data_stage_{sampling_stage}.json"
        with reward_file.open('a') as f:
            for (feedback_data, prompt_data, response_data, feedback_label_data) in zip(self.feedbacks_pool, self.prompts_pool, self.responses_pool, self.feedbacks_labels_pool):
                response_dict = {
                    'feedback': feedback_data,
                    'prompt': prompt_data,
                    'response': response_data,
                    'feedback_label': feedback_label_data
                }
                json.dump(response_dict, f)
                f.write('\n')

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
        self.quantiles_pool = []

        data = zip(self.prompts_pool, self.responses_pool, self.scores_pool)
        data = [x for x in data if x[-1] is not None]
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True) # sorted from maximum to minimum reward scores
        self.prompts_pool, self.responses_pool, self.scores_pool = [list(x) for x in list(zip(*sorted_data))]

        # divide data pool into quantiles of roughly equal size (last quantile will be larger if the length of the data is not 
        # divisible by the desired number of quantiles), and obtain the associated quantile index to each sample in the data pool
        # e.g., currently the data pool has length 14 and we want to use 5 quantiles (the last four '4's are added as 14 % 5 != 0)
        quantiles = [[i] * (len(sorted_data) // self.num_quantiles) for i in range(self.num_quantiles)] # -> [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        quantiles = [y for x in quantiles for y in x] # unfold list of lists into a single list -> [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        quantiles = quantiles + [self.num_quantiles - 1] * (len(sorted_data) - len(quantiles)) # append indices for the last quantile -> [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4]
        
        self.quantiles_pool = [self.reward_quantile_tokens[i] for i in quantiles] # quantile idxs mapped to tokens understandable by the tokenizer (newly added)

    def get_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[str]: A tuple containing the input prompts, response sequences,
            and associated reward quantile tokens.

        """
        return deepcopy(self.prompts_pool), deepcopy(self.responses_pool), deepcopy(self.quantiles_pool)

    def save_data_for_training_in_json(self, save_path, sampling_stage):
        # save tuples of (quantile_token, promp, response, score) in reward_file
        reward_file = Path(save_path) / f"quark_training_data_stage_{sampling_stage}.json"
        with reward_file.open('a') as f:
            for (quantile_data, prompt_data, response_data, score_data) in zip(self.quantiles_pool, self.prompts_pool, self.responses_pool, self.scores_pool):
                response_dict = {
                    'quantile_token': quantile_data,
                    'prompt': prompt_data,
                    'response': response_data,
                    'reward_score': score_data
                }
                json.dump(response_dict, f)
                f.write('\n')