from typing import List
from copy import deepcopy


class DataPool:
    # DOCUMENTED
    def __init__(self, tree_tokens, num_quantiles):
        """
        Initialize a data pool for organizing and managing data into quantiles.

        Args:
            tree_tokens (List[List[str]]): A list of natural language tokens associated with each quantile.
            num_quantiles (int): The number of quantiles to divide the data pool into.

        Attributes:
            tree_tokens (List[List[str]]): A list of NL token lists associated with quantiles.
            num_quantiles (int): The number of quantiles.
            cat_tokens (List[List[str]]): NL tokens associated with the quantiles (initialized to None).
            prompt_pool (List[str]): A list of input prompts.
            response_pool (List[str]): A list of response sequences.
            score_pool (List[float]): A list of toxicity scores.

        Note:
            The `tree_tokens` list should contain NL tokens associated with each quantile (len(tree_tokens) == num_quantiles).
        """
        self.tree_tokens = tree_tokens
        self.num_quantiles = num_quantiles

        self.cat_tokens = None
        self.prompt_pool, self.response_pool, self.score_pool = [], [], []

    # DOCUMENTED
    def add(self, prompts: List[str], responses: List[str], scores: List[float]):
        """
        Add data to the data pool and organize it into quantiles.

        Args:
            prompts (List[str]): A list of input prompts.
            responses (List[str]): A list of response sequences.
            scores (List[float]): A list of reward scores (1 - toxicity scores) corresponding to the responses.

        Note:
            - Data is sorted by reward scores, from highest to lowest reward, and control tokens are assigned to samples based on quantile ranking.
            - Quantile 0 is associated with highest reward (lowest toxicity), and Quantile 4 is associated with lowest reward (highest toxicity)!
        """
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        self.score_pool.extend(scores)

        data = zip(self.prompt_pool, self.response_pool, self.score_pool)
        data = [x for x in data if x[-1] is not None]
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True) # sorted from maximum to minimum reward scores
        self.prompt_pool, self.response_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]

        # divide data pool into quantiles of roughly equal size (last quantile will be larger if the length of the data is not 
        # divisible by the desired number of quantiles), and obtain the associated quantile index to each sample in the data pool
        cat_pos = [[i] * (len(sorted_data) // self.num_quantiles) for i in range(self.num_quantiles)]
        cat_pos = [y for x in cat_pos for y in x] # unfold list of lists into a single list
        cat_pos = cat_pos + [self.num_quantiles - 1] * (len(sorted_data) - len(cat_pos)) # append indices for the last quantile
        # e.g., cat_pos will be [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4] if currently the data pool has length 14 and we want to use 5 quantiles (the last four '4's are added as 14 % 5 != 0)
        
        self.cat_tokens = [self.tree_tokens[i] for i in cat_pos] 
        # cat_tokens will be a list of lists, where each element is a list of NL tokens associated to a quantile, e..g, ['Low', 'est', 'ĠT', 'oxicity']

    # DOCUMENTED
    def get_data(self):
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[List[str]]: A tuple containing the input prompts, response sequences,
            and NL tokens associated with quantiles.

        Note:
            - The returned NL tokens are associated with the quantiles in the same order as the input data.
        """
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.cat_tokens)

