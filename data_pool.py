from typing import List, Optional, Dict, Union
import random
from copy import deepcopy
import os
import json
import pickle
import numpy as np
from transformers import AutoTokenizer
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolate_histogram(hist, bins, new_bins, method='linear'):
    """
    Interpolate histogram onto new bins using first-order interpolation.
    
    Parameters:
        hist (array): Histogram values.
        bins (array): Bin edges of original histogram.
        new_bins (array): New bin edges to interpolate onto.
        method (str): Interpolation method.
        
    Returns:
        array: Interpolated histogram values.
    """
    f = interp1d(bins[:-1], hist, kind=method, fill_value=0.0, bounds_error=False)
    hist_interp = f(new_bins)
    return hist_interp[:-1]

def average_histograms(histograms, bins):
    """
    Compute the average of histograms.
    
    Parameters:
        histograms (list of arrays): List of histograms.
        bins (array): Bin edges of histograms.
        
    Returns:
        array: Averaged histogram.
    """
    # Determine the common range for the bins
    min_bin = min(bin_[0] for bin_ in bins)
    max_bin = max(bin_[-1] for bin_ in bins)
    new_bins = np.linspace(min_bin, max_bin, len(bins[0]))
    
    # Interpolate histograms onto common bins
    interpolated_histograms = [interpolate_histogram(hist, bins[i], new_bins) for i, hist in enumerate(histograms)]
    
    # Compute the average
    average_hist = np.mean(interpolated_histograms, axis=0)
    
    return average_hist, new_bins

class NLFDataPool:
    def __init__(self, num_feedback_labels: Optional[int] = None):
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

    def add(self, prompts: List[str], responses: List[str], feedbacks: List[str], feedbacks_labels: Optional[List[str]] = None):
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
        self.feedbacks_pool.extend(feedbacks)
        if feedbacks_labels:
            self.feedbacks_labels_pool.extend(feedbacks_labels)

    def update_DataPool(self, sampling_file, drop_factor: Optional[float] = None) -> None:

        # subsample (uniformly) the existing data_pool data, so as to keep some data sampled in 
        # previous sampling stages but prioritize the newly sampled data in the current sampling stage.
        if drop_factor:
            num_elements_to_keep = int(len(self.prompts_pool) * (1.0 - drop_factor))

            all_indices = list(range(len(self.prompts_pool)))
            random.shuffle(all_indices)

            indices_to_keep = all_indices[:num_elements_to_keep]

            self.prompts_pool = [self.prompts_pool[i] for i in indices_to_keep]
            self.responses_pool = [self.responses_pool[i] for i in indices_to_keep]
            self.feedbacks_pool = [self.feedbacks_pool[i] for i in indices_to_keep]

            if self.feedbacks_labels_pool:
                self.feedbacks_labels_pool = [self.feedbacks_labels_pool[i] for i in indices_to_keep]

        # get newly sampled data in the current sampling stage (from sampling json file)
        prompts, generations, feedbacks = [], [], []
        feedbacks_labels = []
        with open(sampling_file, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                entry = json.loads(line)
                prompt = entry['prompt']
                generation = entry['generation']
                feedback = entry['feedback']

                prompts.append(prompt)
                generations.append(generation)
                feedbacks.append(feedback)

                if "feedback_label" in entry:
                    feedback_label = entry['feedback_label']
                    feedbacks_labels.append(feedback_label)
                
        # sampling data on the current sampling stage is added to the data_pool,
        if feedbacks_labels:
            self.add(prompts=prompts, responses=generations, feedbacks=feedbacks, feedbacks_labels=feedbacks_labels)    
        else:
            self.add(prompts=prompts, responses=generations, feedbacks=feedbacks)  

    def get_data(self):
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[str]: A tuple containing the input prompts, response sequences,
            and associated NL feedbacks.

        """
        return deepcopy(self.prompts_pool), deepcopy(self.responses_pool), deepcopy(self.feedbacks_pool)

    def save_data_for_training_in_json(self, save_path, sampling_stage):
        # save tuples of (feedback_label, promp, response, score) in reward_file
        reward_file = Path(save_path) / f"nlf_training_data_stage_{sampling_stage}.json"
        with reward_file.open('w') as f:
            if self.feedbacks_labels_pool:
                for (feedback_label_data, feedback_data, prompt_data, response_data) in zip(self.feedbacks_labels_pool, self.feedbacks_pool, self.prompts_pool, self.responses_pool):
                    response_dict = {
                        'feedback_label': feedback_label_data,
                        'feedback': feedback_data,
                        'prompt': prompt_data,
                        'response': response_data,
                    }
                    json.dump(response_dict, f)
                    f.write('\n')
            else:
                for (feedback_data, prompt_data, response_data) in zip(self.feedbacks_pool, self.prompts_pool, self.responses_pool):
                    response_dict = {
                        'feedback': feedback_data,
                        'prompt': prompt_data,
                        'response': response_data,
                    }
                    json.dump(response_dict, f)
                    f.write('\n')

    def save_data_to_files(self, save_path):
        # Save internal lists to separate files
        with open(f"{save_path}/prompts_pool.pkl", "wb") as f:
            pickle.dump(self.prompts_pool, f)
        with open(f"{save_path}/responses_pool.pkl", "wb") as f:
            pickle.dump(self.responses_pool, f)
        with open(f"{save_path}/feedbacks_pool.pkl", "wb") as f:
            pickle.dump(self.feedbacks_pool, f)
        if self.feedbacks_labels_pool:
            with open(f"{save_path}/feedbacks_labels_pool.pkl", "wb") as f:
                pickle.dump(self.feedbacks_labels_pool, f)
    
    def load_data_from_files(self, save_path):
        # Load data from files and repopulate internal lists
        with open(f"{save_path}/prompts_pool.pkl", "rb") as f:
            self.prompts_pool = pickle.load(f)
        with open(f"{save_path}/responses_pool.pkl", "rb") as f:
            self.responses_pool = pickle.load(f)
        with open(f"{save_path}/feedbacks_pool.pkl", "rb") as f:
            self.feedbacks_pool = pickle.load(f)
        if os.path.exists(f"{save_path}/feedbacks_labels_pool.pkl"):
            with open(f"{save_path}/feedbacks_labels_pool.pkl", "rb") as f:
                self.feedbacks_labels_pool = pickle.load(f)

    def serialize_to_dict(self, save_path):
        self.save_data_to_files(save_path)  # Ensure data is saved before storing references
        if self.feedbacks_labels_pool:
            state_dict = {
                "data_pool": {
                    "data_file_paths": {
                        "prompts": f"{save_path}/prompts_pool.pkl",
                        "responses": f"{save_path}/responses_pool.pkl",
                        "feedbacks": f"{save_path}/feedbacks_pool.pkl",
                        "feedbacks_labels": f"{save_path}/feedbacks_labels_pool.pkl",
                    },
                }
            }
        else:
            state_dict = {
                "data_pool": {
                    "data_file_paths": {
                        "prompts": f"{save_path}/prompts_pool.pkl",
                        "responses": f"{save_path}/responses_pool.pkl",
                        "feedbacks": f"{save_path}/feedbacks_pool.pkl",
                    },
                }
            }
        return state_dict

    def load_from_dict(self, state_dict):
        data_pool_info = state_dict["data_pool"]
        data_files = data_pool_info["data_file_paths"]
        self.prompts_pool = pickle.load(open(data_files["prompts"], "rb"))
        self.responses_pool = pickle.load(open(data_files["responses"], "rb"))
        self.feedbacks_pool = pickle.load(open(data_files["feedbacks"], "rb"))
        if "feedbacks_labels" in data_files: 
            self.feedbacks_labels_pool = pickle.load(open(data_files["feedbacks_labels"], "rb"))

class QuarkDataPool():
    def __init__(self, num_quantiles: int, reward_quantile_tokens: List[str]):
        self.datapool = {}
        self.num_quantiles = num_quantiles
        self.reward_quantile_tokens = reward_quantile_tokens
        self.EPSILON = 1e-9
    
    def flush_samples(self, drop_factor: float = 1.0):
        """
            drop_factor: within [0.0, 1.0], 0.0 being no flush and 1.0 being total flush
        """
        if abs(drop_factor - 0.0) < self.EPSILON:
            return
        
        if abs(drop_factor - 1.0) < self.EPSILON:
            self.datapool = {}
            
        else:
            # subsample the number of generations for each prompt 
            for prompt in self.datapool.keys():
                generations = self.datapool[prompt]["generations"]
                rewards = self.datapool[prompt]["rewards"]
                quantiles = self.datapool[prompt]["quantiles"]
                
                kept_generations = []
                kept_rewards = []
                kept_quantiles = []
                # flush samples uniformly for each quantile: even though the quantiles are equally sized 
                # (same number of samples each), keeping a small number of samples might miss-represent
                # the probability distribution
                for quantile in self.reward_quantile_tokens:
                    sublist_indices = [i for i, x in enumerate(quantiles) if x == quantile]
                    
                    sublist_generations = [generations[i] for i in sublist_indices]
                    sublist_rewards = [rewards[i] for i in sublist_indices]
                    sublist_quantiles = [quantiles[i] for i in sublist_indices]
                    
                    num_elem_to_keep = int(len(sublist_generations) * (1.0 - drop_factor))
                    all_indices = list(range(len(sublist_generations)))
                    random.shuffle(all_indices)
                    indices_to_keep = all_indices[:num_elem_to_keep]

                    sublist_generations = [sublist_generations[i] for i in indices_to_keep]
                    sublist_rewards = [sublist_rewards[i] for i in indices_to_keep]
                    sublist_quantiles = [sublist_quantiles[i] for i in indices_to_keep]
                                         
                    kept_generations.extend(sublist_generations)
                    kept_rewards.extend(sublist_rewards)
                    kept_quantiles.extend(sublist_quantiles)
                
                self.datapool[prompt]["generations"] = kept_generations
                self.datapool[prompt]["rewards"] = kept_rewards
                self.datapool[prompt]["quantiles"] = kept_quantiles
        
    def add_samples(self, prompts: List[str], generations: List[List[str]], rewards: List[List[float]]):
        for i, prompt in enumerate(prompts):
            if prompt not in self.datapool:
                self.datapool[prompt] = {
                    "generations": generations[i],
                    "rewards": rewards[i],
                    "quantiles": [],
                }
            else:
                self.datapool[prompt]["generations"].extend(generations[i])
                self.datapool[prompt]["rewards"].extend(rewards[i])
                self.datapool[prompt]["quantiles"] = []
    
    def map_into_quantiles(self):
        for prompt in self.datapool.keys():
            # sorting the rewards
            generations = self.datapool[prompt]["generations"]
            rewards = self.datapool[prompt]["rewards"]
            data = zip(generations, rewards)
            data = [x for x in data if x[-1] is not None]
            sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
            
            # updating lists based on order
            self.datapool[prompt]["generations"], self.datapool[prompt]["rewards"] = [list(x) for x in list(zip(*sorted_data))]
            
            # divide into quantiles of roughly equal size (last quantile will be larger if the length of the data is not 
            # divisible by the desired number of quantiles), and obtain the associated quantile index for each generation
            # e.g., currently the data pool has length 14 and we want to use 5 quantiles (the last four '4's are added as 14 % 5 != 0)
            quantiles = [[q_idx] * (len(sorted_data) // self.num_quantiles) for q_idx in range(self.num_quantiles)] # -> [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
            quantiles = [elem for sublist in quantiles for elem in sublist] # unfold list of lists into a single list -> [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
            quantiles = quantiles + [self.num_quantiles - 1] * (len(sorted_data) - len(quantiles)) # append indices for the last quantile -> [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4]
            
            # map quantile indices to quantile tokens and assign to datapool
            self.datapool[prompt]["quantiles"] = [self.reward_quantile_tokens[q_idx] for q_idx in quantiles]
    
    def update_datapool(self, sampling_file: Union[str, os.PathLike], drop_factor: float = 1.0):
        
        # flush previously sampled data
        self.flush_samples(drop_factor=drop_factor)
        
        # get newly sampled data
        prompts, all_generations, all_rewards = [], [], []
        with open(sampling_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                entry = json.loads(line)
                prompt = entry['prompt']
                generations = entry['generations']
                rewards = entry['rewards']
                prompts.append(prompt)
                all_generations.append(generations)
                all_rewards.append(rewards)
        
        # add data to datapool
        self.add_samples(prompts, all_generations, all_rewards)
        
        # sort data in datapool by reward and map into quantiles (individually for each prompt)
        self.map_into_quantiles()
    
    def get_samples(self, num_samples_per_quantile: Optional[int] = None) -> List[Dict[str, List[str]]]:
        
        samples = []

        for prompt in self.datapool.keys():
            generations = self.datapool[prompt]["generations"]
            quantiles = self.datapool[prompt]["quantiles"]
            
            if not num_samples_per_quantile:
            # return all generations for each prompt
                sampled_generations = generations
                sampled_quantiles = quantiles
            else:
            # subsample the number of generations for each prompt 
                sampled_generations = []
                sampled_quantiles = []
                # sample samples uniformly for each quantile: even though the quantiles are equally sized 
                # (same number of samples each), sampling a small number of samples might miss-represent
                # the probability distribution
                for quantile in self.reward_quantile_tokens:
                    sublist_indices = [i for i, x in enumerate(quantiles) if x == quantile]

                    sublist_generations = [generations[i] for i in sublist_indices]
                    sublist_quantiles = [quantiles[i] for i in sublist_indices]

                    num_elem_to_keep = min(num_samples_per_quantile, len(sublist_generations))
                    all_indices = list(range(len(sublist_generations)))
                    random.shuffle(all_indices)
                    indices_to_keep = all_indices[:num_elem_to_keep]

                    sublist_generations = [sublist_generations[i] for i in indices_to_keep]
                    sublist_quantiles = [sublist_quantiles[i] for i in indices_to_keep]

                    sampled_generations.extend(sublist_generations)
                    sampled_quantiles.extend(sublist_quantiles)

            return_dict = {
                "prompt": prompt,
                "generations": deepcopy(sampled_generations),
                "quantiles": deepcopy(sampled_quantiles)
            }
            samples.append(return_dict)
            
        return samples
    
    def save_datapool_to_disk(self, save_path: Union[str, os.PathLike]):
        with open(f"{save_path}/datapool.pkl", "wb") as f:
            pickle.dump(self.datapool, f)
            
    def serialize_to_dict(self, save_path: Union[str, os.PathLike]):
        self.save_datapool_to_disk(save_path)
        state_dict = {
            "datapool": {
                "data_file_path": f"{save_path}/datapool.pkl",
                "num_quantiles": self.num_quantiles,
                "reward_quantile_tokens": self.reward_quantile_tokens
            }
        }
        return state_dict
    
    def load_from_dict(self, state_dict):
        data_file_path = state_dict["datapool"]["data_file_path"]
        num_quantiles = state_dict["datapool"]["num_quantiles"]
        reward_quantile_tokens = state_dict["datapool"]["reward_quantile_tokens"]
        self.datapool = pickle.load(open(data_file_path, "rb"))
        self.num_quantiles = num_quantiles
        self.reward_quantile_tokens = reward_quantile_tokens
        
    def get_data_statistics(self, save_path: Union[str, os.PathLike], tokenizer: AutoTokenizer, num_bins=50):
        # compute reward and generations length statistics for every quantile accross all prompts
        reward_stats = []
        length_stats = []
        for prompt in self.datapool.keys():
            generations = self.datapool[prompt]["generations"]
            rewards = self.datapool[prompt]["rewards"]
            quantiles = self.datapool[prompt]["quantiles"]
            
            reward_quantile_stats = {}
            length_quantile_stats = {}
            for quantile in self.reward_quantile_tokens:
                sublist_indices = [i for i, x in enumerate(quantiles) if x == quantile]

                sublist_generations = [generations[i] for i in sublist_indices]
                sublist_rewards = [rewards[i] for i in sublist_indices]
                
                encoded_generations = tokenizer(sublist_generations)["input_ids"]
                generations_length = [len(encoded_generation) for encoded_generation in encoded_generations]
                sublist_rewards = np.array(sublist_rewards)
                
                reward_mean = np.mean(sublist_rewards)
                reward_std = np.std(sublist_rewards)
                reward_hist, reward_bins = np.histogram(sublist_rewards, bins=num_bins)
                reward_quantile_stats[quantile] = {
                    "mean": reward_mean,
                    "std": reward_std,
                    "hist": reward_hist,
                    "bins": reward_bins
                }
                
                len_mean = np.mean(generations_length)
                len_std = np.std(generations_length)
                len_hist, len_bins = np.histogram(generations_length, bins=num_bins)
                length_quantile_stats[quantile] = {
                    "mean": len_mean,
                    "std": len_std,
                    "hist": len_hist,
                    "bins": len_bins
                }
            
            reward_stats.append(reward_quantile_stats)
            length_stats.append(length_quantile_stats)
        
        for quantile in self.reward_quantile_tokens:
            r_mean, r_std, r_histograms, r_bins = [], [], [], []
            l_mean, l_std, l_histograms, l_bins = [], [], [], []
            for r_stats, l_stats in zip(reward_stats, length_stats):
                
                r_mean.append(r_stats[quantile]["mean"])
                r_std.append(r_stats[quantile]["std"])
                r_histograms.append(r_stats[quantile]["hist"])
                r_bins.append(r_stats[quantile]["bins"])
                
                l_mean.append(l_stats[quantile]["mean"])
                l_std.append(l_stats[quantile]["std"])
                l_histograms.append(l_stats[quantile]["hist"])
                l_bins.append(l_stats[quantile]["bins"])
            
            r_mean = np.mean(r_mean)
            r_std = np.mean(r_std)
            avg_r_hist, new_r_bins = average_histograms(r_histograms, r_bins)
            
            l_mean = np.mean(l_mean)
            l_std = np.mean(l_std)
            avg_l_hist, new_l_bins = average_histograms(l_histograms, l_bins)
            
            # Plot and save histograms with mean and std
            plt.figure(figsize=(18, 6))

            # Reward Histogram
            plt.subplot(1, 2, 1)
            plt.bar(new_r_bins[:-1], avg_r_hist, width=np.diff(new_r_bins), align="edge", color='salmon', edgecolor='black', alpha=0.7)
            plt.axvline(r_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {r_mean:.2f}')
            plt.axvspan(r_mean - r_std, r_mean + r_std, alpha=0.2, color='red', label=f'Std: {r_std:.2f}')
            plt.title('Reward Histogram')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.legend()

            # Generations Lengths Histogram
            plt.subplot(1, 2, 2)
            plt.bar(new_l_bins[:-1], avg_l_hist, width=np.diff(new_l_bins), align="edge", color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(l_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {l_mean:.2f}')
            plt.axvspan(l_mean - l_std, l_mean + l_std, alpha=0.2, color='red', label=f'Std: {l_std:.2f}')
            plt.title('Generations Length Histogram')
            plt.xlabel('len')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{save_path}/reward_len_hist_{quantile}.png")
            plt.close()
    
