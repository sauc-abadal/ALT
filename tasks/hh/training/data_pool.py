from pathlib import Path
from typing import List, Optional, Dict, Union
import random
from copy import deepcopy
import os
import json
import pickle
import numpy as np
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
class NLFDataPool():
    def __init__(self, tokenizer: AutoTokenizer):
        self.datapool = {}
        self.EPSILON = 1e-9
        self.tokenizer = tokenizer
    
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
                feedbacks = self.datapool[prompt]["feedbacks"]
                
                num_elem_to_keep = int(len(generations) * (1.0 - drop_factor))
                all_indices = list(range(len(generations)))
                random.shuffle(all_indices)
                indices_to_keep = all_indices[:num_elem_to_keep]

                generations = [generations[i] for i in indices_to_keep]
                feedbacks = [feedbacks[i] for i in indices_to_keep]
            
                self.datapool[prompt]["generations"] = generations
                self.datapool[prompt]["feedbacks"] = feedbacks
        
    def add_samples(self, prompts: List[str], generations: List[List[str]], feedbacks: List[List[str]]):
        for i, prompt in enumerate(prompts):
            
            g = [gen for gen, feed in zip(generations[i], feedbacks[i]) if feed is not None]
            f = [feed for feed in feedbacks[i] if feed is not None]

            if prompt not in self.datapool:
                self.datapool[prompt] = {
                    "generations": g,
                    "feedbacks": f,
                }
            else:
                self.datapool[prompt]["generations"].extend(g)
                self.datapool[prompt]["feedbacks"].extend(f)
    
    def update_datapool(self, sampling_file: Union[str, os.PathLike], drop_factor: float = 1.0):
        
        # flush previously sampled data
        self.flush_samples(drop_factor=drop_factor)
        
        # get newly sampled data
        prompts, all_generations, all_feedbacks = [], [], []
        with open(sampling_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                entry = json.loads(line)
                prompt = entry['prompt']
                generations = entry['generations']
                feedbacks = entry['feedbacks']
                scores = entry['scores']
                prompts.append(prompt)
                all_generations.append(generations)
                all_feedbacks.append(feedbacks)
        
        # add data to datapool
        self.add_samples(prompts, all_generations, all_feedbacks)
    
    def get_samples(self, num_samples_per_prompt: Optional[int] = None) -> List[Dict[str, List[str]]]:
        
        samples = []
        
        for prompt in self.datapool.keys():
            generations = self.datapool[prompt]["generations"]
            feedbacks = self.datapool[prompt]["feedbacks"]
            
            if not num_samples_per_prompt:
            # return all generations for each prompt
                sampled_generations = generations
                sampled_feedbacks = feedbacks

            else:
            # subsample the number of generations for each prompt  
                sampled_generations = []
                sampled_feedbacks = []
                
                indices_used = []
                num_samples_to_draw = {
                    "Very helpful and harmless": num_samples_per_prompt // 6,
                    "Very helpful and harmful": num_samples_per_prompt // 6,
                    "Helpful and harmless": num_samples_per_prompt // 6,
                    "Helpful and harmful": num_samples_per_prompt // 6,
                    "Not helpful and harmless": num_samples_per_prompt // 6,
                    "Not helpful and harmful": num_samples_per_prompt // 6
                }
                feedback_categories = [
                    "Helpful and harmless",
                    "Not helpful and harmless",
                    "Harmful"
                    ]
                
                still_to_draw = num_samples_per_prompt % 6
                for feedback_category in feedback_categories:

                    sublist_indices = [i for i, x in enumerate(feedbacks) if x == feedback_category]
                    
                    sublist_generations = [generations[i] for i in sublist_indices]
                    sublist_feedbacks = [feedbacks[i] for i in sublist_indices]
                 
                    num_elem_to_keep = min(num_samples_to_draw[feedback_category], len(sublist_generations))
                    if num_elem_to_keep == 0:
                        still_to_draw += num_samples_to_draw[feedback_category]
                        continue
                        
                    gen_lens = [len(gen) for gen in self.tokenizer(sublist_generations)["input_ids"]]

                    all_indices = list(range(len(sublist_generations)))
                    # shuffle indices to get a random permutation of the generations
                    random.shuffle(all_indices)

                    # rejection sampling to get 'num_elements_to_keep' generations
                    # while rejecting generations with len 64 tokens
                    indices_to_keep = []
                    for idx in all_indices:
                        if gen_lens[idx] < 64:
                            indices_to_keep.append(idx)
                            if len(indices_to_keep) == num_elem_to_keep:
                                break

                    # if we didn't manage to get the 'num_elem_to_keep' generations, 
                    # fill with 64-tokens generations
                    if len(indices_to_keep) != num_elem_to_keep:
                        for idx in all_indices:
                            if idx not in indices_to_keep:
                                indices_to_keep.append(idx)
                                if len(indices_to_keep) == num_elem_to_keep:
                                    break
                                    
                    indices_used.extend([sublist_indices[i] for i in indices_to_keep])
                                        
                    sublist_generations = [sublist_generations[i] for i in indices_to_keep]
                    sublist_feedbacks = [sublist_feedbacks[i] for i in indices_to_keep]
                                        
                    sampled_generations.extend(sublist_generations)
                    sampled_feedbacks.extend(sublist_feedbacks)
                    
                    still_to_draw += num_samples_to_draw[feedback_category] - len(sublist_generations)
                                        
            if still_to_draw > 0:
                for idx, (g, f) in enumerate(zip(generations, feedbacks)):
                    if idx not in indices_used:
                        sampled_generations.append(g)
                        sampled_feedbacks.append(f)
                        still_to_draw -= 1
                        if still_to_draw == 0:
                            break
                            
            return_dict = {
                "prompt": deepcopy(prompt),
                "generations": deepcopy(sampled_generations),
                "feedbacks": deepcopy(sampled_feedbacks),
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
            }
        }
        return state_dict
    
    def load_from_dict(self, state_dict):
        data_file_path = state_dict["datapool"]["data_file_path"]
        self.datapool = pickle.load(open(data_file_path, "rb"))
    
    def get_num_samples(self):
        num_prompts = len(self.datapool.keys())
        total_generations = 0
        max_generations = 0
        min_generations = 1e5
        for prompt in self.datapool.keys():
            generations = self.datapool[prompt]["generations"]
            num_generations = len(generations)
            total_generations += num_generations
            if num_generations > max_generations:
                max_generations = num_generations
            if num_generations < min_generations:
                min_generations = num_generations

        return {
            "num_prompts": num_prompts,
            "total_generations": total_generations,
            "max_generations": max_generations,
            "min_generations": min_generations
        }
class QuarkDataPool():
    def __init__(self, num_quantiles: int, reward_quantile_tokens: List[str], tokenizer: AutoTokenizer):
        self.datapool = {}
        self.num_quantiles = num_quantiles
        self.reward_quantile_tokens = reward_quantile_tokens
        self.EPSILON = 1e-9
        self.tokenizer = tokenizer
    
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

                    gen_lens = [len(gen) for gen in self.tokenizer(sublist_generations)["input_ids"]]

                    all_indices = list(range(len(sublist_generations)))
                    # shuffle indices to get a random permutation of the generations
                    random.shuffle(all_indices)

                    # rejection sampling to get 'num_elements_to_keep' generations
                    # while rejecting generations with len 64 tokens
                    indices_to_keep = []
                    for idx in all_indices:
                        if gen_lens[idx] < 64:
                            indices_to_keep.append(idx)
                            if len(indices_to_keep) == num_elem_to_keep:
                                break
                    
                    # if we didn't manage to get the 'num_elem_to_keep' generations, 
                    # fill with 64-tokens generations
                    if len(indices_to_keep) != num_elem_to_keep:
                        for idx in all_indices:
                            if idx not in indices_to_keep:
                                indices_to_keep.append(idx)
                                if len(indices_to_keep) == num_elem_to_keep:
                                    break

                    sublist_generations = [sublist_generations[i] for i in indices_to_keep]
                    sublist_quantiles = [sublist_quantiles[i] for i in indices_to_keep]

                    sampled_generations.extend(sublist_generations)
                    sampled_quantiles.extend(sublist_quantiles)

            return_dict = {
                "prompt": deepcopy(prompt),
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

    def get_data_statistics(self, save_path: Union[str, os.PathLike], tokenizer: AutoTokenizer, num_bins=100):
        # compute min and max reward/length to align histogram edges
        min_reward, max_reward = 1000, -1000
        min_len, max_len = 5000, 0
        for prompt in self.datapool.keys():
            rewards = self.datapool[prompt]["rewards"]
            generations = self.datapool[prompt]["generations"]
            encoded_generations = tokenizer(generations)["input_ids"]
            generations_length = [len(encoded_generation) for encoded_generation in encoded_generations]
            min_r = np.min(rewards)
            max_r = np.max(rewards)
            min_l = np.min(generations_length)
            max_l = np.max(generations_length)
            min_reward = min(min_reward, min_r)
            max_reward = max(max_reward, max_r)
            min_len = min(min_len, min_l)
            max_len = max(max_len, max_l)
        
        edges_r = np.linspace(min_reward, max_reward, num_bins + 1)
        edges_l = np.linspace(min_len, max_len, num_bins + 1)

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
                reward_hist, _ = np.histogram(sublist_rewards, bins=edges_r)
                reward_quantile_stats[quantile] = {
                    "mean": reward_mean,
                    "std": reward_std,
                    "hist": reward_hist,
                }
                
                len_mean = np.mean(generations_length)
                len_std = np.std(generations_length)
                len_hist, _ = np.histogram(generations_length, bins=edges_l)
                length_quantile_stats[quantile] = {
                    "mean": len_mean,
                    "std": len_std,
                    "hist": len_hist,
                }
            
            reward_stats.append(reward_quantile_stats)
            length_stats.append(length_quantile_stats)
        
        for quantile in self.reward_quantile_tokens:
            r_mean, r_std, r_histograms = [], [], []
            l_mean, l_std, l_histograms = [], [], []
            for r_stats, l_stats in zip(reward_stats, length_stats):
                
                r_mean.append(r_stats[quantile]["mean"])
                r_std.append(r_stats[quantile]["std"])
                r_histograms.append(r_stats[quantile]["hist"])
                
                l_mean.append(l_stats[quantile]["mean"])
                l_std.append(l_stats[quantile]["std"])
                l_histograms.append(l_stats[quantile]["hist"])
            
            r_mean = np.mean(r_mean)
            r_std = np.mean(r_std)
            avg_r_hist = np.sum(r_histograms, axis=0) / len(r_histograms)
            
            l_mean = np.mean(l_mean)
            l_std = np.mean(l_std)
            avg_l_hist = np.sum(l_histograms, axis=0) / len(l_histograms)
            
            # Plot and save histograms with mean and std
            plt.figure(figsize=(18, 6), facecolor='white')

            # Reward Histogram
            plt.subplot(1, 2, 1)
            plt.bar(edges_r[:-1], avg_r_hist, width=np.diff(edges_r), align="edge", color='salmon', edgecolor='black', alpha=0.7)
            plt.axvline(r_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {r_mean:.2f}')
            plt.axvspan(r_mean - r_std, r_mean + r_std, alpha=0.2, color='red', label=f'Std: {r_std:.2f}')
            plt.title(f'Reward Histogram {quantile}')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.legend()

            # Generations Lengths Histogram
            plt.subplot(1, 2, 2)
            plt.bar(edges_l[:-1], avg_l_hist, width=np.diff(edges_l), align="edge", color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(l_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {l_mean:.2f}')
            plt.axvspan(l_mean - l_std, l_mean + l_std, alpha=0.2, color='red', label=f'Std: {l_std:.2f}')
            plt.title(f'Generations Length Histogram {quantile}')
            plt.xlabel('len')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{save_path}/reward_len_hist_{quantile}.png")
            plt.close()
    
    def get_num_samples(self):
        num_prompts = len(self.datapool.keys())
        total_generations = 0
        max_generations = 0
        min_generations = 1e5
        for prompt in self.datapool.keys():
            generations = self.datapool[prompt]["generations"]
            num_generations = len(generations)
            total_generations += num_generations
            if num_generations > max_generations:
                max_generations = num_generations
            if num_generations < min_generations:
                min_generations = num_generations

        return {
            "num_prompts": num_prompts,
            "total_generations": total_generations,
            "max_generations": max_generations,
            "min_generations": min_generations
        }
