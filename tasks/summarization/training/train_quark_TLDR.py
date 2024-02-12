import logging
import os
import argparse
import yaml
import json
from datetime import datetime
from typing import List, Tuple, Optional, Union
from pathlib import Path
import time
import sys
import shutil
from copy import deepcopy

from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler, DataCollatorWithPadding, GenerationConfig
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
import random

from utils import set_seed, ensure_dir, ceil_div, reduce_sum, reduce_mean, WANDB_API_KEY
from tasks.summarization.models.policy import Policy
from tasks.summarization.models.reward import GPTRewardModel, MyRMDataset, MyRMDataCollator
from tasks.summarization.datasets.sampling_dataset_and_collator import TLDRSamplingDataset, QuarkTLDRSamplingPromptCollatorWithPadding
from training_dataset_and_collator import QuarkTrainingDataset, QuarkTrainingSequenceCollatorWithPadding
from data_pool import QuarkDataPool

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)


logging.basicConfig(steam=sys.stdout, level=logging.INFO) # log levels, from least severe to most severe, are: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
log = logging.getLogger(__name__)

class QuarkTrainer:
    def __init__(self,
                 params: dict,
                 policy: Policy,
                 ref_policy: Policy,
                 reward_model: GPTRewardModel,
                 reward_tokenizer: AutoTokenizer,
                 data_pool: QuarkDataPool,
                 quantile_tokens: List[str],
                 sampling_train_dataloader: DataLoader,
                 sampling_dev_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 generation_config: GenerationConfig,
                 ) -> None:
        
        self.params = params
        self.num_quantiles = params['train']['num_quantiles']
        
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.data_pool = data_pool
        self.sampling_train_dataloader = sampling_train_dataloader
        self.sampling_dev_dataloader = sampling_dev_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.generation_config = generation_config

        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        self.quantile_tokens = quantile_tokens
        self.best_quantile_token = self.quantile_tokens[0]
        self.best_quantile_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_quantile_token)

        self.training_dataloader, self.training_sampler = None, None
        self.training_seq_collator = QuarkTrainingSequenceCollatorWithPadding(tokenizer=policy.tokenizer)

        self.rm_collator = MyRMDataCollator(tokenizer=self.reward_tokenizer, max_length=self.reward_tokenizer.max_length)

    def remove_quantile_from_prompt_input_ids(self,
                                              input_ids: torch.Tensor,
                                              attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            input_ids: tensor of shape (batch_size, seq_length) with left-padding and a prepended reward quantile token.
                e.g., [50256, 50256, 50256,    0,  35,  43,  96, 115] -> 0 to be removed
                      [50256, 50256,     0, 3445, 245,  15, 4900, 86] -> 0 to be removed
                      [    0, 1105,     24, 1111,  25, 902, 1500, 10] -> 0 to be removed  
                    input_ids.shape = [3, 8] -> [3, 7]
            attention_mask: tensor of shape (batch_size, seq_length) with left-padding and a prepended reward quantile token attention.
                e.g., [0, 0, 0, 1, 1, 1, 1, 1] -> first 1 to be removed
                      [0, 0, 1, 1, 1, 1, 1, 1] -> first 1 to be removed
                      [1, 1, 1, 1, 1, 1, 1, 1] -> first 1 to be removed   
                      attention_mask.shape = [3, 8] -> [3, 7]
        """
        batch_size, seq_length = input_ids.shape
        first_att_idxs = torch.argmax(attention_mask, dim=1).unsqueeze(1) # shape (batch_size, 1)
        # define boolean masking
        mask = torch.arange(seq_length).unsqueeze(0) != first_att_idxs # shape (batch_size, seq_length)
        # e.g., [True,  True,  True, False, True, True, True, True]
        #       [True,  True, False,  True, True, True, True, True]
        #       [False, True,  True,  True, True, True, True, True]
        input_ids = input_ids[mask].reshape(batch_size, -1)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return (input_ids, attention_mask)
        
    def decode(self, 
               tokenizer: AutoTokenizer,
               prompt_input_ids: torch.Tensor,
               generation_input_ids: Optional[torch.Tensor] = None,
               skip_special_tokens=True) -> Union[List[str], Tuple[List[str], List[str]]]:
        
        prompts = tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
        if generation_input_ids is None:
            return prompts
        
        generations = tokenizer.batch_decode(generation_input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
        return (prompts, generations)

    def sample(self, step_num):
        if step_num % self.params['train']['sample_interval'] != 0:
            return
        
        print(f"[step {step_num}] | Sampling stage...")

        if step_num == 0:
            # in the 1st sampling phase, use collate_fn that collated batches of data without reward quantile tokens
            collate_fn = lambda batch: collate_fn_wrapper(batch, best_quantile=True, conditioning=False)     
        else:
            # in subsequent sampling phases, use collate_fn that collates batches of data with reward quantile tokens
            collate_fn = lambda batch: collate_fn_wrapper(batch, best_quantile=True, conditioning=True)
        
        train_dataloader.collate_fn = collate_fn

        prompts, prompts_quantile, generations = [], [], []
        for i, batch in enumerate(tqdm(self.sampling_train_dataloader, total=len(self.sampling_train_dataloader), desc='Sampling from current policy')):
            input_ids, attention_mask = batch["inputs"]
            prompts_batch = batch["prompts"]

            rollouts = self.policy.sample(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                generation_config=generation_config)
            
            prompts_quantile_batch = self.decode(tokenizer=self.policy.tokenizer, query_input_ids=input_ids, skip_special_tokens=False)
            generations_batch = rollouts["generated_text"]
        
            prompts.extend(prompts_batch)
            generations.extend(generations_batch)
            prompts_quantile.extend(prompts_quantile_batch)

        # rewards are computed on prompts without reward quantile tokens
        samples = [prompt + generation for prompt, generation in zip(prompts, generations)]
        rm_dataset = MyRMDataset(samples=samples)
        rm_dataloader = DataLoader(
            rm_dataset, 
            shuffle=False, 
            batch_size=self.params['rewards']['batch_size'], 
            collate_fn=self.rm_collator)

        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(rm_dataloader), total=len(rm_dataloader)):
                for x in rm_batch:
                    rm_batch[x] = rm_batch[x].cuda()
                rewards_batch = self.reward_model.get_reward(**rm_batch)
                rewards.extend(rm_batch)
        
        # data_pool also receives prompts without rewards quantile tokens, as it orders the data points according to their rewards and then assigns a reward quantile token each to them
        self.data_pool.add(prompts=prompts, responses=generations, scores=rewards)

        # save sampling data and training data in json files for inspection (reward quantile tokens used during sampling)

        # 1. save tuples of (prompt_quantile, prompt, generation, rewards) in reward_file
        reward_file = Path(self.params['reward_dir']) / f"quark_sampling_data_step_{step_num}.json"
        with reward_file.open('a') as f:
            for idx, (prompt_quantile_data, prompt_data, generation_data, reward_data) in enumerate(zip(prompts_quantile, prompts, generations, rewards)):
                response_dict = {
                    'prompt_quantile': prompt_quantile_data,
                    'prompt': prompt_data,
                    'generation': generation_data,
                    'reward': reward_data
                }
                json.dump(response_dict, f)
                f.write('\n')

        # 2. save tuples of (prompt_quantile, prompt, generation, rewards) in reward_file (reward quantile tokens used during training)
        self.data_pool.save_data_for_training_in_json(self.params['reward_dir'], step_num)

        training_dataset = QuarkTrainingDataset(data_pool=self.data_pool, tokenizer=self.policy.tokenizer)
        self.training_dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=self.params['train']['training_batch_size_per_card'],
            shuffle=True,
            drop_last=True,
            collate_fn=self.training_seq_collator
        )
        self.training_sampler = iter(self.training_dataloader)

    def step(self, step_num):
        step_started_at = time.time()
        self.policy.model.eval()
        self.sample(step_num)
        self.policy.model.train()

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params['train']['training_batch_size_per_card'], 'insufficent batch'

        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)  # reset iteration to the beginning of data
            batch = next(self.sampler)

        self.optimizer.zero_grad()
        loss, stats = self.loss(step_num, *batch)
        loss.backward()

        if self.params['train']['clip_grad']:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params['train']['max_grad_norm'])

        self.optimizer.step()
        self.scheduler.step()

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params['train']['training_batch_size_per_card']) / step_time
        print(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")     

        # --- EVALUATION ---
        self.policy.model.eval()
        eval_metric = self.eval(step_num) # eval_metric returned as the avg score between all RMs on all dev samples

       # --- LOGGING ---
        if self.params['logging']['wandb_log']:
            for metric in ['kl', 'entropy']:
                wandb.log({f'Objective/{metric}': stats[f'objective/{metric}']}, step=step_num)

            for metric in ['lm', 'kl', 'entropy', 'total']:
                wandb.log({f'Loss/{metric}': stats[f'loss/{metric}']}, step=step_num)

            wandb.log({f'Params/lr': self.optimizer.param_groups[0]['lr']}, step=step_num)

        # --- SAVING ---
        # Save the top models
        self.save(step_num, eval_metric)

    
    def loss(self, step_num, query_input_ids, query_mask, response_input_ids, response_mask):
        # query_input_ids, query_mask have already NLF / Reward quantiles tokens prepended
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['lm_loss'], outputs['generated_logprobs'], outputs['generated_entropy'], outputs['generated_logits']
        if not self.nlf_cond: # Quark-based
            flattened_feedback_types = [feedback for feedback_type in self.feedback_types for feedback in feedback_type]
            logits = logits[:, :, :-len(flattened_feedback_types)]
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            query_input_ids, query_mask = self.remove_any_feedback_from_prompt_input_ids(input_ids=query_input_ids, 
                                                                                         attention_mask=query_mask, 
                                                                                         tokenizer=self.policy.tokenizer,
                                                                                         nlf_cond=self.nlf_cond)
            ref_outputs = self.ref_policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
            ref_logprobs, ref_logits = ref_outputs['generated_logprobs'], ref_outputs['generated_logits']

        kl = torch.sum(self.kl_loss(F.log_softmax(logits, dim=-1), F.softmax(ref_logits, dim=-1)), dim=-1)
        loss = reduce_mean(lm_loss + self.params['env']['kl_coef']*kl - self.params['env']['entropy_coef']*entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = self.decode(self.policy.tokenizer, query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step_num=step_num)

        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step_num):
        if step_num % self.params['logging']['log_interval'] != 0:
            return

        print(f"[step {step_num}] Printing samples examples ...")
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(f"\nSample {i+1}")
            print(f"{queries[i]} | {responses[i]}")
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params['env']['kl_coef'] * sample_kl:+.2f}")

    import heapq

    def save(self, step_num, eval_metric):
        if step_num % self.params['logging']['save_interval'] != 0:
            return

        eval_metric = float(eval_metric)

        # --- TOP MODELS ---
        if len(self.top_models) < self.top_models_limit:
            model_filename = f'{self.params["model_dir"]}/model_metric_{eval_metric}_step_{step_num}.pth'
            # If the list of top models is not full, add the current model to the queue
            heapq.heappush(self.top_models, (eval_metric, {
                'eval_metric': eval_metric,
                'model_name': model_filename,
                'step': step_num
            }))
            # Save model checkpoint to disk
            torch.save({
                'eval_metric': eval_metric,
                'policy_model': self.policy.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step_num
            }, model_filename)
            print(f"Saved checkpoint with metric {eval_metric} at step {step_num}")
        else:
            # If the list is full, compare the current model with the worst (lowest metric) in the top models queue
            worst_model = heapq.heappop(self.top_models)

            if eval_metric > worst_model[0]:
                # Remove worst model checkpoint from disk
                worst_model_filename = worst_model[1]['model_name']
                if os.path.exists(worst_model_filename):
                    os.remove(worst_model_filename)
                    print(f"The checkpoint {worst_model_filename} has been removed.")
                else:
                    print(f"The checkpoint {worst_model_filename} does not exist.")

                model_filename = f'{self.params["model_dir"]}/model_metric_{eval_metric}_step_{step_num}.pth'
                # Replace the worst model in the queue with the current model
                heapq.heappush(self.top_models, (eval_metric, {
                    'eval_metric': eval_metric,
                    'model_name': model_filename,
                    'step': step_num
                }))
                # Save model checkpoint to disk
                torch.save({
                    'eval_metric': eval_metric,
                    'policy_model': self.policy.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'step': step_num
                }, model_filename)
                print(f"Saved checkpoint with metric {eval_metric} at step {step_num}")
            else:
                # Put the worst model back in the heap queue (it hasn't been erased from disk)
                heapq.heappush(self.top_models, worst_model)

    def eval(self, step_num) -> Union[float, None]:
        if step_num % self.params['logging']['eval_interval'] != 0:
            return None
        print(f"[step {step_num}] Evaluating ...")

        prompts, responses = [], []
        references = []
        for i, batch in enumerate(tqdm(self.dev_dataloader, desc='(eval) Sampling from current policy')):
            with torch.no_grad():
                input_ids, attention_mask = batch["inputs"]
                references.extend(batch["references"])

                input_ids_feedback, attention_mask = self.add_feedback_to_prompt_input_ids(
                    input_ids=input_ids, attention_mask=attention_mask,
                    tokenizer=self.policy.tokenizer,
                    feedback=None,
                    best_feedback=True,
                    feedback_quantiles=None,
                    nlf_cond=self.nlf_cond
                )
                rollouts = self.policy.sample(prompts_input_ids=input_ids_feedback,
                                              prompts_attention_mask=attention_mask,
                                              do_sample=self.params['model']['policy_model']['eval_generation_kwargs']['do_sample'],
                )
                response = rollouts["generated_text"]
                prompt = self.decode(tokenizer=self.policy.tokenizer, query_input_ids=input_ids)

                prompts.extend(prompt)
                responses.extend(response)

        eval_output = self.score_model.get_metrics(
            prompt_texts=prompts, 
            generated_texts=responses, 
            batch_size=self.params['reward']['batch_size'],
            references=references
        )
        # RELEVANCY
        n_sub_sentences = eval_output["n_sub_sentences"]
        n_corrects_rel = eval_output["n_corrects_rel"]
        rel_rewards = eval_output["rel_rewards"]

        rel_scores_mean_over_num_samples = np.mean(rel_rewards) # averaged output score for all dev_set samples 
        rel_scores_mean_over_num_sentences = np.sum(rel_rewards) / np.sum(n_sub_sentences) # averaged output score for all dev_set senteces
        rel_correct_ratio = np.sum(n_corrects_rel) / np.sum(n_sub_sentences) # percentage of al sentences in the dev_set predicted as "no error"

        print(f"Averaged Relevancy score for all dev_set samples = {rel_scores_mean_over_num_samples:+.2f}")
        print(f"Averaged Relevancy score for all dev_set sentences = {rel_scores_mean_over_num_sentences:+.2f}")
        print(f"(rel) Percentage of all subsentences in the dev_set predicted as 'no error' = {rel_correct_ratio:+.2f}")

        # FACTUALITY
        n_sentences = eval_output["n_sentences"]
        n_corrects_fact = eval_output["n_corrects_fact"]
        fact_rewards = eval_output["fact_rewards"]

        fact_scores_mean_over_num_samples = np.mean(fact_rewards) # averaged output score for all dev_set samples 
        fact_scores_mean_over_num_sentences = np.sum(fact_rewards) / np.sum(n_sentences) # averaged output score for all dev_set senteces
        fact_correct_ratio = np.sum(n_corrects_fact) / np.sum(n_sentences) # percentage of al sentences in the dev_set predicted as "no error"
        
        print(f"Averaged Factuality score for all dev_set samples = {fact_scores_mean_over_num_samples:+.2f}")
        print(f"Averaged Factuality score for all dev_set sentences = {fact_scores_mean_over_num_sentences:+.2f}")
        print(f"(fact) Percentage of all sentences in the dev_set predicted as 'no error' = {fact_correct_ratio:+.2f}")

        # COMPLETENESS
        comp_rewards = eval_output["comp_rewards"]

        comp_scores_mean_over_num_samples = np.mean(comp_rewards) # averaged output score for all dev_set samples 

        print(f"Averaged Completeness score for all dev_set samples = {comp_scores_mean_over_num_samples:+.2f}")

        # OTHERS
        generations_lens = eval_output["generations_lens"]
        rouge_scores = eval_output["rouge_scores"]

        avg_generations_lens = np.mean(generations_lens)
        avg_rouge_scores = np.mean(rouge_scores)

        print(f"Average generations lenght = {avg_generations_lens:+.2f}")
        print(f"Average RougeLSum = {avg_rouge_scores:+.2f}")

        if self.params['logging']['wandb_log']:
            wandb.log({f'Evaluation/rel_scores_mean_over_num_samples': rel_scores_mean_over_num_samples}, step=step_num)
            wandb.log({f'Evaluation/rel_scores_mean_over_num_sentences': rel_scores_mean_over_num_sentences}, step=step_num)
            wandb.log({f'Evaluation/rel_correct_ratio': rel_correct_ratio}, step=step_num)

            wandb.log({f'Evaluation/fact_scores_mean_over_num_samples': fact_scores_mean_over_num_samples}, step=step_num)
            wandb.log({f'Evaluation/fact_scores_mean_over_num_sentences': fact_scores_mean_over_num_sentences}, step=step_num)
            wandb.log({f'Evaluation/fact_correct_ratio': fact_correct_ratio}, step=step_num)

            wandb.log({f'Evaluation/comp_scores_mean_over_num_samples': comp_scores_mean_over_num_samples}, step=step_num)

            wandb.log({f'Evaluation/avg_len': avg_generations_lens}, step=step_num)
            wandb.log({f'Evaluation/avg_RougeLSum': avg_rouge_scores}, step=step_num)

        min_rel, max_rel, min_fact, max_fact, min_comp, max_comp = -0.3, 0.3, -0.5, 0.5, -0.3, 0.3
        norm_rel = (rel_scores_mean_over_num_sentences - min_rel) / (max_rel - min_rel) 
        norm_fact = (fact_scores_mean_over_num_sentences - min_fact) / (max_fact - min_fact)
        norm_comp = (comp_scores_mean_over_num_samples - min_comp) / (max_comp - min_comp)
        avg_score = (norm_rel + norm_fact + norm_comp) / 3
        return avg_score

def main():
    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    # Set GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")

    # Set wandb logging
    wandb_log = args['logging']['wandb_log']
    if wandb_log:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            entity=args['logging']['wandb_entity'],
            project=args['logging']['wandb_project'],
            name=f"{args['logging']['run_name']}_{date_time}"
        )
    # Set saving directories
    args['save_dir'] = os.path.join(args['logging']['save_dir'], date_time)
    args['reward_dir'] = os.path.join(args['save_dir'], 'reward')
    args['model_dir'] = os.path.join(args['save_dir'], 'model')
    print(f"Writing to output directory: {args['save_dir']}")
    for dir in [args['save_dir'], args['reward_dir'], args['model_dir']]:
        ensure_dir(dir)

    # Save the config file
    with open(os.path.join(args['save_dir'], 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)
    
    
    print(f'Initializing models ...')

    # -------------- Initialize Tokenizer --------------
    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['tokenizer']['name_or_path'],
        padding_side=args['model']['policy_model']['input_padding_side'], # left padding
        model_max_length=args['train']['max_input_length']) # GPT2Tokenizer -> vocab_size 50257 (id from 0 to 50256) + extra_tokens for efficiency (id from 50257 to 50399) -> 50400 total vocabulary 
    
    if tokenizer.pad_token is None:
        print("Setting PAD token to EOS token for open-ended generation.")
        tokenizer.pad_token = tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    num_quantiles = args['train']['num_quantiles']
    quantile_tokens =  [f"_TREE_TOKEN_{str(quantile_idx)}" for quantile_idx in range(num_quantiles)]

    # add special reward quantile tokens to the tokenizer
    tokenizer.add_tokens(quantile_tokens, special_tokens=True)
    bad_words_ids = [[tokenizer.convert_tokens_to_ids(quantile_token)] for quantile_token in quantile_tokens]

    # -------------- Initialize Reference Policy --------------
    ref_policy = Policy(
        model_checkpoint_name=args['model']['ref_policy']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )

    # -------------- Initialize Policy to be finetuned --------------
    policy = Policy(
        model_checkpoint_name=args['model']['policy_model']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    # resize token_embeddings associated to the newly added tokens
    weights = policy.model.get_input_embeddings().weight.detach().cpu().numpy()
    mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
    new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in flattened_feedback_types])

    policy.model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        new_inits = torch.tensor(new_inits)
        policy.model.get_input_embeddings().weight[-len(flattened_feedback_types):, :] = new_inits

    if args['model']['policy_model']['load_state_dict']:
        state_dict = torch.load(args['model']['policy_model']['state_dict_path'])
        policy.model.load_state_dict(state_dict)

    generation_config = GenerationConfig(
        max_length = args["model"]["policy_model"]["train_generation_kwargs"]["max_length"],
        max_new_tokens = args["model"]["policy_model"]["train_generation_kwargs"]["max_new_tokens"],
        do_sample = args["model"]["policy_model"]["train_generation_kwargs"]["do_sample"], # False means greedy decoding
        num_beams = args["model"]["policy_model"]["train_generation_kwargs"]["num_beams"], # no beam search
        temperature = args["model"]["policy_model"]["train_generation_kwargs"]["temperature"], 
        top_k = args["model"]["policy_model"]["train_generation_kwargs"]["top_k"], # number of highest prob. vocabulary tokens to keep for top-k filtering
        top_p = args["model"]["policy_model"]["train_generation_kwargs"]["top_p"], # if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top-P or higher are kept for generation
        bad_words_ids = bad_words_ids, # List[List[int]] -> useful for Quark-based to avoid sampling of newly added tokens | list of list of tokens ids that are not allowed to be generated
        num_return_sequences = args["model"]["policy_model"]["train_generation_kwargs"]["num_return_sequences"], # may be interesting to sample many completions for which to collect feedback    
        return_dict_in_generate = True,
        pad_token_id = tokenizer.pad_token_id, # error if not passed...
    )

    # -------------- Initialize Reward Model --------------
    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    if args['reward']['load_state_dict']:
        state_dict = torch.load(args['reward']['state_dict_path'])
        reward_model.load_state_dict(state_dict)
    
    max_length = args['reward']['max_length']
    reward_tokenizer = AutoTokenizer.from_pretrained(args['model']['tokenizer']['name_or_path'], padding_side="right")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.max_length = max_length
    
    reward_model.cuda()
    reward_model.eval()
    if args['reward']['half']:
        reward_model.half()

    # -------------- Initialize DataPool --------------
    data_pool = QuarkDataPool(
        reward_quantile_tokens=quantile_tokens, num_quantiles=num_quantiles
    )

    # -------------- Load Sampling datasets and dataloaders --------------
    print(f'Loading data ...')

    prompt_collator = QuarkTLDRSamplingPromptCollatorWithPadding(tokenizer=tokenizer, quantile_tokens=quantile_tokens)
    def collate_fn_wrapper(batch, best_quantile=True, conditioning=True):
        return prompt_collator(batch, best_quantile=best_quantile, conditioning=conditioning)

    splits = []
    if args['data']['train_split_name']:
        splits.append(args['data']['train_split_name'])
    if args['data']['valid_split_name']:
        splits.append(args['data']['valid_split_name'])
    if args['data']['test_split_name']:
        splits.append(args['data']['test_split_name'])

    sampling_datasets = TLDRSamplingDataset(
        local_or_remote_path=args['data']['name_or_path'],
        tokenizer=tokenizer,
        data_format=None,
        splits=splits,
        remote=args['data']['remote'])
    
    sampling_train_dataset = sampling_datasets[args['data']['train_split_name']]
    sampling_train_dataloader = DataLoader(
        dataset=sampling_train_dataset,
        batch_size=args['train']['sampling_batch_size_per_card'],
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: collate_fn_wrapper(batch, best_quantile=True, conditioning=False)
    )
    print(f"Sampling Train dataset loaded with {len(sampling_train_dataset)} samples | Sampling Train dataloader with {len(sampling_train_dataloader)} batches")

    sampling_dev_dataset = sampling_datasets[args['data']['dev_split_name']]
    sampling_dev_dataloader = DataLoader(
        dataset=sampling_dev_dataset,
        batch_size=args['train']['sampling_batch_size_per_card'],
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: collate_fn_wrapper(batch, best_quantile=True, conditioning=True)
    )
    print(f"Sampling Dev dataset loaded with {len(sampling_dev_dataset)} samples | Sampling Dev dataloader with {len(sampling_dev_dataloader)} batches")
    
    # -------------- Prepare Optimizer and Schedulers --------------

    # Freeze 70% of policy model backbone
    layers = policy.model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args['train']['lr'], eps = 1e-5)
    
    total_steps = ceil_div(args['train']['total_episodes'], args['train']['training_batch_size_per_card'])
    
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=args['train']['n_warmup_steps'],
        num_training_steps=total_steps
    )

    # -------------- Set up trainer --------------
    trainer = QuarkTrainer(
        params=args,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        data_pool=data_pool,
        quantile_tokens=quantile_tokens,
        sampling_train_dataloader=sampling_train_dataloader,
        sampling_dev_dataloader=sampling_dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        generation_config=generation_config,
    )

    steps = list(range(total_steps + 1))
    steps = tqdm(steps)
    for step_num in steps:
        try:
            trainer.step(step_num)
        except Exception as e:
            print("There was an Exception while trying to perform trainer.step()!")
            print(e)
            torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    main()

    
