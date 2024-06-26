import os
import json
import time
import logging
import random
import argparse
from typing import List
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
import wandb
from transformers import get_linear_schedule_with_warmup

from arguments import get_args
from policy import Policy
from data_pool import DataPool
from reward import Reward, reward_to_toxicity
from datasets_and_collators import PromptDataset, PromptCollator, SequenceDataset, SequenceCollator
from alt.utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness
from alt.utils.utils import WANDB_API_KEY

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) # log levels, from least severe to most severe, are: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
log = logging.getLogger(__name__)

class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult


class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: Policy,
                 data_pool: DataPool,
                 score_model: Reward,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.data_pool = data_pool
        self.score_model = score_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.best_cat = self.tree_tokens[0]
        self.best_cat_ids = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer)

    # MODIFIED
    def add_best_control_code(self, input_ids, attention_mask):
        """
        Prepend control tokens associated with the best performing quantile to a batch of input sequences.

        This function takes a batch of input token IDs and their corresponding attention masks and adds control tokens
        associated with the best performing quantile to the beginning of each input sequence. It also inserts a special
        <|separator|> token between the control tokens and the original input tokens (newly added as not contemplated within the GPT2Tokenizer).

        Args:
            self (object): The instance of the class containing this method.
            input_ids (torch.Tensor): A tensor containing token IDs for a batch of input sequences.
            attention_mask (torch.Tensor): A tensor containing attention masks for the input sequences.

        Returns:
            torch.Tensor: A tensor containing the modified input token IDs with control tokens prepended, and the separator token.
            torch.Tensor: A tensor containing the modified attention masks.

        Note:
            - `self.best_cat_ids` should be set to the control tokens associated with the best performing quantile.
            - The <|separator|> token is used to separate the control tokens from the input tokens.
        """
        input_ids = torch.cat([input_ids.new([self.best_cat_ids] * len(input_ids)),
                               input_ids.new([[self.policy.tokenizer.sep_token_id]]*len(input_ids)),
                                input_ids], dim=1)
        
        attention_mask = torch.cat([attention_mask.new([[1]*len(self.best_cat_ids)] * len(attention_mask)), 
                                    attention_mask.new([[1]]*len(attention_mask)),
                                    attention_mask], dim=1)

        return input_ids, attention_mask

    # NEWLY ADDED
    def remove_any_control_code(self, input_ids, attention_mask, rmv_sep_token=False):
        """
        Remove control tokens from a batch of input sequences.

        This function takes a batch of input token IDs and their corresponding attention masks and removes control tokens
        added for conditioning during generation. It also provides the option to remove the separator token.

        Args:
            self (object): The instance of the class containing this method.
            input_ids (torch.Tensor]): A tensor containing token IDs for a batch of input sequences.
            attention_mask (torch.Tensor]): A tensor containing attention masks for the input sequences.
            rmv_sep_token (bool, optional): Set to True to remove the separator token from the sequences.

        Returns:
            torch.Tensor]: A tensor containing the modified input token IDs with control tokens removed.
            torch.Tensor]: A tensor containing the modified attention masks.

        Note:
            - Control tokens are removed from each sequence, and the separator token can also be removed if specified.
        """

        bs, _ = input_ids.shape

        sep_token_id = self.policy.tokenizer.sep_token_id
        sep_token_mask = (input_ids == sep_token_id)
        cumulative_mask = sep_token_mask.cumsum(dim=1)
        tokens_after_special_mask = cumulative_mask > 0
        
        input_ids = input_ids[tokens_after_special_mask].reshape(bs, -1)
        attention_mask = attention_mask[tokens_after_special_mask].reshape(bs, -1)

        if rmv_sep_token:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
            
        return input_ids, attention_mask
    
    # MODIFIED
    def decode(self, query_input_ids, response_input_ids=None, skip_special_tokens=True):
        """
        Decode token sequences into human-readable text.

        This function takes token IDs or sequences and converts them into human-readable text using the tokenizer's decoding
        capabilities.

        Args:
            self (object): The instance of the class containing this method.
            query_input_ids (torch.Tensor or List[List[int]]): A tensor or list of token IDs representing input sequences.
            response_input_ids (torch.Tensor or List[List[int]], optional): A tensor or list of token IDs representing response
                sequences. If not provided (None), only the input sequences are decoded.

        Returns:
            List[str] or Tuple[List[str], List[str]]: If `response_input_ids` is provided, it returns a tuple containing two lists:
            1. List of decoded input sequences.
            2. List of decoded response sequences.
            If `response_input_ids` is not provided, it returns a list containing the decoded input sequences.
        """

        query = [self.policy.tokenizer.decode(p, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
                for p in query_input_ids]
            
        if response_input_ids is None:
            return query

        response = [self.policy.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for r in response_input_ids]
        return query, response

    # MODIFIED
    def sample(self, step):
        if step % self.params.sample_interval != 0:
            return
        log.info(f"[step {step}] Sampling ...")

        prompts, responses = [], []
        for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), desc='Sampling from current policy')):
            
            input_ids, attention_mask = batch

            if step == 0:
                rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p)
                prompt, response = rollouts['query/text'], rollouts['response/text']

            else:
                input_ids, attention_mask = self.add_best_control_code(input_ids, attention_mask)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p)
                response = rollouts['response/text']
                query_input_ids, _ = self.remove_any_control_code(input_ids, attention_mask, rmv_sep_token=True)
                prompt = self.decode(query_input_ids)

            prompts.extend(prompt)
            responses.extend(response)

        scores = self.score_model.get_reward(prompts, responses, f'step{step}') # this gives directly rewards (i.e., 1 - toxicity scores) !!!
        self.data_pool.add(prompts=prompts, responses=responses, scores=scores)

        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, drop_last=True, collate_fn=self.seq_collator)
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):
        step_started_at = time.time()
        self.sample(step=step_num)

        try:
            batch = next(self.sampler) # tuple of (query_input_ids, query_mask, response_input_ids, response_mask)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'

        except (StopIteration, AssertionError): # StopIteration -> if the iterator reaches the end of the data | AssertionError -> insufficient batch size
            self.sampler = iter(self.sample_dataloader) # This essentially resets the iterator to the beginning of the data...
            batch = next(self.sampler)

        self.optimizer.zero_grad()
        # as the batch is obtained from an iterator of the SequenceDataset containing data from the actual data pool using the SequenceCollator class
        # as a collator function, it already contains the corresponding NLF tokens prepended to each element of the batch,
        # with left padding (according to the maximum NLF tokens length on the batch) and the SEP token.
        ppo_loss, stats = self.loss(step_num, *batch)
        ppo_loss.backward()

        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        # --- LOGGING ---
        for metric in ['kl', 'entropy']:
            wandb.log({f'Objective/{metric}': stats[f'objective/{metric}']}, step=step_num)

        for metric in ['lm', 'kl', 'entropy', 'total']:
            wandb.log({f'Loss/{metric}': stats[f'loss/{metric}']}, step=step_num)

        wandb.log({f'Params/lr': self.optimizer.param_groups[0]['lr']}, step=step_num)
        wandb.log({f'Params/kl_coef': self.kl_ctl.value}, step=step_num)
        wandb.log({f'Params/entropy_coef': self.entropy_ctl.value}, step=step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True) # this does nothing if using a FixedController for the KL
        self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False) # this does nothing if using a FixedController for the Entorpy

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        log.info(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")     
        self.save(step=step_num)
        self.eval(step=step_num)

    # MODIFIED
    def loss(self, step, query_input_ids, query_mask, response_input_ids, response_mask):
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']        
        logits = outputs['response/logits'][:, :, :-1] # don't consider the newly added logit associated to the "<|separator|>" token
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            query_input_ids, query_mask = self.remove_any_control_code(query_input_ids, query_mask, rmv_sep_token=True)        
            ref_outputs = self.ref_policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
            ref_logprobs, ref_logits = ref_outputs['response/log_prob'], ref_outputs['response/logits']


        # Note 1: To avoid underflow issues when computing this quantity, this loss expects the argument 
        # input ('prediction') in the log-space. The argument target may also be provided in 
        # the log-space if log_target= True.
        # Note 2: As all the other losses in PyTorch, this function expects the first argument, input, 
        # to be the output of the model (e.g. the neural network) and the second, target, 
        # to be the observations in the dataset. This differs from the standard mathematical 
        # notation KL(P ∣∣ Q) where P denotes the distribution of the observations and Q denotes the model.

        # REVIEW THIS... I WOULD CHANGE THE ORDER OF THE ARGUMENTS!
        # the sum is taken just over the vocabulary tokens dimension, and would be averaged later using the response mask
        # kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)
        kl = torch.sum(self.kl_loss(F.log_softmax(logits, dim=-1), F.softmax(ref_logits, dim=-1)), dim=-1)
        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = self.decode(query_input_ids, response_input_ids) # query_input_ids has already had their NLF tokens removed
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step)

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

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples

        log.info(f"[step {step}] Printing samples examples ...")
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            log.info(f"\nSample {i+1}")
            log.info(f"{queries[i]} |{responses[i]}")
            log.info(f"  lm_loss = {lm_loss[i].item():+.2f}")
            log.info(f"  kl = {sample_kl:+.2f}")
            log.info(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save(self, step):
        if step % self.params.save_interval != 0:
            return
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        log.info(f"[step {step}] model checkpoint saved")

    # MODIFIED
    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[step {step}] evaluating ...")

        generations, perplexities, toxicities = [], [], []
        for i, (input_ids, attention_mask) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids, attention_mask = self.add_best_control_code(input_ids, attention_mask)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p)

                input_ids, attention_mask = self.remove_any_control_code(input_ids, attention_mask, rmv_sep_token=True)
                forward_inputs = {'query_input_ids': input_ids, # this has the NLF tokens and SEP token removed
                                  'query_mask': attention_mask,
                                  'response_input_ids': rollouts['response/input_ids'], # we set the SEP token logit to -inf so this ID cannot be predicted as the next token
                                  'response_mask': rollouts['response/mask']}
                
                ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']

                # WRONGLY COMPUTED?
                # perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                perplexity = torch.exp(-1 * reduce_mean(ref_logprobs, rollouts['response/mask'].float(), axis=1), dim=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                prompt = self.decode(input_ids) # input_ids has already had their NLF tokens removed
                response = rollouts['response/text']
                score = self.score_model.get_reward(prompt, response, f'step{step}_eval{i}')
                toxicity = [reward_to_toxicity(x) for x in score if x is not None]
                toxicities.extend(toxicity)

                generations.extend(rollouts['response/text'])

        ppl_score, toxicity_score = np.nanmean(perplexities), np.mean(toxicities)
        dist_1, dist_2, dist_3 = distinctness(generations)
        log.info(f"  perplexity = {ppl_score:+.2f}")
        log.info(f"  toxicity = {toxicity_score:+.2f}")
        log.info(f'dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}')
        wandb.log({f'Evaluation/perplexity': ppl_score}, step=step)
        wandb.log({f'Evaluation/toxicity': toxicity_score}, step=step)
        wandb.log({f'Evaluation/Dist-1': dist_1}, step=step)
        wandb.log({f'Evaluation/Dist-2': dist_2}, step=step)
        wandb.log({f'Evaluation/Dist-3': dist_3}, step=step)


def main():
    args = get_args() # args is an "argparse.Namespace" object

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="sauc-ms-thesis", config=args, name=date_time)

    args.save_dir = os.path.join(args.output_dir, date_time)
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir]:
        ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log.info(f'Initializing models ...')
    ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device)
    policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device, reward_cond=True)
    
    tags = ["Lowest Toxicity", "Low-Moderate Toxicity", "Moderate Toxicity", "High-Moderate Toxicity", "Maximum Toxicity"]
    tree_tokens = [policy.tokenizer.convert_ids_to_tokens(policy.tokenizer(tag)["input_ids"]) for tag in tags]
    log.info(f"Using {args.num_quantiles} quantiles, associated with the following Natural Language tags: {tags}")
    log.info(f"The tags are converted to the following tokens: {tree_tokens}")
    
    reward = Reward(save_path=args.reward_dir, rate_limit=args.perspective_rate_limit, batch_size=args.batch_size)
    data_pool = DataPool(tree_tokens=tree_tokens, num_quantiles=args.num_quantiles)
    log.info(f'Initialization done!')

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
    train_dataset = PromptDataset(path=args.dataset_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size*2, shuffle=True, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(path=args.dataset_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    # set up optimizer and scheduler
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-8)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size) # ((3,000,000 episodes - 1) // 128 bs) + 1 = 23,438 steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps) # wu 500 steps (~2%)

    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, data_pool=data_pool,
                               score_model=reward, tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               optimizer=optimizer, scheduler=scheduler)

    for step_num in range(args.total_steps):
        try:
            trainer.step(step_num)

        except Exception as e:
            log.info("There was an Exception while trying to perform trainer.step()!")
            log.info(e)
            torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    main()
