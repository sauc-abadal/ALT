from typing import Dict, List, Tuple, Optional, Union
import time

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator

from alt.utils.utils import reduce_mean
from alt.models.policy import Policy

class QuarkTrainer_KL:
    def __init__(self,
                 params: dict,
                 policy: Policy,
                 ref_policy: Policy,
                 quantile_tokens: List[str],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 accelerator: Accelerator,
                 training_dataloader: DataLoader
                 ) -> None:
        
        self.params = params
        self.num_quantiles = params['train']['num_quantiles']
        self.policy = policy
        self.policy.model.train()
        self.ref_policy = ref_policy
        self.ref_policy.model.eval()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.training_dataloader = training_dataloader
              
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        self.quantile_tokens = quantile_tokens
        self.best_quantile_token = self.quantile_tokens[0]
        self.best_quantile_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_quantile_token)
        
        self.training_sampler = iter(self.training_dataloader)

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
        mask = torch.arange(seq_length, device=first_att_idxs.device).unsqueeze(0) != first_att_idxs # shape (batch_size, seq_length)
        # e.g., [True,  True,  True, False, True, True, True, True]
        #       [True,  True, False,  True, True, True, True, True]
        #       [False, True,  True,  True, True, True, True, True]
        input_ids = input_ids[mask].reshape(batch_size, -1)
        attention_mask = attention_mask[mask].reshape(batch_size, -1)
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

    def step(self, step_num) -> None:
        step_started_at = time.time()        

        try:
            batch = next(self.training_sampler) # dictionary with keys "inputs", "outputs", "prompts", "input_seqs", "output_seqs"
            assert len(batch["inputs"]["input_ids"]) == (self.params['train']['training_batch_size_per_card']*self.params['train']['num_samples_per_quantile']*self.params['train']['num_quantiles']), 'insufficent batch'

        except (StopIteration, AssertionError):
            self.training_sampler = iter(self.training_dataloader)  # reset iteration to the beginning of data
            batch = next(self.training_sampler)

        self.optimizer.zero_grad()

        inputs_dict = batch["inputs"]
        outputs_dict = batch["outputs"]
        
        loss, stats = self.loss(step_num, inputs_dict, outputs_dict)
        self.accelerator.backward(loss)

        if self.params['train']['clip_grad']:
            self.accelerator.clip_grad_norm_(self.policy.model.parameters(), self.params['train']['max_grad_norm'])

        self.optimizer.step()
        self.scheduler.step()

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params['train']['training_batch_size_per_card']) / step_time
        self.accelerator.print(f"[step {step_num}] | Training ... step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")     

       # --- LOGGING ---
        if self.params['logging']['wandb_log']:
            for metric in ['lm', 'kl', 'entropy', 'total']:
                self.accelerator.log({f'Loss/{metric}': stats[f'loss/{metric}']}, step=step_num)
            self.accelerator.log({f'Params/lr': self.optimizer.param_groups[0]['lr']}, step=step_num)
 
    def loss(self, step_num, inputs_dict, outputs_dict) -> Tuple[torch.Tensor, Dict[str, float]]:

        prompts_input_ids = inputs_dict["input_ids"]
        prompts_attention_mask = inputs_dict["attention_mask"]
        generations_input_ids = outputs_dict["input_ids"]
        generations_attention_mask = outputs_dict["attention_mask"]

        outputs = self.policy.forward_pass(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            generated_input_ids=generations_input_ids,
            generated_attention_mask=generations_attention_mask
        )

        generated_logits = outputs["generated_logits"] # shape (bs, gen_seq_len, V + num_quantiles)
        generated_entropy = outputs["generated_entropy"] # shape (bs, gen_seq_len)
        lm_loss = outputs["lm_loss"] # shape (bs, gen_seq_len)

        generated_logits = generated_logits[:, :, :-self.num_quantiles] # shape (bs, gen_seq_len, V)

        masks = generations_attention_mask.to(self.policy.device)

        with torch.no_grad():
            prompts_input_ids_raw, prompts_attention_mask_raw = self.remove_quantile_from_prompt_input_ids(
                input_ids=prompts_input_ids, 
                attention_mask=prompts_attention_mask
            )
            ref_outputs = self.ref_policy.forward_pass(
                input_ids=prompts_input_ids_raw,
                attention_mask=prompts_attention_mask_raw,
                generated_input_ids=generations_input_ids,
                generated_attention_mask=generations_attention_mask
            )

            ref_logits = ref_outputs['generated_logits'] # shape (bs, gen_seq_len, V)

        kl = torch.sum(self.kl_loss(F.log_softmax(generated_logits, dim=-1), F.softmax(ref_logits, dim=-1)), dim=-1) # shape (bs, gen_seq_len)
        loss = reduce_mean(lm_loss + self.params['train']['kl_coef']*kl - self.params['train']['entropy_coef']*generated_entropy, masks) # shape (1)

        # gather tensors accross threads for wandb logging metrics
        with torch.no_grad():
            self.accelerator.wait_for_everyone()
            
            # lm loss
            lm_loss = reduce_mean(lm_loss, masks) 
            lm_loss = torch.mean(self.accelerator.gather(lm_loss.unsqueeze(dim=0)))

            # kl loss
            kl = reduce_mean(kl, masks)
            kl = torch.mean(self.accelerator.gather(kl.unsqueeze(dim=0)))

            # entropy loss
            generated_entropy = reduce_mean(generated_entropy, masks)
            generated_entropy = torch.mean(self.accelerator.gather(generated_entropy.unsqueeze(dim=0)))

            # total loss
            total_loss = lm_loss + self.params['train']['kl_coef']*kl - self.params['train']['entropy_coef']*generated_entropy
            self.accelerator.wait_for_everyone()

            stats = {
                'loss/total': total_loss.item(),
                'loss/kl': kl.item(),
                'loss/lm': lm_loss.item(),
                'loss/entropy': generated_entropy.item(),
            } 

        if self.accelerator.is_main_process:
            prompts, generations = self.decode(self.policy.tokenizer, prompts_input_ids_raw, generations_input_ids, skip_special_tokens=True)
            self.print_samples(queries=prompts, responses=generations, step_num=step_num)

        return loss, stats

    def print_samples(self, queries, responses, step_num) -> None:
        if step_num % self.params['logging']['log_interval'] != 0:
            return

        self.accelerator.print(f"[step {step_num}] Printing samples examples ...")
        for i in range(min(3, len(queries))):
            self.accelerator.print(f"\nSample {i+1}")
            self.accelerator.print(queries[i] + responses[i])

    def save(self, step_num, save_dir: Optional[str] = None) -> None:
        if not save_dir:
            save_dir = self.params['model_dir']

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        unwrapped_model.save_pretrained(
            f"{save_dir}/model_ckp_{step_num}",
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.policy.model),
        )
        self.accelerator.print(f"[step {step_num}] | Model checkpoint saved!")

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_dir=f"{save_dir}/full_ckp_{step_num}")
        self.accelerator.print(f"[step {step_num}] | Accelerator state (Model, Optimizer, Scheduler, etc. checkpoint) saved!")

class QuarkTrainer_noKL:
    def __init__(self,
                 params: dict,
                 policy: Policy,
                 quantile_tokens: List[str],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 accelerator: Accelerator,
                 training_dataloader: DataLoader
                 ) -> None:
        
        self.params = params
        self.num_quantiles = params['train']['num_quantiles']
        self.policy = policy
        self.policy.model.train()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.training_dataloader = training_dataloader
            
        self.quantile_tokens = quantile_tokens
        self.best_quantile_token = self.quantile_tokens[0]
        self.best_quantile_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_quantile_token)
        
        self.training_sampler = iter(self.training_dataloader)

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
        mask = torch.arange(seq_length, device=first_att_idxs.device).unsqueeze(0) != first_att_idxs # shape (batch_size, seq_length)
        # e.g., [True,  True,  True, False, True, True, True, True]
        #       [True,  True, False,  True, True, True, True, True]
        #       [False, True,  True,  True, True, True, True, True]
        input_ids = input_ids[mask].reshape(batch_size, -1)
        attention_mask = attention_mask[mask].reshape(batch_size, -1)
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

    def step(self, step_num) -> None:
        step_started_at = time.time()        

        try:
            batch = next(self.training_sampler) # dictionary with keys "inputs", "outputs", "prompts", "input_seqs", "output_seqs"
            assert len(batch["inputs"]["input_ids"]) == self.params['train']['training_batch_size_per_card'], 'insufficent batch'

        except (StopIteration, AssertionError):
            self.training_sampler = iter(self.training_dataloader)  # reset iteration to the beginning of data
            batch = next(self.training_sampler)

        self.optimizer.zero_grad()

        inputs_dict = batch["inputs"]
        outputs_dict = batch["outputs"]
        
        loss, stats = self.loss(step_num, inputs_dict, outputs_dict)
        self.accelerator.backward(loss)

        if self.params['train']['clip_grad']:
            self.accelerator.clip_grad_norm_(self.policy.model.parameters(), self.params['train']['max_grad_norm'])

        self.optimizer.step()
        self.scheduler.step()

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params['train']['training_batch_size_per_card']) / step_time
        self.accelerator.print(f"[step {step_num}] | Training ... step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")     

       # --- LOGGING ---
        if self.params['logging']['wandb_log']:
            for metric in ['lm', 'entropy', 'total']:
                self.accelerator.log({f'Loss/{metric}': stats[f'loss/{metric}']}, step=step_num)
            self.accelerator.log({f'Params/lr': self.optimizer.param_groups[0]['lr']}, step=step_num)
 
    def loss(self, step_num, inputs_dict, outputs_dict) -> Tuple[torch.Tensor, Dict[str, float]]:

        prompts_input_ids = inputs_dict["input_ids"]
        prompts_attention_mask = inputs_dict["attention_mask"]
        generations_input_ids = outputs_dict["input_ids"]
        generations_attention_mask = outputs_dict["attention_mask"]

        outputs = self.policy.forward_pass(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            generated_input_ids=generations_input_ids,
            generated_attention_mask=generations_attention_mask
        )

        generated_entropy = outputs["generated_entropy"] # shape (bs, gen_seq_len)
        lm_loss = outputs["lm_loss"] # shape (bs, gen_seq_len)

        loss = reduce_mean(lm_loss - self.params['train']['entropy_coef']*generated_entropy, generations_attention_mask) # shape (1)

        # gather tensors accross threads for wandb logging metrics
        with torch.no_grad():
            self.accelerator.wait_for_everyone()
            
            # lm loss
            lm_loss = reduce_mean(lm_loss, generations_attention_mask) 
            lm_loss = torch.mean(self.accelerator.gather(lm_loss.unsqueeze(dim=0)))

            # entropy loss
            generated_entropy = reduce_mean(generated_entropy, generations_attention_mask)
            generated_entropy = torch.mean(self.accelerator.gather(generated_entropy.unsqueeze(dim=0)))

            # total loss
            total_loss = lm_loss - self.params['train']['entropy_coef']*generated_entropy
            self.accelerator.wait_for_everyone()

            stats = {
                'loss/total': total_loss.item(),
                'loss/lm': lm_loss.item(),
                'loss/entropy': generated_entropy.item(),
            } 

        if self.accelerator.is_main_process:
            prompts, generations = self.decode(self.policy.tokenizer, prompts_input_ids, generations_input_ids, skip_special_tokens=True)
            self.print_samples(queries=prompts, responses=generations, step_num=step_num)

        return loss, stats

    def print_samples(self, queries, responses, step_num) -> None:
        if step_num % self.params['logging']['log_interval'] != 0:
            return

        self.accelerator.print(f"[step {step_num}] Printing samples examples ...")
        for i in range(min(3, len(queries))):
            self.accelerator.print(f"\nSample {i+1}")
            self.accelerator.print(queries[i] + responses[i])

    def save(self, step_num, save_dir: Optional[str] = None) -> None:
        if not save_dir:
            save_dir = self.params['model_dir']

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        unwrapped_model.save_pretrained(
            f"{save_dir}/model_ckp_{step_num}",
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.policy.model),
        )
        self.accelerator.print(f"[step {step_num}] | Model checkpoint saved!")

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_dir=f"{save_dir}/full_ckp_{step_num}")
        self.accelerator.print(f"[step {step_num}] | Accelerator state (Model, Optimizer, Scheduler, etc. checkpoint) saved!")
       