from typing import Dict, List, Tuple, Optional, Union
import time

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator

from alt.utils.utils import reduce_mean
from alt.models.policy import Policy


class ALTTrainer_KL:
    def __init__(self,
                 params: dict,
                 policy: Policy,
                 ref_policy: Policy,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 accelerator: Accelerator,
                 training_dataloader: DataLoader
                 ) -> None:
        
        self.params = params
        self.policy = policy
        self.policy.model.train()
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.training_dataloader = training_dataloader
        
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        self.training_sampler = iter(self.training_dataloader)
        
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

        generated_logits = outputs["generated_logits"] # shape (bs, gen_seq_len, V)
        generated_entropy = outputs["generated_entropy"] # shape (bs, gen_seq_len)
        lm_loss = outputs["lm_loss"] # shape (bs, gen_seq_len)

        masks = generations_attention_mask.to(self.policy.device)

        with torch.no_grad():
            ref_outputs = self.ref_policy.forward_pass(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
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
            lm_loss = reduce_mean(lm_loss, generations_attention_mask) 
            lm_loss = torch.mean(self.accelerator.gather(lm_loss.unsqueeze(dim=0)))

            # kl loss
            kl = reduce_mean(kl, masks)
            kl = torch.mean(self.accelerator.gather(kl.unsqueeze(dim=0)))

            # entropy loss
            generated_entropy = reduce_mean(generated_entropy, generations_attention_mask)
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

    def save(self, step_num, iteration, save_dir: Optional[str] = None) -> None:
        if not save_dir:
            save_dir = self.params['model_dir']

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        unwrapped_model.save_pretrained(
            f"{save_dir}/model_ckp_{iteration}",
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.policy.model),
        )
        self.accelerator.print(f"[step {step_num}] | Model checkpoint saved!")
    
class ALTTrainer_noKL:
    def __init__(self,
                 params: dict,
                 policy: Policy,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 accelerator: Accelerator,
                 training_dataloader: DataLoader
                 ) -> None:
        
        self.params = params
        self.policy = policy
        self.policy.model.train()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.training_dataloader = training_dataloader
        
        self.training_sampler = iter(self.training_dataloader)
        
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

    def save(self, step_num, iteration, save_dir: Optional[str] = None) -> None:
        if not save_dir:
            save_dir = self.params['model_dir']

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        unwrapped_model.save_pretrained(
            f"{save_dir}/model_ckp_{iteration}",
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.policy.model),
        )
        self.accelerator.print(f"[step {step_num}] | Model checkpoint saved!")
     