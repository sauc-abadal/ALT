import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")
print(sys.path)

import os
import argparse
import yaml
import json
from typing import Dict, List, Tuple, Optional, Union
import time
import gc

from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from utils import set_seed, ensure_dir, ceil_div, reduce_mean, WANDB_API_KEY
from tasks.summarization.models.policy import Policy
from training_dataset_and_collator import QuarkTrainingDataset, QuarkTrainingSequenceCollatorWithPadding
from data_pool import QuarkDataPool
from state import load_state, save_state

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

class QuarkTrainer:
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

        print(f"Reference Policy device set to {self.ref_policy.device}")
        print(f"Finetuning Policy device set to {self.policy.device}")
        print(f"Accelerator device set to {self.accelerator.device}")
              
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
        print(f"[step {step_num}] | Training ... step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")     

        # --- LOGGING ---
        # if self.params['logging']['wandb_log']:
        #     for metric in ['lm', 'kl', 'entropy', 'total']:
        #        wandb.log({f'Loss/{metric}': stats[f'loss/{metric}']}, step=step_num)
        #     wandb.log({f'Params/lr': self.optimizer.param_groups[0]['lr']}, step=step_num)
 
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

        generated_logits = outputs["generated_logits"]
        generated_logprobs = outputs["generated_logprobs"]
        generated_entropy = outputs["generated_entropy"]
        lm_loss = outputs["lm_loss"]

        generated_logits = generated_logits[:, :, :-self.num_quantiles]

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

            ref_logits = ref_outputs['generated_logits']
            ref_logprobs = ref_outputs['generated_logprobs']

        kl = torch.sum(self.kl_loss(F.log_softmax(generated_logits, dim=-1), F.softmax(ref_logits, dim=-1)), dim=-1)
        loss = reduce_mean(lm_loss + self.params['train']['kl_coef']*kl - self.params['train']['entropy_coef']*generated_entropy, masks)

        data = {'logprobs': generated_logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': generated_logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(generated_entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        prompts, generations = self.decode(self.policy.tokenizer, prompts_input_ids_raw, generations_input_ids, skip_special_tokens=True)
        self.print_samples(queries=prompts, responses=generations, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=generated_logprobs, ref_logprobs=ref_logprobs, masks=masks, step_num=step_num)

        return loss, stats

    def record_step_stats(self, data) -> Dict[str, float]:
        stats = {
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        }
        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step_num) -> None:
        if step_num % self.params['logging']['log_interval'] != 0:
            return

        print(f"[step {step_num}] Printing samples examples ...")
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(f"\nSample {i+1}")
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params['train']['kl_coef'] * sample_kl:+.2f}")

    def save(self, step_num) -> None:
        model_state = self.accelerator.get_state_dict(self.policy.model) # This will call the unwrap model as well
        self.accelerator.save(model_state, f"{self.params['model_dir']}/model_ckp_{step_num}.pth") # Use in place of `torch.save`
        print(f"[step {step_num}] | Model checkpoint saved!")

        self.accelerator.save_state(f"{self.params['model_dir']}/full_ckp_{step_num}.pth")
        print(f"[step {step_num}] | Model, Optimizer, Scheduler, etc. checkpoint saved!")
       

def main():
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

    accelerator.print("############### quark_train.py ###############")
    gc.collect()
    torch.cuda.empty_cache()

    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    # Set GPUs / Accelerator
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')
    device = accelerator.device
    
    # Set wandb logging
    # wandb_log = args['logging']['wandb_log']
    # if wandb_log:
    #     wandb.login(key=WANDB_API_KEY)
    #     wandb.init(
    #         entity=args['logging']['wandb_entity'],
    #         project=args['logging']['wandb_project'],
    #         name=f"{args['logging']['run_name']}",
    #         id=f"{args['logging']['run_id']}"
    #     )

    # Load the state
    state_file_path = args['train']['state_file_path'] 
    state_dict = load_state(state_file_path)
    if "step_num" not in state_dict:
        state_dict["step_num"] = 0
    sampling_stage = state_dict["sampling_stage"] - 1 # training is occurring in the current sampling stage, despite the variable being already incremented after sampling
    step_num = state_dict["step_num"]
    print(f"state_dict loaded: {state_dict}")

    # Set saving directories
    args['save_dir'] = f"{args['logging']['save_dir']}/saving_test"
    args['model_dir'] = os.path.join(args['save_dir'], 'model')
    ensure_dir(args['model_dir'])
    print(f"Loading/Saving policy model from directory: {args['model_dir']}")

    # Save the config file
    # with open(os.path.join(args['save_dir'], 'args.json'), 'w') as f:
    #     json.dump(args, f, indent=2)

    print(f'Initializing models ...')

    # -------------- Initialize Tokenizer --------------
    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['tokenizer']['name_or_path'],
        padding_side=args['model']['policy_model']['input_padding_side'], # left padding
        max_length=args['train']['max_input_length']) # GPT2Tokenizer -> vocab_size 50257 (id from 0 to 50256) + extra_tokens for efficiency (id from 50257 to 50399) -> 50400 total vocabulary 
    
    if not tokenizer.pad_token:
        print("Setting PAD token to EOS token for open-ended generation.")
        tokenizer.pad_token = tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    num_quantiles = args['train']['num_quantiles']
    quantile_tokens =  [f"_QUANTILE_TOKEN_{str(quantile_idx)}_" for quantile_idx in range(num_quantiles)]

    # add special reward quantile tokens to the tokenizer
    tokenizer.add_tokens(quantile_tokens, special_tokens=True)

    # -------------- Initialize Reference Policy --------------
    ref_policy = Policy(
        model_checkpoint_name=args['model']['ref_policy']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    print(f"Reference policy loaded to {device}.")

    # -------------- Initialize Policy to be finetuned --------------
    policy = Policy(
        model_checkpoint_name=args['model']['policy_model']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    print(f"Pre-trained Policy correctly loaded to {device}.")
    # resize token_embeddings associated to the newly added tokens
    weights = policy.model.get_input_embeddings().weight.detach().cpu().numpy()
    mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
    new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in quantile_tokens])

    policy.model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        new_inits = torch.tensor(new_inits)
        policy.model.get_input_embeddings().weight[-len(quantile_tokens):, :] = new_inits

    # -------------- Initialize DataPool --------------
    # Load existing DataPool
    datapool_load_dict = state_dict["data_pool"]
    data_pool = QuarkDataPool(
        reward_quantile_tokens=quantile_tokens, num_quantiles=num_quantiles
    )
    data_pool.load_from_dict(datapool_load_dict)
    print("Existing data_pool correctly loaded.")

    # -------------- Prepare Optimizer and Schedulers --------------

    # Freeze 70% of policy model backbone
    unfrozen_layers_ratio = args['train']['unfrozen_layers_ratio']
    layers = policy.model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(unfrozen_layers_ratio * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    num_trainable_params = 0
    num_non_trainable_params = 0
    for param in policy.model.parameters():
        num_params = torch.numel(param)
        if param.requires_grad:
            num_trainable_params += num_params
        else:
            num_non_trainable_params += num_params
    print(f"Finetuning {num_trainable_params/1e9:.2f}/{(num_trainable_params + num_non_trainable_params)/1e9:.2f}B parameters.")

    total_steps = ceil_div(args['train']['total_episodes'], args['train']['training_batch_size_per_card']*num_gpus)
    
    # Initialize new Optimizer and Scheduler
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=float(args['train']['lr']), eps = 1e-5)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=args['train']['n_warmup_steps'],
        num_training_steps=total_steps
    )

    # -------------- Set up Accelerator ----------
    training_dataset = QuarkTrainingDataset(data_pool=data_pool, tokenizer=policy.tokenizer)
    training_seq_collator = QuarkTrainingSequenceCollatorWithPadding(tokenizer=policy.tokenizer)
    training_dataloader = DataLoader(
        dataset=training_dataset.dataset["train"],
        batch_size=args['train']['training_batch_size_per_card'],
        shuffle=True,
        drop_last=True,
        collate_fn=training_seq_collator
    )

    policy.model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        policy.model, optimizer, training_dataloader, scheduler
    )

    # -------------- Restoring Accelerator state (Model, Optimizer, Scheduler, etc.) --------------
    if sampling_stage > 1:
        last_ckp = state_dict["last_ckp"]
        last_ckp_path = f"{args['model_dir']}/full_ckp_{last_ckp}.pth"
        print(f"Loading Accelerator state (Model, Optimizer, Scheduler, etc.) from {last_ckp_path}.")
        accelerator.load_state(last_ckp_path)

    # -------------- Set up trainer --------------
    trainer = QuarkTrainer(
        params=args,
        policy=policy,
        ref_policy=ref_policy,
        quantile_tokens=quantile_tokens,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        training_dataloader=training_dataloader,
    )

    sample_interval = 5
    steps_taken = 0
    steps = list(range(step_num+1, total_steps+1)) # starts at [1, total_steps], then [1+steps_taken, total_steps], etc.
    steps_bar = tqdm(total=len(steps), initial=step_num, position=0)

    while steps_taken < sample_interval:
        try:
            trainer.step(step_num+1)
            steps_taken += 1
            step_num += 1
            state_dict["step_num"] += 1
            steps_bar.update(1)
        except Exception as e:
            print("There was an Exception while trying to perform trainer.step()!")
            print(e)
            torch.cuda.empty_cache()
            steps_bar.update(0)
            continue

    steps_bar.close()

    trainer.save(state_dict["step_num"])
    state_dict["last_ckp"] = state_dict["step_num"]
    save_state(state_dict, state_file_path)
    print(f"state_dict saved: {state_dict}")

if __name__ == "__main__":
    main()

    
