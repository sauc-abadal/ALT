import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")

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
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DummyOptim, DummyScheduler

from utils import set_seed, ensure_dir, ceil_div, reduce_mean, WANDB_API_KEY, NEGATIVE_INF
from tasks.summarization.models.policy import Policy
from training_dataset_and_collator import QuarkToNLFTrainingDataset, QuarkToNLFTrainingSequenceCollatorWithPadding
from data_pool import QuarkDataPool
from state import load_state, save_state

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--iteration', type=int, help='number of sampling/reward phases carried out')
parser.add_argument('--input_sampling_file', required=True, type=str, help='path to input sampling file in JSONL format containing the newly sampled data to be added to the datapool')
parser.add_argument('--model_path', required=False, type=str, help='path to model ckpt to resume fine-tuning')
parser.add_argument('--ds_optimizer', action='store_true', help='whether we are using a DeepSpeed optimizer or not, if provided -> set to True')
parser.add_argument('--ds_scheduler', action='store_true', help='whether we are using a DeepSpeed scheduler or not, if provided -> set to True')
args = parser.parse_args()
iteration = args.iteration
input_sampling_file = args.input_sampling_file
if args.model_path:
    model_path = args.model_path
else:
    model_path = None
ds_optimizer = args.ds_optimizer
ds_scheduler = args.ds_scheduler

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['iteration'] = iteration
    args['input_sampling_file'] = input_sampling_file
    args['model_path'] = model_path
    args['ds_optimizer'] = ds_optimizer
    args['ds_scheduler'] = ds_scheduler

class NLFTrainer:
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
       
def main():

    ###############################################################
    # -------------------- Set up Environment -------------------- #
    ################################################################
    gc.collect()
    torch.cuda.empty_cache()
    # Set seed
    set_seed(
        seed=args['train']['seed'] + args["iteration"], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)
    accelerator.print("############### quarkToNLF_train_noKL.py ###############")
    accelerator.print(f"{AcceleratorState()}")
    device = accelerator.device
    num_gpus = accelerator.num_processes
    accelerator.print(f'Detected {num_gpus} GPUS')

    num_quantiles = args['train']['num_quantiles']
    quantile_tokens =  [f"_QUANTILE_TOKEN_{str(quantile_idx)}_" for quantile_idx in range(num_quantiles)]
    
    # Set up wandb logging
    if args['logging']['wandb_log']:
        wandb_config = {
            "entity": args['logging']['wandb_entity'],
            "name": args['logging']['run_name'],
            "id": args['logging']['run_id']
        }
        accelerator.init_trackers(
            project_name=args['logging']['wandb_project'],
            init_kwargs={"wandb": wandb_config}
        )

    # Load the state from the state_dict
    iteration = args["iteration"]
    if accelerator.is_main_process:
        ensure_dir(args['logging']['save_dir'])
        state_dir = f"{args['logging']['save_dir']}/state"
        ensure_dir(state_dir)
        if iteration == 1:
            accelerator.print("Creating a new state.json file.")
            with open(f"{state_dir}/state_iter_{iteration}.json", "w") as f:
                json.dump({"overall_steps": 0}, f)
        else:
            accelerator.print("Copying previous state.json file.")
            with open(f"{state_dir}/state_iter_{iteration-1}.json", "r") as f:
                previous_dict = json.load(f)
            with open(f"{state_dir}/state_iter_{iteration}.json", "w") as f:
                json.dump(previous_dict, f)

    accelerator.wait_for_everyone() # wait for all threads to ensure save_dir exists and state is created
    
    state_file_path = f"{args['logging']['save_dir']}/state/state_iter_{iteration}.json"
    state_dict = load_state(state_file_path)
    step_num = state_dict["overall_steps"]
    accelerator.print(f"state_dict loaded: {state_dict}")

    # Set saving directories
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], f'output_iter_{iteration}')
    args['model_dir'] = os.path.join(args['save_dir'], f'model/iter_{iteration}')
    args['model_scratch_dir'] = os.path.join(args['logging']['scratch_dir'], 'model')
    if accelerator.is_main_process:
        ensure_dir(args['sampling_dir'])
        ensure_dir(args['model_dir'])
        ensure_dir(args['model_scratch_dir'])
    accelerator.wait_for_everyone()
    accelerator.print(f"Loading/Saving policy model from directories: {args['model_dir']}, {args['model_scratch_dir']}")
    accelerator.print(f"Reading sampling data from: {args['input_sampling_file']}")

    if accelerator.is_main_process:
        # Save the config file
        with open(os.path.join(args['save_dir'], f'training_args_iter_{iteration}.json'), 'w') as f:
            json.dump(args, f, indent=2)

    accelerator.print(f'--------------------- Initializing models ... ---------------------')

    ################################################################
    # ------------------- Initialize Tokenizer ------------------- #
    ################################################################

    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['tokenizer']['name_or_path'],
        padding_side=args['model']['policy_model']['input_padding_side'], # left padding
        model_max_length=args['train']['max_input_length']) 
    
    if tokenizer.__class__.__name__ == 'GPTNeoXTokenizerFast': # Pythia
        tokenizer.pad_token = "<|padding|>" # model has special padding token used during pre-training
    
    else: # GPT-J
        tokenizer.pad_token = tokenizer.eos_token 

    accelerator.print(f"{tokenizer.__class__.__name__} correctly loaded!")
    accelerator.print(f"Tokenizer pad_token: {tokenizer.pad_token} | pad_token_id: {tokenizer.pad_token_id}")
    accelerator.print(f"Tokenizer padding side set to: {tokenizer.padding_side}")
    accelerator.print(f"Tokenizer model_max_length set to: {tokenizer.model_max_length}")
    accelerator.print(f"Tokenizer has {len(tokenizer)} vocabulary tokens after loading from pre-trained.")
    
    ################################################################
    # ------------ Initialize Policy to be finetuned ------------- #
    ################################################################

    if iteration == 1:
        policy = Policy(
            model_checkpoint_name=args['model']['policy_model']['name_or_path'],
            device=device,
            tokenizer=tokenizer
        )
        accelerator.print(f"{policy.model.__class__.__name__} Pre-trained SFT Policy model correctly loaded to {device}.")
        accelerator.print(f"Pre-trained Policy model has dtype: {policy.model.dtype}")
        if policy.model.__class__.__name__ == 'GPTNeoXForCausalLM': # Pythia
            accelerator.print(f"Input embeddings matrix shape: {policy.model.gpt_neox.embed_in.weight.shape}")
        else: # GPT-J
            accelerator.print(f"Input embeddings matrix shape: {policy.model.transformer.wte.weight.shape}")
    
    else:
        policy = Policy(
            model_checkpoint_name=args['model_path'],
            device=device,
            tokenizer=tokenizer
        )
        accelerator.print(f"{policy.model.__class__.__name__} Policy model correctly loaded to {device}.")
        accelerator.print(f"Policy model has dtype: {policy.model.dtype}")
        if policy.model.__class__.__name__ == 'GPTNeoXForCausalLM': # Pythia
            accelerator.print(f"Input embeddings matrix shape: {policy.model.gpt_neox.embed_in.weight.shape}")
        else: # GPT-J
            accelerator.print(f"Input embeddings matrix shape: {policy.model.transformer.wte.weight.shape}")

    ################################################################
    # ------------------ Initialize DataPool --------------------- #
    ################################################################
        
    data_pool = QuarkDataPool(num_quantiles=num_quantiles, reward_quantile_tokens=quantile_tokens)
    
    if iteration > 1:
        # Load existing DataPool
        datapool_load_dict = state_dict["data_pool"]
        data_pool.load_from_dict(datapool_load_dict)
        accelerator.print("Existing data_pool correctly loaded.")
    
    accelerator.print(f"Current DataPool has {data_pool.get_num_samples()}.")

    # Update DataPool with the newly sampled data in the current iteration
    sampling_file = args['input_sampling_file']
    accelerator.print(f"Updating DataPool with sampling_file from: {sampling_file}, drop_factor: {args['train']['datapool_drop_factor']}")
    data_pool.update_datapool(
        sampling_file=sampling_file, 
        drop_factor=args['train']['datapool_drop_factor']
    )
    accelerator.print("DataPool correctly updated!")
    accelerator.print(f"Updated DataPool has {data_pool.get_num_samples()}.")

    if accelerator.is_main_process:
        # Save new DataPool state to state_dict (state_dict to be saved once training completes)
        datapool_save_dict = data_pool.serialize_to_dict(save_path=args['sampling_dir'])
        accelerator.print("Updated DataPool correctly serialized!")
        state_dict["data_pool"] = datapool_save_dict

    accelerator.wait_for_everyone()

    ################################################################
    # ------------ Prepare Optimizer and Schedulers -------------- #
    ################################################################

    if 'unfrozen_layers_ratio' in args['train']:
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

    accelerator.print(f"Finetuning {num_trainable_params/1e9:.2f}/{(num_trainable_params + num_non_trainable_params)/1e9:.2f}B parameters.")

    # Initialize new Optimizer and Scheduler
    global_batch_size = args['train']['training_batch_size_per_card'] * num_gpus
    total_steps = ceil_div(args['train']['total_episodes'], global_batch_size) # total episodes per iteration = 2048*num_samples_per_prompt*num_epochs = 2048*10*2 = 40960
    warmup_steps = total_steps * args['train']['warmup_ratio']
    if not args['ds_optimizer']:
        accelerator.print("Using a PyTorch optimizer!")
        optimizer = torch.optim.Adam(
            params=policy.model.parameters(),
            lr=float(args['train']['lr']),
            betas=(0.8, 0.999),
            eps=1e-8,
            weight_decay=3e-7
        )
    else:
        # If we are using the DS optimizer, we must also use the DS scheduler 
        # using a non-DS scheduler when using the DS optimizer is not compatible
        accelerator.print("Using a DeepSpeed optimizer!")
        optimizer = DummyOptim(
            params=policy.model.parameters(),
            lr=float(args['train']['lr']),
            betas=(0.8, 0.999),
            eps=1e-8,
            weight_decay=3e-7
        )
        accelerator.print("Using a DeepSpeed scheduler!")
        scheduler = DummyScheduler(
            optimizer=optimizer,
            warmup_num_steps=warmup_steps,
            total_num_steps=total_steps*accelerator.num_processes # required to fix bug and obtain desired behavior
        )

    if not args['ds_scheduler']:
        accelerator.print("Using a PyTorch scheduler!")
        scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    ################################################################
    # --------------------- Dataset / Dataloader ----------------- #
    ################################################################

    accelerator.print("Loading the training dataset and dataloader from the DataPool.")
    training_dataset = QuarkToNLFTrainingDataset(
        datapool=data_pool, 
        num_samples_per_prompt=args['train']['num_samples_per_prompt'],
        eos_token=tokenizer.eos_token,
    ).dataset['train']
    training_seq_collator = QuarkToNLFTrainingSequenceCollatorWithPadding(tokenizer=policy.tokenizer)
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=args['train']['training_batch_size_per_card'],
        shuffle=True,
        drop_last=True,
        collate_fn=training_seq_collator
    )
    accelerator.print("Dataset and Dataloader correctly initialized!")

    ################################################################
    # ---------------------- Set up Accelerator ------------------ #
    ################################################################

    accelerator.print("\nCalling accelerator.prepare()...\n")
    policy.model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        policy.model, optimizer, training_dataloader, scheduler
    )
    accelerator.print("Model, optimizer, dataloader, scheduler correctly prepared!")
    accelerator.print(f"After .prepare(): Training dataloader has {len(training_dataloader)} batches.")
    accelerator.print(f"Policy model dtype set to {policy.model.dtype} after accelerator.prepare().")
    param_types_set = set()
    for _, param in policy.model.named_parameters():
        param_types_set.add(param.dtype)
    accelerator.print(f"Model after accelerator.prepare() have the following dtypes: {param_types_set}")
    accelerator.print(f"Model after accelerator.prepare() wrapped into {policy.model.__class__.__name__}")

    ################################################################
    # ---------------------- Set up trainer ---------------------- #
    ################################################################
        
    trainer = NLFTrainer(
        params=args,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        training_dataloader=training_dataloader,
    )

    steps_taken = 0
    steps_bar = tqdm(total=total_steps, initial=steps_taken, position=0, disable=not accelerator.is_main_process)

    accelerator.print("\n--------------------- STARTING TRAINING! ---------------------\n")
    while steps_taken < total_steps:
        try:
            accelerator.wait_for_everyone()
            trainer.step(step_num)

            steps_taken += 1
            step_num += 1
            if accelerator.is_main_process:
                steps_bar.update(1)

            if steps_taken % args['logging']['save_interval'] == 0:
                trainer.save(step_num, save_dir=args["model_scratch_dir"])

        except Exception as e:
            accelerator.print("\nThere was an Exception while trying to perform trainer.step()!\n")
            accelerator.print(e)
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                steps_bar.update(0)
            continue

    steps_bar.close()
    accelerator.end_training()
    accelerator.wait_for_everyone()
    trainer.save(step_num)
    state_dict["overall_steps"] = step_num
    if accelerator.is_main_process:
        save_state(state_dict, state_file_path)
    accelerator.print(f"state_dict saved: {state_dict}")

if __name__ == "__main__":
    main()

    

    