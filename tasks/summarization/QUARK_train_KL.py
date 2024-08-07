import os
import argparse
import yaml
import json
import gc

from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
import torch
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DummyOptim, DummyScheduler

from alt.utils import set_seed, ensure_dir, ceil_div
from alt.models.policy import Policy
from alt.training_dataset_and_collator import Quark_TrainingDataset, TrainingSequenceCollatorWithPadding
from alt.data_pool import ALT_RM_DataPool
from alt.utils.state import load_state, save_state
from alt.trainer.quark_trainer import QuarkTrainer_KL

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--iteration', type=int, help='number of sampling/reward phases carried out')
parser.add_argument('--input_sampling_file', required=True, type=str, help='path to input sampling file in JSONL format containing the newly sampled data to be added to the datapool')
parser.add_argument('--ds_optimizer', action='store_true', help='whether we are using a DeepSpeed optimizer or not, if provided -> set to True')
parser.add_argument('--ds_scheduler', action='store_true', help='whether we are using a DeepSpeed scheduler or not, if provided -> set to True')
args = parser.parse_args()
iteration = args.iteration
input_sampling_file = args.input_sampling_file
ds_optimizer = args.ds_optimizer
ds_scheduler = args.ds_scheduler

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['iteration'] = iteration
    args['input_sampling_file'] = input_sampling_file
    args['ds_optimizer'] = ds_optimizer
    args['ds_scheduler'] = ds_scheduler


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
    accelerator.print("############### QUARK_train.py ###############")
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
        if iteration == 1:
            state_dir = f"{args['logging']['save_dir']}/state"
            ensure_dir(state_dir)
            accelerator.print("Creating a new state.json file.")
            with open(f"{state_dir}/state_iter_{iteration}.json", "w") as f:
                json.dump({"step_num": 0}, f)

    accelerator.wait_for_everyone() # wait for all threads to ensure save_dir exists and state is created
    
    state_file_path = f"{args['logging']['save_dir']}/state/state_iter_{iteration}.json"
    state_dict = load_state(state_file_path)
    step_num = state_dict["step_num"]
    accelerator.print(f"state_dict loaded: {state_dict}")

    # Set saving directories
    args['save_dir'] = args['logging']['save_dir']
    args['sampling_dir'] = os.path.join(args['save_dir'], 'sampling')
    args['model_dir'] = os.path.join(args['save_dir'], 'model')
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
    tokenizer_initial_len = len(tokenizer)
    accelerator.print(f"Tokenizer has {tokenizer_initial_len} vocabulary tokens after loading from pre-trained.")
    
    # add special reward quantile tokens to the tokenizer
    tokenizer.add_tokens(quantile_tokens, special_tokens=True)
    accelerator.print(f"Reward Quantile Tokens added to the tokenizer: {quantile_tokens}")
    accelerator.print(f"Tokenizer vocabulary tokens extended to {len(tokenizer)}.")

    ################################################################
    # --------------- Initialize Reference Policy ---------------- #
    ################################################################

    ref_policy = Policy(
        model_checkpoint_name=args['model']['ref_policy']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    accelerator.print(f"{ref_policy.model.__class__.__name__} Pre-trained reference Policy model correctly loaded to {device}.")
    accelerator.print(f"Pre-trained Policy model has dtype: {ref_policy.model.dtype}")
    if ref_policy.model.__class__.__name__ == 'GPTNeoXForCausalLM': # Pythia
        accelerator.print(f"Input embeddings matrix shape: {ref_policy.model.gpt_neox.embed_in.weight.shape}")
        ref_policy.model.resize_token_embeddings(tokenizer_initial_len)
        accelerator.print(f"Input embeddings matrix reshaped to: {ref_policy.model.gpt_neox.embed_in.weight.shape}")
    else: # GPT-J
        accelerator.print(f"Input embeddings matrix shape: {ref_policy.model.transformer.wte.weight.shape}")
        ref_policy.model.resize_token_embeddings(tokenizer_initial_len)
        accelerator.print(f"Input embeddings matrix reshaped to: {ref_policy.model.transformer.wte.weight.shape}")
    
    ################################################################
    # ------------ Initialize Policy to be finetuned ------------- #
    ################################################################
        
    policy = Policy(
        model_checkpoint_name=args['model']['policy_model']['name_or_path'],
        device=device,
        tokenizer=tokenizer
    )
    accelerator.print(f"{policy.model.__class__.__name__} Pre-trained Policy model correctly loaded to {device}.")
    accelerator.print(f"Pre-trained Policy model has dtype: {policy.model.dtype}")
    if policy.model.__class__.__name__ == 'GPTNeoXForCausalLM': # Pythia
        accelerator.print(f"Input embeddings matrix shape: {policy.model.gpt_neox.embed_in.weight.shape}")
        policy.model.resize_token_embeddings(tokenizer_initial_len)
        accelerator.print(f"Input embeddings matrix reshaped to: {policy.model.gpt_neox.embed_in.weight.shape}")
    else: # GPT-J
        accelerator.print(f"Input embeddings matrix shape: {policy.model.transformer.wte.weight.shape}")
        policy.model.resize_token_embeddings(tokenizer_initial_len)
        accelerator.print(f"Input embeddings matrix reshaped to: {policy.model.transformer.wte.weight.shape}")
    
    # resize token_embeddings associated to the newly added tokens
    weights = policy.model.get_input_embeddings().weight.detach().cpu().numpy()
    mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
    new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in quantile_tokens])

    policy.model.resize_token_embeddings(len(tokenizer))
    if policy.model.__class__.__name__ == 'GPTNeoXForCausalLM': # Pythia
        accelerator.print(f"After adding quantile tokens, Input embeddings matrix reshaped to: {policy.model.gpt_neox.embed_in.weight.shape}")
    else: # GPT-J
        accelerator.print(f"After adding quantile tokens, Input embeddings matrix reshaped to: {policy.model.transformer.wte.weight.shape}")
    
    with torch.no_grad():
        new_inits = torch.tensor(new_inits)
        policy.model.get_input_embeddings().weight[-len(quantile_tokens):, :] = new_inits

    ################################################################
    # ------------------ Initialize DataPool --------------------- #
    ################################################################
        
    data_pool = ALT_RM_DataPool(
        reward_quantile_tokens=quantile_tokens, num_quantiles=num_quantiles
    )
    
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
        datapool_save_dict = data_pool.serialize_to_dict(save_path=args['save_dir'])
        accelerator.print("Updated DataPool correctly serialized!")
        state_dict["data_pool"] = datapool_save_dict

        # Save updated datapool statistics, i.e., reward/genenerations length histograms for each quantile
        data_pool.get_data_statistics(save_path=args['sampling_dir'], tokenizer=tokenizer)

    accelerator.wait_for_everyone()

    ################################################################
    # --------------------- Dataset / Dataloader ----------------- #
    ################################################################

    accelerator.print("Loading the training dataset and dataloader from the DataPool.")
    training_dataset = Quark_TrainingDataset(
        data_pool=data_pool, 
        num_samples_per_quantile=args['train']['num_samples_per_quantile'],
        tokenizer=tokenizer,
        max_new_tokens=args['train']['max_new_tokens']
    ).dataset['train']
    training_dataset = training_dataset.shuffle(seed=iteration)
    training_seq_collator = TrainingSequenceCollatorWithPadding(tokenizer=tokenizer)
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=args['train']['training_batch_size_per_card'],
        shuffle=True,
        drop_last=True,
        collate_fn=training_seq_collator
    )
    accelerator.print("Dataset and Dataloader correctly initialized!")

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
    num_samples = len(training_dataset)
    total_episodes = num_samples * args['train']['num_epochs']
    total_steps = ceil_div(total_episodes, global_batch_size) # total episodes per iteration = 2048*num_quantiles*num_samples_per_quantile*num_epochs = 2048*5*2*2 = 40960
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
    for name, param in policy.model.named_parameters():
        param_types_set.add(param.dtype)
    accelerator.print(f"Model after accelerator.prepare() have the following dtypes: {param_types_set}")
    accelerator.print(f"Model after accelerator.prepare() wrapped into {policy.model.__class__.__name__}")

    if iteration > 1:
        # Restoring Accelerator state (Model, Optimizer, Scheduler, etc.)
        last_ckp = state_dict["last_ckp"]
        last_ckp_path = f"{args['model_dir']}/full_ckp_{last_ckp}"
        accelerator.print(f"\nLoading Accelerator state (Model, Optimizer, Scheduler, etc.) from {last_ckp_path}.")
        accelerator.load_state(last_ckp_path)
        accelerator.print("Accelerator state correclty loaded!")

    ################################################################
    # ---------------------- Set up trainer ---------------------- #
    ################################################################
        
    trainer = QuarkTrainer_KL(
        params=args,
        policy=policy,
        ref_policy=ref_policy,
        quantile_tokens=quantile_tokens,
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

    

    

    