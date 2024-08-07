import argparse
import yaml
import json
import gc

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from alt.utils.utils import set_seed, ensure_dir, remove_conditioning_from_str
from alt.models.reward import GPTRewardModel, MyRMDataCollator, MyRMDataset

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--input_sampling_file', required=True, type=str, help='path to input sampling file in JSONL format containing dicts with "prompt": str, "generations": List[str] as keys and values for every line.')
parser.add_argument('--output_dir', required=True, type=str, help='otuput dir where to save sampling file with the computed rewards in JSONL format by adding the key "reward": List[float] to every line')
parser.add_argument('--split_number', required=True, type=int, help='thread number / split number of the data file, in range 0..total_splits-1')
parser.add_argument('--total_splits', required=True, type=int, help='total number of threads / splits of the data file')
parser.add_argument('--num_generations', required=True, type=int, help='number of generations per prompt')
parser.add_argument('--ALT', action='store_true', help='ALT case for removing feedback part of the prompt')
args = parser.parse_args()
input_sampling_file = args.input_sampling_file
output_dir = args.output_dir
split_number = args.split_number
total_splits = args.total_splits
num_generations = args.num_generations
flag = args.ALT

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['input_sampling_file'] = input_sampling_file
    args['output_dir'] = output_dir
    args['split_number'] = split_number
    args['total_splits'] = total_splits
    args['num_generations'] = num_generations
    args['ALT'] = flag
    print(f"ALT: {args['ALT']}")

class Rewarder:
    def __init__(self,
                 params: dict,
                 reward_model: GPTRewardModel,
                 reward_dataloader: DataLoader,
                 sampling_file: str,
                 num_generations: int
                 ) -> None:
        
        self.params = params
        self.reward_model = reward_model
        self.reward_dataloader = reward_dataloader
        self.sampling_file = sampling_file
        self.num_generations = num_generations

    def get_rewards(self) -> None:
        rewards = []
        with torch.no_grad():
            for step, rm_batch in tqdm(enumerate(self.reward_dataloader), total=len(self.reward_dataloader)):
                
                for x in rm_batch:
                    rm_batch[x] = rm_batch[x].cuda()
                rewards_batch = self.reward_model.get_reward(**rm_batch)
                rewards.extend(rewards_batch)

        with open(self.sampling_file, 'r') as input_file:
            lines = input_file.readlines()

        # Adding the scores to each dictionary
        for i, line in enumerate(lines):
            data = json.loads(line)
            start_idx = i * self.num_generations
            end_idx = i * self.num_generations + self.num_generations
            data['rewards'] = rewards[start_idx:end_idx]
            data['prompt'] = remove_conditioning_from_str(data['prompt'], flag=args["ALT"])

            lines[i] = json.dumps(data)

        # Write the modified dictionaries with rewards to the sampling JSONL file
        with open(self.sampling_file, 'w') as out_file:
            out_file.write('\n'.join(lines))
            out_file.write('\n')

def main():

    ################################################################
    # -------------------- Set up Environment -------------------- #
    ################################################################
    gc.collect()
    torch.cuda.empty_cache()
    # Set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic']
    )
    print(f"############### reward.py ###############")
    
    # Set GPU device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    sampling_file = args['input_sampling_file']
    print(f"Reading sampled data from sampling_file: {sampling_file}")
    save_dir = args['output_dir']
    print(f"Writing reward data to: {save_dir}")
    ensure_dir(save_dir)
    
    ################################################################
    # ------------------ Initialize Reward Model ----------------- #
    ################################################################
    print('Initializing base Reward Model ...')
    reward_model = GPTRewardModel(args['reward']['name_or_path'])
    print("Base Reward Model correctly loaded!")
    if args['reward']['load_state_dict']:
        print("Attempting to load Reward Model checkpoint ...")
        rm_state_dict = torch.load(args['reward']['state_dict_path'], map_location=torch.device('cpu'))
        reward_model.load_state_dict(rm_state_dict)
        print("Reward Model checkpoint correctly loaded!")

    reward_model.tokenizer.max_length = args['reward']['max_length']
    
    if args['reward']['half']:
        reward_model.half()
    reward_model.to(device)
    reward_model.eval()

    ################################################################
    # -------------- Rewarding Dataset / Dataloader -------------- #
    ################################################################

    num_generations = args['num_generations']
    
    samples = []
    with open(sampling_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            generations = entry["generations"]
            assert len(generations) == num_generations
            samples.append(entry)
    
    # Split the data into chunks.
    chunk_size = len(samples) // args["total_splits"] + 1
    start = (args["split_number"]) * chunk_size
    end = min((args["split_number"] + 1) * chunk_size, len(samples))
    samples = samples[start:end]
    print(f"Thread {args['split_number']} processing {len(samples)*num_generations} samples.")
    
    # Save chunk of sampling data into json for writing the reward scores afterward
    new_sampling_file = f"{save_dir}/{sampling_file.split('.')[0].split('/')[-1]}_reward_thread_{args['split_number']}.json"
    with open(new_sampling_file, 'w') as output_file:
        for x in samples:
            output_file.write(json.dumps(x) + '\n')

    # flatten samples into a single list
    all_samples = []
    for sample in samples:
        prompt = sample["prompt"].strip()
        prompt = remove_conditioning_from_str(prompt, flag=args["ALT"])
        generations = sample["generations"]
        y = [prompt + ' ' + gen.strip() for gen in generations]
        all_samples.extend(y)

    assert len(all_samples) % num_generations == 0    

    rm_dataset = MyRMDataset(samples=all_samples)
    rm_collator = MyRMDataCollator(tokenizer=reward_model.tokenizer, max_length=reward_model.tokenizer.max_length)
    rm_dataloader = DataLoader(
        rm_dataset, 
        shuffle=False, 
        drop_last=False,
        batch_size=args['reward']['batch_size'], 
        collate_fn=rm_collator)
    
    ################################################################
    # ---------------------- Set up Rewarder --------------------- #
    ################################################################
        
    rewarder = Rewarder(
        params=args,
        reward_model=reward_model,
        reward_dataloader=rm_dataloader,
        sampling_file=new_sampling_file,
        num_generations=num_generations
    )

    print("\n--------------------- STARTING REWARDING! ---------------------\n")
    rewarder.get_rewards()
    print("\n--------------------- REWARDING COMPLETED ---------------------\n")
    
if __name__ == "__main__":
    main()
