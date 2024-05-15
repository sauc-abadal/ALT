from transformers import AutoTokenizer
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(
        args['model']['tokenizer']['name_or_path'],
        padding_side=args['model']['policy_model']['input_padding_side'], # left padding
        model_max_length=args['train']['max_input_length']) 
    
if tokenizer.__class__.__name__ == 'GPTNeoXTokenizerFast': # Pythia
    tokenizer.pad_token = "<|padding|>" # model has special padding token used during pre-training

else: # GPT-J
    tokenizer.pad_token = tokenizer.eos_token 

print(f"{tokenizer.__class__.__name__} correctly loaded!")
print(f"Tokenizer eos_token: {tokenizer.eos_token} | eos_token_id: {tokenizer.eos_token_id}")
print(f"Tokenizer pad_token: {tokenizer.pad_token} | pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer padding side set to: {tokenizer.padding_side}")
print(f"Tokenizer model_max_length set to: {tokenizer.model_max_length}")
print(f"Tokenizer has {len(tokenizer)} vocabulary tokens after loading from pre-trained.")


save_dir = args['logging']['save_dir']
tokenizer_save_path = f"{save_dir}/NLF_TLDR_tokenizer"
tokenizer.save_pretrained(tokenizer_save_path)