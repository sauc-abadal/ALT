import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")
print(sys.path)

import torch
import numpy as np
from transformers import AutoTokenizer

from tasks.summarization.models.policy import Policy

checkpoint_path = "/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/model/ckp_5000.pth"

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-j-6b",
    padding_side="left", 
    model_max_length=1024
) 

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

num_quantiles = 5
quantile_tokens =  [f"_QUANTILE_TOKEN_{str(quantile_idx)}_" for quantile_idx in range(num_quantiles)]

# add special reward quantile tokens to the tokenizer
tokenizer.add_tokens(quantile_tokens, special_tokens=True)

policy = Policy(
    model_checkpoint_name="CarperAI/openai_summarize_tldr_sft",
    device=device,
    tokenizer=tokenizer
)

weights = policy.model.get_input_embeddings().weight.detach().cpu().numpy()
mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in quantile_tokens])

policy.model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    new_inits = torch.tensor(new_inits)
    policy.model.get_input_embeddings().weight[-len(quantile_tokens):, :] = new_inits

######################
    
policy_state_dict = torch.load(checkpoint_path)["policy_model"]
for k, v in policy_state_dict.items():
    print(k)
    print(v.shape)
    
policy.model.load_state_dict(policy_state_dict)