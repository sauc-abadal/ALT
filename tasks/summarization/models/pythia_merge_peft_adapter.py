import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel

peft_model_id = "mnoukhov/pythia410m-tldr-sft-rm-adapter"
peft_config = PeftConfig.from_pretrained(peft_model_id)
torch_dtype = torch.float32

if peft_config.task_type == "SEQ_CLS":
    print("PEFT is for reward model so load sequence classification.")
    print("Initializing pre-trained model from peft config...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=1,
        torch_dtype=torch_dtype,
    )
    print("Pre-trained model initialized!")
else:
    reward_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch_dtype,
    )

reward_tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Lora model
print("Loading the LoRA checkpoint...")
reward_model = PeftModel.from_pretrained(reward_model, peft_model_id)
print("LoRA checkpoint loaded!")
reward_model.eval()

print("Merging LoRA checkpoint...")
reward_model = reward_model.merge_and_unload()
print("LoRA checkpoint merged!")

save_path = "/cluster/work/sachan/sauc/nlf/pythia_quark_TLDR/pythia-410m-tldr-sft-rm-adapter-merged"
print("Saving reward_model and reward_tokenizer to ease loading...")
reward_model.save_pretrained(save_path)
reward_tokenizer.save_pretrained(save_path)
print(f"Correctly saved to {save_path}")