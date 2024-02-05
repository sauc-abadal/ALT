from tasks.summarization.models.policy import NLFPolicy
from tasks.summarization.datasets.dataset_and_dataloader import TLDRDataset, PromptCollatorWithPadding
from utils import reduce_mean

from transformers import AutoTokenizer, GenerationConfig
import torch
from torch.utils.data import DataLoader

import json
from statistics import mean


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_checkpoints = {
        "Pre-trained": "EleutherAI/gpt-j-6b",
        "SFT": "CarperAI/openai_summarize_tldr_sft"
    }
    model_checkpoint_name = model_checkpoints["SFT"]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints["Pre-trained"], padding_side='left') # GPT2Tokenizer -> vocab_size 50257 (id from 0 to 50256) + extra_tokens for efficiency (id from 50257 to 50399) -> 50400 total vocabulary 
    if tokenizer.pad_token is None:
        print("Setting PAD token to EOS token for open-ended generation.")
        tokenizer.pad_token = tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy = NLFPolicy(
        model_checkpoint_name=model_checkpoint_name,
        device=device,
        tokenizer=tokenizer
    )
    print("Policy correctly initialized!")

    generation_config = GenerationConfig(
        max_length = 2048,
        max_new_tokens = 256,
        do_sample = False, # False means greedy decoding
        num_beams = 1, # no beam search
        temperature = 1.0, 
        top_k = 50, # number of highest prob. vocabulary tokens to keep for top-k filtering
        top_p = 1.0, # if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top-P or higher are kept for generation
        bad_words_ids = None, # List[List[int]] -> useful for Quark-based to avoid sampling of newly added tokens | list of list of tokens ids that are not allowed to be generated
        num_return_sequences = 1, # may be interesting to sample many completions for which to collect feedback    
        return_dict_in_generate = True,
        pad_token_id = tokenizer.pad_token_id, # error if not passed...
    )

    datasets = TLDRDataset(
        local_or_remote_path="CarperAI/openai_summarize_tldr",
        tokenizer=tokenizer,
        splits=["train", "valid", "test"],
        remote=True
    )
    prompt_collator = PromptCollatorWithPadding(tokenizer)
    print("Dataset and collator corretly initialized!")

    valid_dataset = datasets.datasets["valid"]
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=2, collate_fn=prompt_collator)

    HF_valid_loss = []
    ours_valid_loss = []

    with open("output/TLDR_SFT_greedy_decoding_valid.jsonl", "w") as jsonl_file:
        for n_batch, batch in enumerate(valid_dataloader):
            
            input_dict = batch["inputs"]
            prompts = batch["prompts"]
            summaries = batch["summaries"]

            output_dict = policy.sample(
                input_ids=input_dict["input_ids"],
                attention_mask=input_dict["attention_mask"],
                generation_config=generation_config,
            )

            input_ids = output_dict["input_ids"]
            attention_mask = output_dict["attention_mask"]
            generated_input_ids = output_dict["generated_input_ids"]
            generated_attention_mask = output_dict["generated_attention_mask"]
            generated_text = output_dict["generated_text"]
            
            outputs = policy.forward_pass(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generated_input_ids=generated_input_ids,
                generated_attention_mask=generated_attention_mask,
            )
            entropy = reduce_mean(value=outputs["generated_entropy"], mask=generated_attention_mask, axis=-1)
            lm_loss = reduce_mean(value=outputs["lm_loss"], mask=generated_attention_mask, axis=-1)

            HF_valid_loss.append(outputs["loss"].item())
            ours_valid_loss.append(reduce_mean(value=outputs["lm_loss"], mask=generated_attention_mask).item())

            for i in range(len(prompts)):
                output_data = {
                    "prompt": prompts[i],
                    "generation": generated_text[i],
                    "entropy": entropy[i].item(),
                    "lm_loss": lm_loss[i].item(),
                }
                jsonl_file.write(json.dumps(output_data) + "\n")

        HF_valid_loss = mean(HF_valid_loss)
        ours_valid_loss = mean(ours_valid_loss)
        output_data = {
            "HF_valid_loss": HF_valid_loss,
            "ours_valid_loss": ours_valid_loss
        }
        jsonl_file.write(json.dumps(output_data) + "\n")

if __name__ == "__main__":
    main()


