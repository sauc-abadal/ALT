from tasks.summarization.models.policy import Policy
from tasks.summarization.datasets.sampling_dataset_and_collator import TLDRSamplingDataset, TLDRSamplingPromptCollatorWithPadding

from transformers import AutoTokenizer, GenerationConfig
import torch
from torch.utils.data import DataLoader

from utils import reduce_mean

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

    policy = Policy(
        model_checkpoint_name=model_checkpoint_name,
        device=device,
        tokenizer=tokenizer
    )
    print("Policy correctly initialized!")

    generation_config = GenerationConfig(
        max_length = 2048,
        max_new_tokens = 256,
        do_sample = True, # False means greedy decoding
        num_beams = 1, # no beam search
        temperature = 1.0, 
        top_k = 50, # number of highest prob. vocabulary tokens to keep for top-k filtering
        top_p = 1.0, # if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top-P or higher are kept for generation
        bad_words_ids = None, # List[List[int]] -> useful for Quark-based to avoid sampling of newly added tokens | list of list of tokens ids that are not allowed to be generated
        num_return_sequences = 1, # may be interesting to sample many completions for which to collect feedback    
        return_dict_in_generate = True,
        pad_token_id = tokenizer.pad_token_id, # error if not passed...
    )

    datasets = TLDRSamplingDataset(
        local_or_remote_path="CarperAI/openai_summarize_tldr",
        tokenizer=tokenizer,
        splits=["train", "valid", "test"],
        remote=True
    )
    prompt_collator = TLDRSamplingPromptCollatorWithPadding(tokenizer)
    print("Dataset and collator corretly initialized!")

    train_dataset = datasets.datasets["train"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=prompt_collator)

    for batch in train_dataloader:
        input_dict = batch["inputs"]
        prompts = batch["prompts"]
        summaries = batch["summaries"]
        print("------------------------------------------------")
        print(f"Prompt text: {prompts}")
        print("------------------------------------------------")
        print("Sampling a batch of generations...")
        print("------------------------------------------------")
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
        print(f"Prompt input IDs shape: {input_ids.shape}")
        print(f"Prompt attention mask shape: {attention_mask.shape}")
        print(f"Prompt input IDs: {input_ids}")
        print(f"Prompt attention mask: {attention_mask}")
        print("------------------------------------------------")
        print(f"Generated input IDs shape: {generated_input_ids.shape}")
        print(f"Generated attention mask shape: {generated_attention_mask.shape}")
        print(f"Generated input IDs: {generated_input_ids}")
        print(f"Generated attention mask shape: {generated_attention_mask}")
        print(f"Generated text: {generated_text}")

        outputs = policy.forward_pass(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generated_input_ids=generated_input_ids,
            generated_attention_mask=generated_attention_mask,
        )
        print("------------------------------------------------")
        print(f"Generated logits shape: {outputs["generated_logits"].shape}")
        print(f"Generated logits: {outputs["generated_logits"]}")
        print(f"Generated logprobs shape: {outputs["generated_logprobs"].shape}")
        print(f"Generated logprobs: {outputs["generated_logprobs"]}")
        print(f"Generated entropy shape: {outputs["generated_entropy"].shape}")
        print(f"Generated entropy: {outputs["generated_entropy"]}")
        print(f"Generated lm_loss shape: {outputs["lm_loss"].shape}")
        print(f"Generated lm_loss: {outputs["lm_loss"]}")
        print("------------------------------------------------")
        print(f"HF loss computed by passing in the arg. 'labels' with -100 masking on prompt tokens: {outputs["loss"]}")
        print(f"Mean reduction (with masking on pad tokens of generation, except EOS token) of our lm_loss: {reduce_mean(value=outputs["lm_loss"], mask=generated_attention_mask)}")

        break

if __name__ == "__main__":
    main()


