from dataset_and_dataloader import TLDRSamplingDataset, TLDRSamplingPromptCollatorWithPadding
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b") # GPT2Tokenizer -> vocab_size 50257 (id from 0 to 50256) + extra_tokens for efficiency (id from 50257 to 50399) -> 50400 total vocabulary 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # as GPT-J's tokenizer doesn't have a padding token -> eos_token = bos_token = unk_token = pad_token = "<|endoftext|>", eos_token_id = bos_token_id = unk_token_id = pad_token_id = 50256
    datasets = TLDRSamplingDataset(
        local_or_remote_path="CarperAI/openai_summarize_tldr",
        tokenizer=tokenizer,
        splits=["train", "valid", "test"],
        remote=True
    )
    print(len(datasets.datasets["train"])) # __len__() function already implemented
    print(datasets.datasets["train"][6]) # __getitem__() function already implemented
    print(len(datasets.datasets["train"][6]["prompt_input_ids"])) # prompt's num_tokens for a random example 

    prompt_collator = TLDRSamplingPromptCollatorWithPadding(tokenizer)

    train_dataset = datasets.datasets["train"]
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8, collate_fn=prompt_collator)

    for batch in train_dataloader:
        inputs = batch["inputs"]
        prompts = batch["prompts"]
        summaries = batch["summaries"]
        print(f"input_ids: {inputs["input_ids"].shape}")
        print(f"attention_mask: {inputs["attention_mask"].shape}")                      
        print(inputs["input_ids"]) # input tensors start and end with token_ids: [50, 10526, 22083] = 'S' 'UB' 'RED' ("SUBREDDIT: "); [7707, 25, 220] = 'DR' ':' 'Ġ' ("TL;DR: ")
        print(inputs["attention_mask"])
        print(prompts[6])
        print(summaries[6])
        break

if __name__ == "__main__":
    main()