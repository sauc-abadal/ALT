import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")

from tasks.summarization.datasets.sampling_dataset_and_collator import TLDRSamplingDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json


def main():
    
    out_file_path = "/cluster/work/sachan/sauc/nlf/quark_TLDR_5q/sampling/references_data_valid.json" 

    print("Loading data...")

    splits = ["valid"]

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6b",
        padding_side="left", 
        max_length=1024)
    

    datasets = TLDRSamplingDataset(
        local_or_remote_path="CarperAI/openai_summarize_tldr",
        tokenizer=tokenizer,
        data_format=None,
        splits=splits,
        remote=True
    )
    dataset = datasets.datasets[splits[0]]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    print(f"Dataset loaded with {len(dataset)} samples | Dataloader with {len(dataloader)} batches")
        
    prompts, ref_summaries = [], []
    for batch in dataloader:
        prompt = batch["prompt"]
        ref_summary = batch["summary"]

        prompts.extend(prompt)
        ref_summaries.extend(ref_summary)

    with open(out_file_path, 'w') as f:
        for prompt, summary in zip(prompts, ref_summaries):
            save_dict = {
                "prompt": prompt,
                "summary": summary,
            }
            json.dump(save_dict, f)
            f.write('\n')

if __name__ == "__main__":
    main()

