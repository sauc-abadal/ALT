import argparse
from pathlib import Path
import os
import math

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def conditional_perplexity(generations_df, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating fluency'):
        prompt = row["prompt"]
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids).loss * (prompt_input_ids.shape[1]-1)

        # for every generation conditioned on the prompt
        generations = [g['text'] for g in row['generations'] if g['text']]
        for gen in generations:
            full_input_ids = tokenizer.encode(prompt+gen, return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids).loss * (full_input_ids.shape[1]-1)

            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            ppl = math.exp(loss.item())
            if ppl < 1e4:   # for sanity
                perplexities.append(ppl)
                
    return np.nanmean(perplexities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to evaluate conditional perplexity after running sample.py')
    parser.add_argument('--save_path', type=str, default='output/savepath', help='Path to the directory containing the generations file (same output directory specified in sample.py)')
    args = parser.parse_args()

    save_path = args.save_path
    generations_file = f'{save_path}/reward.json'
    print(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    assert os.path.exists(generations_file)
    generations_df = pd.read_json(generations_file, lines=True)

    # calculate fluency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
    eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    print('model initialization done!')

    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device)
    print(f'perplexity = {ppl}')

    # write output results
    with open(output_dir / 'eval_results.txt', 'a') as fo:
        fo.write(f'perplexity = {ppl}\n')