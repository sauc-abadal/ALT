import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='path to root directory containing the input json file and where output files will be saved')
parser.add_argument('--input_json_file', required=True, help='json file name with sampled len data')
parser.add_argument('--output_file_prefix', required=True, help='output file prefix')
parser.add_argument('--references', required=True, help='boolean, whether the key should be "generation" or "summary"')
args = parser.parse_args()

def compute_and_save_histogram(jsonl_file, output_file_prefix):

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

    # Lists to store values
    generations = []

    if args.references == "True":
        key = 'summary'
    else:
        key = 'generation'

    # Read the JSONL file
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            entry = json.loads(line)
            generations.append(entry[key])

    encoded_generations = tokenizer(generations)["input_ids"]
    generations_lens = [len(encoded_gen) for encoded_gen in encoded_generations]

    # Convert lists to NumPy arrays
    generations_lens = np.array(generations_lens)

    # Compute statistics
    len_mean = np.mean(generations_lens)
    len_std = np.std(generations_lens)

    # Plot and save histograms with mean and std
    plt.figure(figsize=(18, 6))
    # len Histogram
    plt.hist(generations_lens, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(len_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {len_mean:.2f}')
    plt.axvspan(len_mean - len_std, len_mean + len_std, alpha=0.2, color='red', label=f'Std: {len_std:.2f}')
    plt.title('Generations lengths Histogram')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_gen_lengths_histogram.png")
    plt.close()

    print(f"Histogram saved to {output_file_prefix}_gen_lengths_histogram.png")

def main():
    root_path = args.root_path
    compute_and_save_histogram(
        jsonl_file=f"{root_path}/{args.input_json_file}", 
        output_file_prefix=f"{root_path}/{args.output_file_prefix}")
    
if __name__ == "__main__":
    main()

