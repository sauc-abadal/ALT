import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='path to root directory containing the input json file and where output files will be saved')
parser.add_argument('--input_json_file', required=True, help='json file name with sampled ppl data')
parser.add_argument('--output_file_prefix', required=True, help='output file prefix ')
args = parser.parse_args()

def compute_and_save_histogram(jsonl_file, output_file_prefix):
    # Lists to store values
    ppl_values = []

    # Read the JSONL file
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            entry = json.loads(line)
            ppl_values.append(entry['perplexity'])

    # Convert lists to NumPy arrays
    ppl_values = np.array(ppl_values)

    # Compute statistics
    ppl_mean = np.nanmean(ppl_values)
    ppl_std = np.nanstd(ppl_values)

    # Plot and save histograms with mean and std
    plt.figure(figsize=(18, 6))
    # Perplexity Histogram
    plt.hist(ppl_values, bins=50, color='palegreen', edgecolor='black', alpha=0.7)
    plt.axvline(ppl_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {ppl_mean:.2f}')
    plt.axvspan(ppl_mean - ppl_std, ppl_mean + ppl_std, alpha=0.2, color='red', label=f'Std: {ppl_std:.2f}')
    plt.title('Perplexity Histogram')
    plt.xlabel('ppl')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_ppl_histogram.png")
    plt.close()

    print(f"Histogram saved to {output_file_prefix}_ppl_histogram.png")

def main():
    root_path = args.root_path
    compute_and_save_histogram(
        jsonl_file=f"{root_path}/{args.input_json_file}", 
        output_file_prefix=f"{root_path}/{args.output_file_prefix}"
    )
    
if __name__ == "__main__":
    main()

