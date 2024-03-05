import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='path to root directory containing the input json file and where output files will be saved')
parser.add_argument('--input_json_file', required=True, help='json file name with sampled reward data')
parser.add_argument('--output_file_prefix', required=True, help='output file prefix ')
args = parser.parse_args()

def compute_and_save_histogram(jsonl_file, output_file_prefix):
    # Lists to store values
    reward_values = []
    quantile_tokens = []

    # Read the JSONL file
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except Exception as e:
                print(e)
                print(i+1)
            reward_values.append(entry['reward'])
            quantile_tokens.append(entry['quantile_token'])

    # Convert lists to NumPy arrays
    reward_values = np.array(reward_values)

    # Find the indices where quantile changes from 4 to 3, 3 to 2, and 2 to 1
    th4_index = next(i for i, q in enumerate(quantile_tokens) if q == "_QUANTILE_TOKEN_4_" and quantile_tokens[i - 1] != "_QUANTILE_TOKEN_4_")
    th3_index = next(i for i, q in enumerate(quantile_tokens) if q == "_QUANTILE_TOKEN_3_" and quantile_tokens[i - 1] != "_QUANTILE_TOKEN_3_")
    th2_index = next(i for i, q in enumerate(quantile_tokens) if q == "_QUANTILE_TOKEN_2_" and quantile_tokens[i - 1] != "_QUANTILE_TOKEN_2_")
    th1_index = next(i for i, q in enumerate(quantile_tokens) if q == "_QUANTILE_TOKEN_1_" and quantile_tokens[i - 1] != "_QUANTILE_TOKEN_1_")

    # Calculate the thresholds
    th4 = reward_values[th4_index]
    th3 = reward_values[th3_index]
    th2 = reward_values[th2_index]
    th1 = reward_values[th1_index]
    print(th4)
    print(th3)
    print(th2)
    print(th1)

    # Compute statistics
    reward_mean = np.mean(reward_values)
    reward_std = np.std(reward_values)

    # Plot and save histograms with mean and std
    plt.figure(figsize=(18, 6))
    
    # Define bin edges
    bins = 200
    bin_edges = np.histogram_bin_edges(reward_values, bins=bins)

    # Calculate the number of bins in each color region
    bins_per_region = [
        np.sum((bin_edges >= bin_start) & (bin_edges <= bin_end))
        for bin_start, bin_end, _ in [(bin_edges[0], th4, 'red'), (th4, th3, 'orange'), (th3, th2, 'yellow'), (th2, th1, 'lightgreen'), (th1, bin_edges[-1], 'green')]
    ]

    # Color mapping based on thresholds
    color_mapping = [
        (bin_edges[0], th4, 'red', bins_per_region[0], 'QUANTILE_TOKEN_4'),
        (th4, th3, 'orange', bins_per_region[1], 'QUANTILE_TOKEN_3'),
        (th3, th2, 'yellow', bins_per_region[2], 'QUANTILE_TOKEN_2'),
        (th2, th1, 'lightgreen', bins_per_region[3], 'QUANTILE_TOKEN_1'),
        (th1, bin_edges[-1], 'green', bins_per_region[4], 'QUANTILE_TOKEN_0')
    ]

    # Color the histogram bins based on thresholds
    for start, end, color, num_bins, label in color_mapping:
        plt.hist(reward_values, bins=np.linspace(start, end, num_bins+1), color=color, edgecolor='black', alpha=0.7, label=label)
        plt.legend()

    # Add mean and std lines
    plt.axvline(reward_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {reward_mean:.2f}')
    plt.axvspan(reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color='red', label=f'Std: {reward_std:.2f}')

    # Plot details
    plt.title('Reward Histogram')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    # Save the histogram
    plt.savefig(f"{output_file_prefix}_reward_histogram.png")
    plt.close()

    print(f"Histogram saved to {output_file_prefix}_reward_histogram.png")

def main():
    root_path = args.root_path
    compute_and_save_histogram(
        jsonl_file=f"{root_path}/{args.input_json_file}", 
        output_file_prefix=f"{root_path}/{args.output_file_prefix}"
    )
    
if __name__ == "__main__":
    main()
