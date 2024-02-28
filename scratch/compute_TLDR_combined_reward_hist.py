import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='path to root directory containing the input json files and where output files will be saved')
parser.add_argument('--input_json_files', nargs='+', required=True, help='json files with sampled reward data')
parser.add_argument('--output_file_prefixes', nargs='+', required=True, help='output file prefixes')
args = parser.parse_args()

def compute_stats_and_save_histograms(jsonl_files, output_file_prefixes):
    plt.figure(figsize=(18, 6))
    
    for jsonl_file, output_file_prefix in zip(jsonl_files, output_file_prefixes):
        # Lists to store values
        reward_values = []

        # Read the JSONL file
        with open(jsonl_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                entry = json.loads(line)
                reward_values.append(entry['reward'])

        # Convert lists to NumPy arrays
        reward_values = np.array(reward_values)

        # Compute statistics
        reward_mean = np.mean(reward_values)
        reward_std = np.std(reward_values)

        # Get a color for the histogram and remove transparency for the mean line
        hist_color = plt.hist(reward_values, bins=100, alpha=0.3, edgecolor='black', label=f'{output_file_prefix}')[2][0].get_facecolor()[:3]

        # Plot mean line with adjusted properties
        plt.axvline(reward_mean, color=hist_color, linestyle='solid', linewidth=1.5, alpha=1.0)
        plt.axvspan(reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color=hist_color)

    plt.title('Reward Histograms')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{args.root_path}/combined_reward_histogram.png")
    plt.close()

    print(f"Histograms saved to {args.root_path}/combined_reward_histogram.png")

def main():
    root_path = args.root_path
    compute_stats_and_save_histograms(
        jsonl_files=[f"{root_path}/{json_file}" for json_file in args.input_json_files],
        output_file_prefixes=args.output_file_prefixes)

if __name__ == "__main__":
    main()
