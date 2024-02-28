import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='path to root directory containing the input json files and where output files will be saved')
parser.add_argument('--input_json_files', nargs='+', required=True, help='json files with sampled reward data')
parser.add_argument('--output_file_prefixes', nargs='+', required=True, help='output file prefixes')
args = parser.parse_args()

def compute_stats_and_save_subplots(jsonl_files, output_file_prefixes):
    num_files = len(jsonl_files)
    
    fig, axes = plt.subplots(nrows=num_files, ncols=1, figsize=(18, 6 * num_files))

    common_xlim = None

    for i, (jsonl_file, output_file_prefix) in enumerate(zip(jsonl_files, output_file_prefixes)):

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

        ax = axes[i]

        _, bins, _ = ax.hist(reward_values, bins=100, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(reward_mean, color='red', linestyle='dashed', linewidth=2, alpha=1.0, label=f'Mean: {reward_mean:.2f}')
        ax.axvspan(reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color='red', label=f'Std: {reward_std:.2f}')

        ax.set_title(f'Reward Histogram - {output_file_prefix}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.legend()

        # Update common x-axis limits
        if common_xlim is None:
            common_xlim = ax.get_xlim()
        else:
            common_xlim = (min(common_xlim[0], bins[0]), max(common_xlim[1], bins[-1]))

    # Set common x-axis limits for all subplots
    for ax in axes:
        ax.set_xlim(common_xlim)

    plt.tight_layout()
    plt.savefig(f"{args.root_path}/individual_reward_histograms.png")
    plt.close()

    print(f"Histograms saved to {args.root_path}/individual_reward_histograms.png")

def main():
    root_path = args.root_path
    compute_stats_and_save_subplots(
        jsonl_files=[f"{root_path}/{json_file}" for json_file in args.input_json_files],
        output_file_prefixes=args.output_file_prefixes)

if __name__ == "__main__":
    main()
