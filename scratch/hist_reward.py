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

    # Plot and save histograms with mean and std
    plt.figure(figsize=(18, 6))
    # Reward Histogram
    plt.hist(reward_values, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    plt.axvline(reward_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {reward_mean:.2f}')
    plt.axvspan(reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color='red', label=f'Std: {reward_std:.2f}')
    plt.title('Reward Histogram')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
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
