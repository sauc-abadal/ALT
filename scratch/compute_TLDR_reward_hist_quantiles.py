import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def plot_histogram(data, output_file):
    rewards = [entry['reward_score'] for entry in data]
    quantile_tokens = [entry['quantile_token'] for entry in data]

    # Plot histogram with quantile-based coloring
    fig, ax = plt.subplots()
    quantiles = [int(token.split('_')[-2]) for token in quantile_tokens]
    colormap = plt.cm.get_cmap('viridis', len(set(quantiles)))

    ax.hist(rewards, bins=5, color=colormap(quantiles), edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Rewards')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Rewards with Quantile-Based Coloring')

    plt.savefig(output_file)

def main():
    parser = argparse.ArgumentParser(description='Generate a histogram of rewards with quantile-based coloring.')
    parser.add_argument('input_json_file', type=str, help='Path to the input JSON file.')
    parser.add_argument('output_prefix', type=str, help='Prefix for the saved figure.')
    parser.add_argument('root_path', type=str, help='Root path for reading the JSON file and saving the figure.')

    args = parser.parse_args()

    input_file_path = os.path.join(args.root_path, args.input_json_file)
    output_file_path = os.path.join(args.root_path, f"{args.output_prefix}_reward_quantile_histogram.png")

    data = load_json_data(input_file_path)
    plot_histogram(data, output_file_path)

if __name__ == "__main__":
    main()
