import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='path to root directory containing the input json file and where output files will be saved')
parser.add_argument('--input_json_file', required=True, help='json file name with sampled reward data')
parser.add_argument('--output_file_prefix', required=True, help='output file prefix')
parser.add_argument('--references', required=True, help='boolean, whether the key should be "generation" or "summary"')
args = parser.parse_args()

def compute_and_save_corr(jsonl_file, output_file_prefix):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

    # Lists to store values
    generations, rewards = [], []

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
            rewards.append(entry["reward"])

    encoded_generations = tokenizer(generations)["input_ids"]
    generations_lens = [len(encoded_gen) for encoded_gen in encoded_generations]

    # Convert lists to NumPy arrays
    generations_lens = np.array(generations_lens)
    rewards = np.array(rewards)

    # Plot the correlation between rewards and generation lengths
    plt.scatter(generations_lens, rewards, color='tomato', alpha=0.2)
    plt.title('Correlation between Rewards and Generation Lengths')
    plt.ylabel('Rewards')
    plt.xlabel('Generation Lengths')

    z = np.polyfit(generations_lens, rewards, 1)
    p = np.poly1d(z)
    plt.plot(generations_lens, p(generations_lens), color='red', linewidth=2)

    # Compute Pearson correlation coefficient
    correlation_coefficient = np.corrcoef(generations_lens, rewards)[0, 1]

    # Display Pearson correlation coefficient in the legend
    plt.legend([f'Pearson Correlation: {correlation_coefficient:.2f}'])
    
    # Save the plot as an image file
    plt.savefig(f'{output_file_prefix}_correlation_plot.png')

def main():
    root_path = args.root_path
    compute_and_save_corr(
        jsonl_file=f"{root_path}/{args.input_json_file}", 
        output_file_prefix=f"{root_path}/{args.output_file_prefix}")

if __name__ == "__main__":
    main()
