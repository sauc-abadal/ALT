import json
import numpy as np
import matplotlib.pyplot as plt

def compute_stats_and_save_histograms(jsonl_file, output_file_prefix):
    # Lists to store values
    entropy_values = []
    lm_loss_values = []
    reward_values = []

    # Read the JSONL file, excluding the last line
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()[:-1]  # Exclude the last line
        for line in lines:
            entry = json.loads(line)
            entropy_values.append(entry['entropy'])
            lm_loss_values.append(entry['lm_loss'])
            reward_values.append(entry['reward'])

    # Convert lists to NumPy arrays
    entropy_values = np.array(entropy_values)
    lm_loss_values = np.array(lm_loss_values)
    reward_values = np.array(reward_values)

    # Compute statistics
    entropy_mean = np.mean(entropy_values)
    entropy_std = np.std(entropy_values)

    lm_loss_mean = np.mean(lm_loss_values)
    lm_loss_std = np.std(lm_loss_values)

    reward_mean = np.mean(reward_values)
    reward_std = np.std(reward_values)

    # Plot and save histograms with mean and std
    plt.figure(figsize=(18, 6))

    # Entropy Histogram
    plt.subplot(1, 3, 1)
    plt.hist(entropy_values, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(entropy_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {entropy_mean:.2f}')
    plt.axvspan(entropy_mean - entropy_std, entropy_mean + entropy_std, alpha=0.2, color='red', label=f'Std: {entropy_std:.2f}')
    plt.title('Entropy Histogram')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.legend()

    # LM Loss Histogram
    plt.subplot(1, 3, 2)
    plt.hist(lm_loss_values, bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.axvline(lm_loss_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {lm_loss_mean:.2f}')
    plt.axvspan(lm_loss_mean - lm_loss_std, lm_loss_mean + lm_loss_std, alpha=0.2, color='red', label=f'Std: {lm_loss_std:.2f}')
    plt.title('LM Loss Histogram')
    plt.xlabel('LM Loss')
    plt.ylabel('Frequency')
    plt.legend()

    # Reward Histogram
    plt.subplot(1, 3, 3)
    plt.hist(reward_values, bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.axvline(reward_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {reward_mean:.2f}')
    plt.axvspan(reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color='red', label=f'Std: {reward_std:.2f}')
    plt.title('Reward Histogram')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_histograms.png")
    plt.close()

    print(f"Histograms saved to {output_file_prefix}_histograms.png")

    # Save computed statistics to a text file
    with open(f"{output_file_prefix}_statistics.txt", 'w') as output:
        output.write(f"Entropy Mean: {entropy_mean}\n")
        output.write(f"Entropy Standard Deviation: {entropy_std}\n")
        output.write(f"LM Loss Mean: {lm_loss_mean}\n")
        output.write(f"LM Loss Standard Deviation: {lm_loss_std}\n")
        output.write(f"Reward Mean: {reward_mean}\n")
        output.write(f"Reward Standard Deviation: {reward_std}\n")

    print(f"Statistics saved to {output_file_prefix}_statistics.txt")

# Replace 'your_file.jsonl' and 'output_results' with the actual file paths
compute_stats_and_save_histograms(
    jsonl_file='output/TLDR_SFT_greedy_decoding_valid.jsonl', 
    output_file_prefix='output/TLDR_SFT_greedy_decoding_valid')

