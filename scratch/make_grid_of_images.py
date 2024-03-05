import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def save_image_grid(images, grid_size, output_filename):
    """
    Create a grid of images and save it to a file.

    Parameters:
    - images: List of image filenames
    - grid_size: Tuple specifying the number of rows and columns in the grid
    - output_filename: Filename to save the aggregated image

    Returns:
    - None (saves the image to the specified filename)
    """
    # fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 6), dpi=300)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(18, 6), dpi=300)

    """
    plot_titles = ["Quark sampled data stage 1 (train split)", 
                   "Quark sampled data stage 2 (train split)", 
                   "Quark sampled data stage 3 (train split)"]
    """
    """
    plot_titles = ["Quark sampled data stage 1 (train split)",
                   "",
                   "Quark sampled data stage 2 (train split)", 
                   "",
                   "Quark sampled data stage 3 (train split)",
                   ""]
    """
    plot_titles = ["References - human-written (valid split)",
                   "SFT sampled data (valid split)", 
                   "Quark sampled data stage 1 (valid split)",
                   "Quark sampled data stage 2 (valid split)",
                   "Quark sampled data stage 3 (valid split)"]
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = mpimg.imread(images[i])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(plot_titles[i], fontsize=6)
        else:
            ax.axis('off')

    # plt.subplots_adjust(wspace=0.15)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

# Example usage:
# root_path = "/home/sauc/Personal/Thesis/Work/Plots/sampled_data/train_split"
root_path = "/home/sauc/Personal/Thesis/Work/Plots/sampled_data/valid_split"

"""
image_filenames = [f'{root_path}/quark_train_stage_1_correlation_plot.png', 
                   f'{root_path}/quark_train_stage_2_correlation_plot.png', 
                   f'{root_path}/quark_train_stage_3_correlation_plot.png']
"""
"""
image_filenames = [f'{root_path}/quark_train_stage_1_reward_histogram.png', 
                   f'{root_path}/quark_train_stage_1_gen_lengths_histogram.png',  
                   f'{root_path}/quark_train_stage_2_reward_histogram.png',
                   f'{root_path}/quark_train_stage_2_gen_lengths_histogram.png',  
                   f'{root_path}/quark_train_stage_3_reward_histogram.png', 
                   f'{root_path}/quark_train_stage_3_gen_lengths_histogram.png']
"""

image_filenames = [f'{root_path}/references_valid_reward_ppl_len_histograms.png',  
                   f'{root_path}/SFT_valid_reward_ppl_len_histograms.png',
                   f'{root_path}/quark_valid_stage_1_reward_ppl_len_histograms.png', 
                   f'{root_path}/quark_valid_stage_2_reward_ppl_len_histograms.png',  
                   f'{root_path}/quark_valid_stage_3_reward_ppl_len_histograms.png']
"""
image_filenames = [f'{root_path}/references_valid_correlation_plot.png',  
                   f'{root_path}/SFT_valid_correlation_plot.png',
                   f'{root_path}/quark_valid_stage_1_correlation_plot.png', 
                   f'{root_path}/quark_valid_stage_2_correlation_plot.png',  
                   f'{root_path}/quark_valid_stage_3_correlation_plot.png']
"""
# grid_size = (3, 1)
# grid_size = (3, 2)
grid_size = (5, 1)

# output_filename = f'{root_path}/trends_stage_1_2_3_corr.png'
# output_filename = f'{root_path}/trends_stage_1_2_3.png'
output_filename = f'{root_path}/trends_stage_ref_SFT_1_2_3.png'
# output_filename = f'{root_path}/trends_stage_ref_SFT_1_2_3_corr.png'

save_image_grid(image_filenames, grid_size, output_filename)
