import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# File paths
path = './files_results/'
ids = ['100n_2500s_3ad_6md_1exp_10rep_', '200n_2500s_3ad_6md_1exp_10rep_', '500n_2500s_3ad_6md_1exp_10rep_',
       '1000n_2500s_3ad_6md_1exp_10rep_']

# Prepare containers for the metrics
metrics = {'prec_adj': [], 'rec_adj': [], 'delta_shd': []}
nodes = [100, 200, 500, 1000]  # Corresponding number of nodes

# Load the data from the files
for id in ids:
    input_name = path + id + 'files_mb_cd.pkl'

    with open(input_name, 'rb') as f:
        files = pickle.load(f)

    # Aggregate precision, recall, and delta_shd across reps
    prec = [files[rep]['adj_prec'] for rep in range(10)]
    rec = [files[rep]['adj_rec'] for rep in range(10)]
    delta_shd = [files[rep]['delta_shd'] for rep in range(10)]
    print(rec)
    print(prec)
    print('-----------------')
    metrics['prec_adj'].append(prec)
    metrics['rec_adj'].append(rec)
    metrics['delta_shd'].append(delta_shd)

# Create the plot with improved line colors and explanation box without statistical values
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Colors and labels for better visualization
colors = ['skyblue', 'salmon', 'lightgreen']
mean_line_color = 'darkblue'
median_line_color = 'crimson'
other_line_color = 'black'  # Changed to black for better visibility
titles = ['Adj Precision', 'Adj Recall', 'Δ SHD']

# Iterate over the metrics to create box plots for better visualization
for i, (metric, title, color) in enumerate(zip(['prec_adj', 'rec_adj', 'delta_shd'], titles, colors)):
    bp = axs[i].boxplot(metrics[metric], patch_artist=True, showmeans=True)

    for patch in bp['boxes']:
        patch.set(facecolor=color, edgecolor='black', alpha=0.7)

    # Set different line colors for mean, median, and other elements
    if 'means' in bp:
        for mean in bp['means']:
            mean.set(color=mean_line_color, linewidth=2)
    if 'medians' in bp:
        for median in bp['medians']:
            median.set(color=median_line_color, linewidth=2)
    if 'whiskers' in bp:
        for whisker in bp['whiskers']:
            whisker.set(color=other_line_color, linewidth=1.5)
    if 'caps' in bp:
        for cap in bp['caps']:
            cap.set(color=other_line_color, linewidth=1.5)

    axs[i].set_title(title, fontsize=16)
    axs[i].set_xlabel('Number of Nodes', fontsize=14)
    axs[i].set_ylabel(title, fontsize=14)
    axs[i].set_xticks([1, 2, 3, 4])
    axs[i].set_xticklabels(nodes)
    axs[i].grid(True, linestyle='--', alpha=0.6)

# Add a legend to explain the line colors for mean and median with the actual colors
fig.legend(
    handles=[
        plt.Line2D([0], [0], color=mean_line_color, lw=2, label='Mean'),
        plt.Line2D([0], [0], color=median_line_color, lw=2, label='Median')
    ],
    loc='lower center', fontsize=12, bbox_to_anchor=(0.5, -0.01), frameon=False
)

# Add an overall title
fig.suptitle('CL: Graph Structure', fontsize=20, fontweight='bold')

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('cl_plot')

# Show the improved plot
# plt.show()