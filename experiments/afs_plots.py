import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# File paths
path = './files_results/'
ids = ['100n_2500s_3ad_6md_1exp_10rep_', '200n_2500s_3ad_6md_1exp_10rep_', '500n_2500s_3ad_6md_1exp_10rep_',
       '1000n_2500s_3ad_6md_1exp_10rep_']

# Prepare containers for the metrics
metrics = {'prec_mb': [], 'rec_mb': [], 'deltar2': []}
nodes = [100, 200, 500, 1000]  # Corresponding number of nodes

# Load the data from the files
for id in ids:
    input_name = path + id + 'files_mb.pkl'

    with open(input_name, 'rb') as f:
        files = pickle.load(f)

    # Aggregate precision, recall, and deltar2 across reps
    prec = [files[rep]['prec_mb'] for rep in range(10)]
    rec = [files[rep]['rec_mb'] for rep in range(10)]
    delta_r2 = [files[rep]['deltar2'] for rep in range(10)]

    print(prec)
    print(rec)
    print('------------------')

    metrics['prec_mb'].append(prec)
    metrics['rec_mb'].append(rec)
    metrics['deltar2'].append(delta_r2)

# Create the plot with improved line colors and explanation box without statistical values
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Colors and labels for better visualization
colors = ['skyblue', 'salmon', 'lightgreen']
mean_line_color = 'darkblue'
median_line_color = 'crimson'
titles = ['Mb Precision', 'Mb Recall', r'$\Delta R^2$']

# Iterate over the metrics to create boxplots for better visualization
for i, (metric, title, color) in enumerate(zip(['prec_mb', 'rec_mb', 'deltar2'], titles, colors)):
    parts = axs[i].boxplot(metrics[metric], patch_artist=True, showmeans=True, meanline=True)

    for patch in parts['boxes']:
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.7)

    # Set different line colors for mean, median, and other elements
    if 'means' in parts:
        for mean in parts['means']:
            mean.set_color(mean_line_color)
            mean.set_linewidth(2)
    if 'medians' in parts:
        for median in parts['medians']:
            median.set_color(median_line_color)
            median.set_linewidth(2)
    if 'caps' in parts:
        for cap in parts['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.5)
    if 'whiskers' in parts:
        for whisker in parts['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.5)

    axs[i].set_title(title, fontsize=16)
    axs[i].set_xlabel('Number of Nodes', fontsize=14)
    axs[i].set_ylabel(title, fontsize=14)
    axs[i].set_xticks([1, 2, 3, 4])
    axs[i].set_xticklabels(nodes)
    axs[i].grid(True, linestyle='--', alpha=0.6)
    if (i == 2):
        axs[i].set_ylim(0, 0.1)
    elif (i == 0):
        axs[i].set_ylim(0.8, 1.0)

# Add a legend to explain the line colors for mean and median with the actual colors
fig.legend(
    handles=[
        plt.Line2D([0], [0], color=mean_line_color, lw=2, label='Mean'),
        plt.Line2D([0], [0], color=median_line_color, lw=2, label='Median')
    ],
    loc='lower center', fontsize=12, bbox_to_anchor=(0.5, -0.01), frameon=False
)

# Add an overall title
fig.suptitle('AFS: Mb Identification Metrics', fontsize=20, fontweight='bold')

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('afs_plot')

# Show the improved plot
# plt.show()