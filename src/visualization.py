import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_attention_3d(points, attention_classes, save_path=None):
    # Use a sequential colormap instead of discrete colors
    cmap = plt.cm.RdYlBu_r  # Red (high) to Blue (low)
    
    # Normalize classes to [0,1] for colormap
    normalized_classes = attention_classes / 4.0  # Assuming 5 classes (0-4)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with continuous colormap
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=normalized_classes, cmap=cmap, alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['No attention', 'Low', 'Medium-low', 'Medium-high', 'High'])
    
    plt.title('3D Attention Heatmap')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close() 