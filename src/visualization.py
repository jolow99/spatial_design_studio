import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_attention_3d(points, attention_classes, save_path=None):
    # Define colors for each class
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    class_names = ['No attention', 'Low', 'Medium-low', 'Medium-high', 'High']
    
    # Create color array based on classes
    point_colors = [colors[cls] for cls in attention_classes]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with colors based on attention class
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=point_colors, alpha=0.6)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=c, label=class_names[i],
                                 markersize=10)
                      for i, c in enumerate(colors)]
    ax.legend(handles=legend_elements)
    
    plt.title('3D Attention Heatmap')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close() 