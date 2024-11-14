import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_combined_point_clouds(sphere_file, cube_file):
    # Read both datasets
    sphere_df = pd.read_csv(sphere_file)
    cube_df = pd.read_csv(cube_file)
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(15, 6))
    
    # Sphere subplot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(sphere_df['x'], sphere_df['y'], sphere_df['z'], 
                          c=sphere_df['NormalizedScore'], cmap='hot', marker='o')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Sphere Point Cloud')
    plt.colorbar(scatter1, ax=ax1, label='Saliency Score')
    
    # Cube subplot
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(cube_df['x'], cube_df['y'], cube_df['z'], 
                          c=cube_df['NormalizedScore'], cmap='hot', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Cube Point Cloud')
    plt.colorbar(scatter2, ax=ax2, label='Saliency Score')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Visualize both sphere and cube
plot_combined_point_clouds('curved1saliency_kdtree.csv', 'kdtree_rect1.csv')