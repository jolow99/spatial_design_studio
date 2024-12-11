import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import math
import numpy as np

def convert_to_classes(normalized_scores):
    """Match the class conversion in data_loader.py"""
    bins = [0, 0.05, 0.10, 0.2, 1.0]
    classes = np.digitize(normalized_scores, bins) - 1
    # Handle the zero case separately
    classes[normalized_scores == 0] = 0
    return classes

def plot_point_clouds_batch(data_dir, subject_name, start_idx, batch_size=2):
    """
    Plot a batch of point clouds from the data directory with discrete class colors
    """
    # Colors and class names matching test.py style
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    class_names = ['No attention', 'Low', 'Medium-low', 'Medium-high', 'High']
    
    # Get batch of files
    curved_files = [f"{subject_name}_curved{i}score.csv" for i in range(start_idx, min(start_idx + batch_size, 16))]
    rect_files = [f"{subject_name}_rect{i}score.csv" for i in range(start_idx, min(start_idx + batch_size, 16))]
    all_files = curved_files + rect_files
    
    # Calculate grid dimensions
    n_files = len(all_files)
    n_cols = 2  # 2 columns (curved and rect side by side)
    n_rows = math.ceil(n_files / 2)
    
    # Create figure
    fig = plt.figure(figsize=(15, 6*n_rows))
    
    # Plot each point cloud
    for idx, file in enumerate(all_files, 1):
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: File {file} not found")
            continue
        
        # Read dataset
        df = pd.read_csv(file_path)
        points = df[['x', 'y', 'z']].values
        
        # Convert normalized scores using the same function as data_loader
        scores = df['NormalizedScore'].values
        classes = convert_to_classes(scores)
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
        
        # Plot points with discrete colors
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=[colors[int(c)] for c in classes],
                           s=2)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Extract form number and type from filename
        if 'curved' in file:
            form_num = file.split('curved')[1].split('score')[0]
            title = f'Curved Form #{form_num}'
        else:
            form_num = file.split('rect')[1].split('score')[0]
            title = f'Rect Form #{form_num}'
        
        ax.set_title(title, pad=20)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=c, label=class_names[i],
                                    markersize=10)
                         for i, c in enumerate(colors)]
        ax.legend(handles=legend_elements)
        
        # Set consistent view angle
        ax.view_init(elev=30, azim=45)
        
        # Set consistent axis limits
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.show()

def plot_comparison(data_dir, subject_name, form_number):
    """
    Plot specific curvilinear and rectilinear forms side by side with discrete classes
    """
    # Colors and class names matching test.py style
    colors = ['lightgray', 'green', 'yellow', 'orange', 'red']
    class_names = ['No attention', 'Low', 'Medium-low', 'Medium-high', 'High']
    
    # Construct filenames
    curved_file = f'{subject_name}_curved{form_number}score.csv'
    rect_file = f'{subject_name}_rect{form_number}score.csv'
    
    # Create figure
    fig = plt.figure(figsize=(20, 8))
    
    # Process both files
    for idx, (file, title) in enumerate([
        (curved_file, f'Curved Form #{form_number}'),
        (rect_file, f'Rect Form #{form_number}')
    ]):
        # Read dataset
        df = pd.read_csv(os.path.join(data_dir, file))
        points = df[['x', 'y', 'z']].values
        
        # Convert normalized scores using the same function as data_loader
        scores = df['NormalizedScore'].values
        classes = convert_to_classes(scores)
        
        # Create subplot
        ax = fig.add_subplot(121 + idx, projection='3d')
        
        # Create color and size arrays
        point_colors = []
        point_sizes = []
        point_alphas = []
        
        for c in classes:
            if c == 0:  # No attention points
                point_colors.append(colors[c])
                point_sizes.append(2)  # Smaller size for no attention points
                point_alphas.append(0.4)  # More transparent for no attention points
            else:  # Attention points
                point_colors.append(colors[c])
                point_sizes.append(8)  # Larger size for attention points
                point_alphas.append(1.0)  # Fully opaque for attention points
        
        # Plot points with varying sizes and transparency
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=point_colors,
                           s=point_sizes,
                           alpha=point_alphas)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, pad=20, fontsize=14)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=c, label=class_names[i],
                                    markersize=10,
                                    alpha=1.0 if i > 0 else 0.1)  # Make legend match point transparency
                         for i, c in enumerate(colors)]
        ax.legend(handles=legend_elements)
        
        # Set view angle
        ax.view_init(elev=30, azim=45)
        
        # Set consistent axis limits
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.show()

if __name__ == "__main__":
    subject_name = "Abram"  # Change this to "Sean" or "Zixuan" as needed
    data_path = f'data/novice/{subject_name}'
    
    # Example: Compare specific pairs
    plot_comparison(data_path, subject_name, 3)
    
    # Or plot in batches
    # for start in range(1, 16, 2):
    #     print(f"Plotting forms {start} to {min(start+1, 15)}...")
    #     plot_point_clouds_batch(data_path, subject_name, start)
