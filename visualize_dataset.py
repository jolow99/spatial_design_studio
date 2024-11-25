import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import math

def plot_point_clouds_batch(data_dir, start_idx, batch_size=2):
    """
    Plot a batch of point clouds from the data directory
    
    Args:
        data_dir: Path to directory containing CSV files
        start_idx: Starting index (1-15)
        batch_size: Number of pairs to plot at once (default 2)
    """
    # Get batch of files
    curved_files = [f"curved{i}_score.csv" for i in range(start_idx, min(start_idx + batch_size, 16))]
    rect_files = [f"rect{i}_score.csv" for i in range(start_idx, min(start_idx + batch_size, 16))]
    all_files = curved_files + rect_files
    
    # Calculate grid dimensions
    n_files = len(all_files)
    n_cols = 2  # 2 columns (curved and rect side by side)
    n_rows = math.ceil(n_files / 2)
    
    # Create figure
    fig = plt.figure(figsize=(12, 4*n_rows))
    
    # Plot each point cloud
    for idx, file in enumerate(all_files, 1):
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: File {file} not found")
            continue
            
        # Read dataset
        df = pd.read_csv(file_path)
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
        scatter = ax.scatter(df['x'], df['y'], df['z'], 
                           c=df['NormalizedScore'], 
                           cmap='plasma',  # Changed from 'hot' to 'plasma'
                           marker='o',
                           s=2)  # Slightly larger points
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Clean up filename for title
        if 'curved' in file:
            number = file.split('_')[0][6:]  # Extract number after 'curved'
            title = f'Curved Form {number}'
        else:
            number = file.split('_')[0][4:]  # Extract number after 'rect'
            title = f'Rect Form {number}'
        ax.set_title(title)
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, ax=ax, label='Saliency Score')
        cbar.ax.set_ylabel('Saliency Score', rotation=270, labelpad=15)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_comparison(data_dir, form_number):
    """
    Plot specific curvilinear and rectilinear forms side by side
    
    Args:
        data_dir: Path to directory containing CSV files
        form_number: Index number of forms to compare (1-15)
    """
    # Construct filenames
    curved_file = f'curved{form_number}_score.csv'
    rect_file = f'rect{form_number}_score.csv'
    
    # Create figure
    fig = plt.figure(figsize=(15, 6))
    
    # Plot curvilinear form
    ax1 = fig.add_subplot(121, projection='3d')
    df1 = pd.read_csv(os.path.join(data_dir, curved_file))
    scatter1 = ax1.scatter(df1['x'], df1['y'], df1['z'], 
                          c=df1['NormalizedScore'], 
                          cmap='plasma',  # Changed colormap
                          marker='o',
                          s=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Curved Form {form_number}')
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Saliency Score')
    cbar1.ax.set_ylabel('Saliency Score', rotation=270, labelpad=15)
    
    # Plot rectilinear form
    ax2 = fig.add_subplot(122, projection='3d')
    df2 = pd.read_csv(os.path.join(data_dir, rect_file))
    scatter2 = ax2.scatter(df2['x'], df2['y'], df2['z'], 
                          c=df2['NormalizedScore'], 
                          cmap='plasma',  # Changed colormap
                          marker='o',
                          s=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Rect Form {form_number}')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Saliency Score')
    cbar2.ax.set_ylabel('Saliency Score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Plot in batches of 2
    for start in range(1, 16, 2):
        print(f"Plotting forms {start} to {min(start+1, 15)}...")
        plot_point_clouds_batch('data/novice/subject1', start)
    
    # Example: Compare specific pairs
    # plot_comparison('.', 1)  # Compare first pair of forms