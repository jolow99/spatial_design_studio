import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import math

def plot_point_clouds_batch(data_dir, subject_name, start_idx, batch_size=2):
    """
    Plot a batch of point clouds from the data directory
    """
    # Get batch of files with new naming pattern
    curved_files = [f"{subject_name}_curved{i}score.csv" for i in range(start_idx, min(start_idx + batch_size, 16))]
    rect_files = [f"{subject_name}_rect{i}score.csv" for i in range(start_idx, min(start_idx + batch_size, 16))]
    all_files = curved_files + rect_files
    
    # Calculate grid dimensions
    n_files = len(all_files)
    n_cols = 2  # 2 columns (curved and rect side by side)
    n_rows = math.ceil(n_files / 2)
    
    # Create figure with larger size
    fig = plt.figure(figsize=(15, 6*n_rows))
    
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
                           cmap='plasma',
                           marker='o',
                           s=5)  # Increased point size
        
        # Set labels and title
        ax.set_xlabel('X', labelpad=10)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=10)
        
        # Extract form number and type from filename
        if 'curved' in file:
            form_num = file.split('curved')[1].split('score')[0]
            title = f'Curved Form #{form_num}'
        else:
            form_num = file.split('rect')[1].split('score')[0]
            title = f'Rect Form #{form_num}'
            
        ax.set_title(title, pad=20, fontsize=12)
        
        # Add grid lines
        ax.grid(True)
        
        # Set consistent view angle
        ax.view_init(elev=20, azim=45)
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, ax=ax, label='Saliency Score')
        cbar.ax.set_ylabel('Saliency Score', rotation=270, labelpad=25)
    
    # Adjust layout with more space
    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.show()

def plot_comparison(data_dir, subject_name, form_number):
    """
    Plot specific curvilinear and rectilinear forms side by side
    """
    # Construct filenames with new pattern
    curved_file = f'{subject_name}_curved{form_number}score.csv'
    rect_file = f'{subject_name}_rect{form_number}score.csv'
    
    # Create figure with larger size
    fig = plt.figure(figsize=(20, 8))
    
    # Plot curvilinear form
    ax1 = fig.add_subplot(121, projection='3d')
    df1 = pd.read_csv(os.path.join(data_dir, curved_file))
    scatter1 = ax1.scatter(df1['x'], df1['y'], df1['z'], 
                          c=df1['NormalizedScore'], 
                          cmap='plasma',
                          marker='o',
                          s=5)  # Increased point size
    ax1.set_xlabel('X', labelpad=10)
    ax1.set_ylabel('Y', labelpad=10)
    ax1.set_zlabel('Z', labelpad=10)
    ax1.set_title(f'Curved Form #{form_number}', pad=20, fontsize=14)
    ax1.grid(True)
    ax1.view_init(elev=20, azim=45)
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Saliency Score')
    cbar1.ax.set_ylabel('Saliency Score', rotation=270, labelpad=25)
    
    # Plot rectilinear form
    ax2 = fig.add_subplot(122, projection='3d')
    df2 = pd.read_csv(os.path.join(data_dir, rect_file))
    scatter2 = ax2.scatter(df2['x'], df2['y'], df2['z'], 
                          c=df2['NormalizedScore'], 
                          cmap='plasma',
                          marker='o',
                          s=5)  # Increased point size
    ax2.set_xlabel('X', labelpad=10)
    ax2.set_ylabel('Y', labelpad=10)
    ax2.set_zlabel('Z', labelpad=10)
    ax2.set_title(f'Rect Form #{form_number}', pad=20, fontsize=14)
    ax2.grid(True)
    ax2.view_init(elev=20, azim=45)
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Saliency Score')
    cbar2.ax.set_ylabel('Saliency Score', rotation=270, labelpad=25)
    
    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.show()

if __name__ == "__main__":
    subject_name = "Abram"  # Change this to "Sean" or "Zixuan" as needed
    data_path = f'data/novice/{subject_name}'
    
    # Example: Compare specific pairs
    # plot_comparison(data_path, subject_name, 1)  # Compare first pair of forms
    
    # Or plot in batches
    for start in range(1, 16, 2):
        print(f"Plotting forms {start} to {min(start+1, 15)}...")
        plot_point_clouds_batch(data_path, subject_name, start)