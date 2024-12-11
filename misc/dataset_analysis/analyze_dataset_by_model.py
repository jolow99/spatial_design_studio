import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import numpy as np

def plot_model_comparison(data_dir, model_name, config_type='et'):
    """
    Plot ground truth data for a specific model across all subjects
    """
    # Set up colors and class names based on config type
    if config_type == 'et':
        colors = ['lightgray', 'green', 'yellow', 'orange', 'red']
        class_names = ['No attention', 'Very Low', 'Low', 'Medium', 'High']
    else:  # eeg
        colors = ['red', 'orange', 'lightgray', 'yellowgreen', 'green']
        class_names = ['Very negative', 'Slightly negative', 'No attention', 'Slightly positive', 'Very positive']

    # Get all subject directories
    demographics = ['novice', 'expert']
    all_subjects = []
    
    # First add all novice subjects (sorted)
    novice_dirs = [d for d in os.listdir(os.path.join(data_dir, 'novice')) 
                   if os.path.isdir(os.path.join(data_dir, 'novice', d))]
    novice_dirs.sort(key=lambda x: int(x.split('_')[1]))
    for subject_dir in novice_dirs:
        all_subjects.append(('novice', subject_dir))
    
    # Then add all expert subjects (sorted)
    expert_dirs = [d for d in os.listdir(os.path.join(data_dir, 'expert')) 
                   if os.path.isdir(os.path.join(data_dir, 'expert', d))]
    expert_dirs.sort(key=lambda x: int(x.split('_')[1]))
    for subject_dir in expert_dirs:
        all_subjects.append(('expert', subject_dir))

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Model {model_name} - {config_type.upper()} Ground Truth Comparison', 
                 fontsize=16, y=0.95)  # Removed 'pad' parameter

    # Create subplot for each subject
    for idx, (demo, subject) in enumerate(all_subjects, 1):
        # Construct file path
        file_path = os.path.join(data_dir, demo, subject, config_type, f"{model_name}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        # Read data
        df = pd.read_csv(file_path)
        points = df[['x', 'y', 'z']].values
        
        # Get scores and convert to classes
        score_column = 'NormalizedEEGScore' if config_type == 'eeg' else 'NormalizedScore'
        scores = df[score_column].values
        
        if config_type == 'eeg':
            classes = np.zeros_like(scores, dtype=int)
            classes[scores <= -0.5] = 0
            classes[(-0.5 < scores) & (scores <= 0)] = 1
            classes[scores == 0] = 2
            classes[(0 < scores) & (scores <= 0.5)] = 3
            classes[scores > 0.5] = 4
        else:  # et
            classes = np.zeros_like(scores, dtype=int)
            classes[scores == 0] = 0
            classes[(0 < scores) & (scores <= 0.025)] = 1
            classes[(0.025 < scores) & (scores <= 0.050)] = 2
            classes[(0.050 < scores) & (scores <= 0.1)] = 3
            classes[scores > 0.1] = 4

        # Create subplot (will naturally place novice 1-3 on top, expert 1-2 on bottom)
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # Set point colors and sizes
        point_colors = [colors[int(c)] for c in classes]
        if config_type == 'et':
            point_sizes = [2 if c == 0 else 8 for c in classes]
            point_alphas = [0.4 if c == 0 else 1.0 for c in classes]
        else:  # eeg
            point_sizes = [8 for _ in classes]
            point_alphas = [1.0 for _ in classes]

        # Plot points
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=point_colors,
                           s=point_sizes,
                           alpha=point_alphas)

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=c, label=class_names[i],
                                    markersize=10,
                                    alpha=1.0 if (config_type == 'eeg' or i > 0) else 0.4)
                         for i, c in enumerate(colors)]
        ax.legend(handles=legend_elements)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{demo.capitalize()} - {subject}')
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    
    # Save visualization
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'model_comparison_{model_name}_{config_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")

def main():
    data_dir = 'data'
    
    # Process all models for both ET and EEG data
    for config_type in ['et', 'eeg']:
        print(f"\nProcessing {config_type.upper()} data...")
        
        # Process curved and rect models
        for form_type in ['curved', 'rect']:
            for i in range(1, 16):
                model_name = f"{form_type}{i}"
                print(f"Processing model: {model_name}")
                plot_model_comparison(data_dir, model_name, config_type)

if __name__ == "__main__":
    main()
