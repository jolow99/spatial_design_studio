import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def analyze_subject_scores(subject_path, config_type='et', all_subjects_data=None):
    """
    Analyze score distributions for a specific subject and collect data for cross-subject comparison
    """
    # Determine which folder to look in based on config_type
    data_folder = os.path.join(subject_path, config_type)
    if not os.path.exists(data_folder):
        print(f"Error: Folder {data_folder} does not exist")
        return
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # Determine which score column to use
    score_column = 'NormalizedEEGScore' if config_type == 'eeg' else 'NormalizedScore'
    
    # Create figure for histograms
    n_models = len(csv_files)
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig_hist = plt.figure(figsize=(20, 4*n_rows))
    fig_hist.suptitle(f'Score Distributions for Each Model\nSubject: {os.path.basename(subject_path)}, Type: {config_type}')
    
    # Create figure for box plot
    fig_box = plt.figure(figsize=(15, 6))
    all_scores = []
    model_names = []
    
    # Initialize data collection for this subject
    subject_data = defaultdict(list)
    subject_name = os.path.basename(subject_path)
    
    # Process each file
    for idx, file in enumerate(sorted(csv_files), 1):
        df = pd.read_csv(os.path.join(data_folder, file))
        scores = df[score_column].values
        
        # Convert scores to classes using the same ranges as in data_loader.py
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
        
        # Collect class distribution data
        unique, counts = np.unique(classes, return_counts=True)
        distribution = np.zeros(5)  # Initialize with zeros for all 5 classes
        distribution[unique] = counts
        subject_data['distributions'].append(distribution)
        subject_data['model_names'].append(file.split('.')[0])
        
        # Add to collection for box plot
        all_scores.extend(scores)
        model_names.extend([file.split('.')[0]] * len(scores))
        
        # Create histogram
        ax = fig_hist.add_subplot(n_rows, n_cols, idx)
        sns.histplot(data=scores[scores != 0], bins=20, ax=ax)  # Exclude zeros
        ax.set_title(f'Model: {file.split(".")[0]}')
        ax.set_xlabel('Normalized Score')
        ax.set_ylabel('Count')
        
        # Print basic statistics
        print(f"\nStatistics for {file}:")
        print(f"Mean score (excluding zeros): {np.mean(scores[scores != 0]):.4f}")
        print(f"Median score (excluding zeros): {np.median(scores[scores != 0]):.4f}")
        print(f"Std dev (excluding zeros): {np.std(scores[scores != 0]):.4f}")
        print(f"Percentage of non-zero scores: {(np.count_nonzero(scores)/len(scores))*100:.2f}%")
    
    # Create box plot
    plt.figure(fig_box.number)
    df_box = pd.DataFrame({'Model': model_names, 'Score': all_scores})
    df_box = df_box[df_box['Score'] != 0]  # Exclude zeros for better visualization
    sns.boxplot(x='Model', y='Score', data=df_box)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Score Distribution Across Models\nSubject: {os.path.basename(subject_path)}, Type: {config_type}')
    plt.tight_layout()

    # Save figures
    output_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    fig_hist.savefig(os.path.join(output_dir, f'{os.path.basename(subject_path)}_{config_type}_histograms.png'))
    fig_box.savefig(os.path.join(output_dir, f'{os.path.basename(subject_path)}_{config_type}_boxplot.png'))
    plt.close('all')

    if all_subjects_data is not None:
        all_subjects_data[subject_name] = subject_data

def plot_subject_comparisons(all_subjects_data, config_type, demographic):
    """
    Create comparison plots across all subjects
    """
    # Set up class labels based on config type
    if config_type == 'eeg':
        class_names = ['Very negative', 'Slightly negative', 'No attention', 'Slightly positive', 'Very positive']
    else:  # et
        class_names = ['No attention', 'Very Low', 'Low', 'Medium', 'High']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{demographic.capitalize()} Subject Comparisons - {config_type.upper()} Data')
    
    # Prepare data for plotting
    subjects = sorted(list(all_subjects_data.keys()))  # Sort to ensure subject_1 is first
    avg_distributions = []
    
    for subject in subjects:
        distributions = np.array(all_subjects_data[subject]['distributions'])
        avg_dist = distributions.mean(axis=0)
        avg_distributions.append(avg_dist)
    
    # Convert to numpy array for easier manipulation
    avg_distributions = np.array(avg_distributions)
    
    # Set up bar positions
    x = np.arange(len(subjects))
    width = 0.15  # Width of bars
    
    # Create bars for each class
    for i in range(5):
        bars = ax.bar(x + i*width, avg_distributions[:, i], width, 
                     label=class_names[i])
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', rotation=0)
    
    ax.set_title('Class Distribution by Subject')
    ax.set_xlabel('Subject')
    ax.set_ylabel('Count')
    ax.set_xticks(x + width * 2)  # Center the subject labels
    ax.set_xticklabels(subjects)
    ax.legend()
    
    plt.tight_layout()
    
    # Save comparison plot with demographic in filename
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, f'{demographic}_subject_comparisons_{config_type}.png'))
    plt.close()

def main(demographic='novice'):
    """
    Analyze data for all subjects in a demographic group
    """
    if demographic not in ['novice', 'expert']:
        raise ValueError("demographic must be either 'novice' or 'expert'")
    
    # Get all subject directories
    base_path = os.path.join('data', demographic)
    subject_dirs = [d for d in os.listdir(base_path) if d.startswith('subject_')]
    
    # Collect data for all subjects
    all_subjects_data_et = {}
    all_subjects_data_eeg = {}
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(base_path, subject_dir)
        print(f"\nAnalyzing {demographic} {subject_dir}...")
        
        analyze_subject_scores(subject_path, 'et', all_subjects_data_et)
        analyze_subject_scores(subject_path, 'eeg', all_subjects_data_eeg)
    
    # Create comparison plots
    plot_subject_comparisons(all_subjects_data_et, 'et', demographic)
    plot_subject_comparisons(all_subjects_data_eeg, 'eeg', demographic)

if __name__ == "__main__":
    main(demographic='expert')
