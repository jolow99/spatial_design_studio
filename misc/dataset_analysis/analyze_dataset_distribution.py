import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def analyze_subject_scores(subject_path, config_type='et', demographic=None):
    """
    Analyze score distributions for a specific subject and collect data for cross-subject comparison
    """
    # Determine which folder to look in based on config_type
    data_folder = os.path.join(subject_path, config_type)
    if not os.path.exists(data_folder):
        print(f"Error: Folder {data_folder} does not exist")
        return
    
    # Get all CSV files (curved1.csv to curved15.csv and rect1.csv to rect15.csv)
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and (f.startswith('curved') or f.startswith('rect'))]
    
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
        
        # Update class definitions based on provided ranges
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

    # Save figures to the new directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Include demographic in filenames
    demographic_str = f"{demographic}_" if demographic else ""
    fig_hist.savefig(os.path.join(output_dir, f'{demographic_str}{os.path.basename(subject_path)}_{config_type}_histograms.png'),
                     bbox_inches='tight',
                     pad_inches=0.1)
    fig_box.savefig(os.path.join(output_dir, f'{demographic_str}{os.path.basename(subject_path)}_{config_type}_boxplot.png'),
                    bbox_inches='tight',
                    pad_inches=0.1)
    plt.close('all')
    
    return subject_data  # Return the collected data

def compare_all_subjects(all_subjects_data, config_type):
    """
    Create a comparison plot for all subjects in one visualization
    """
    if not all_subjects_data:
        print(f"No data available for {config_type}")
        return
        
    # Define output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    subjects = sorted(list(all_subjects_data.keys()))
    if not subjects:
        print(f"No subjects found for {config_type}")
        return
        
    avg_distributions = []

    for subject in subjects:
        if len(all_subjects_data[subject]['distributions']) > 0:
            distributions = np.array(all_subjects_data[subject]['distributions'])
            avg_dist = distributions.mean(axis=0)
            avg_distributions.append(avg_dist)
        else:
            avg_distributions.append(np.zeros(5))  # Ensure this is a 1D array of zeros

    avg_distributions = np.array(avg_distributions)

    # Ensure avg_distributions is 2D
    if avg_distributions.ndim == 1:
        avg_distributions = avg_distributions[np.newaxis, :]  # Convert to 2D

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'All Subjects Comparison - {config_type.upper()} Data')

    # Set up bar positions
    x = np.arange(len(subjects))
    width = 0.15

    # Create bars for each class
    for i in range(5):
        bars = ax.bar(x + i * width, avg_distributions[:, i], width, label=f'Class {i}')

        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', rotation=0)

    ax.set_title('Class Distribution by Subject')
    ax.set_xlabel('Subject')
    ax.set_ylabel('Count')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(subjects)
    ax.legend()
    plt.tight_layout()

    # Save comparison plot
    plt.savefig(os.path.join(output_dir, f'all_subjects_comparison_{config_type}.png'))
    plt.close()

def compare_demographics(all_subjects_data_novice, all_subjects_data_expert, config_type):
    """
    Create a comparison plot between demographics
    """
    # Define output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    avg_distributions_novice = []
    avg_distributions_expert = []

    for subject in sorted(all_subjects_data_novice.keys()):
        distributions = np.array(all_subjects_data_novice[subject]['distributions'])
        avg_distributions_novice.append(distributions.mean(axis=0))

    for subject in sorted(all_subjects_data_expert.keys()):
        distributions = np.array(all_subjects_data_expert[subject]['distributions'])
        avg_distributions_expert.append(distributions.mean(axis=0))

    avg_distributions_novice = np.array(avg_distributions_novice)
    avg_distributions_expert = np.array(avg_distributions_expert)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Demographic Comparison - {config_type.upper()} Data')

    # Set up bar positions
    x = np.arange(5)  # 5 classes
    width = 0.35  # Width of bars

    # Create bars for novice and expert
    bars_novice = ax.bar(x - width/2, avg_distributions_novice.mean(axis=0), width, label='Novice')
    bars_expert = ax.bar(x + width/2, avg_distributions_expert.mean(axis=0), width, label='Expert')

    # Add value labels on the bars
    for bar in bars_novice:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=0)

    for bar in bars_expert:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=0)

    ax.set_title('Class Distribution by Demographic')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in range(5)])
    ax.legend()
    plt.tight_layout()

    # Save demographic comparison plot
    plt.savefig(os.path.join(output_dir, f'demographic_comparison_{config_type}.png'))
    plt.close()

def main():
    """
    Analyze data for all subjects in both demographic groups
    """
    demographics = ['novice', 'expert']

    # Initialize separate dictionaries for each demographic
    all_subjects_data_et = {'novice': {}, 'expert': {}}
    all_subjects_data_eeg = {'novice': {}, 'expert': {}}

    # Analyze subjects for both demographics
    for demographic in demographics:
        base_path = os.path.join('data', demographic)
        subject_dirs = [d for d in os.listdir(base_path) if d.startswith('subject_')]

        for subject_dir in subject_dirs:
            subject_path = os.path.join(base_path, subject_dir)
            print(f"\nAnalyzing {demographic} {subject_dir}...")

            # Collect data for both 'et' and 'eeg'
            et_data = analyze_subject_scores(subject_path, 'et', demographic)
            eeg_data = analyze_subject_scores(subject_path, 'eeg', demographic)
            
            # Store the data with demographic separation
            if et_data:
                all_subjects_data_et[demographic][subject_dir] = et_data
            if eeg_data:
                all_subjects_data_eeg[demographic][subject_dir] = eeg_data

    # Create comparison plots for each demographic separately
    for demographic in demographics:
        compare_all_subjects(all_subjects_data_et[demographic], f'et_{demographic}')
        compare_all_subjects(all_subjects_data_eeg[demographic], f'eeg_{demographic}')

    # Compare between demographics
    compare_demographics(
        all_subjects_data_et['novice'],
        all_subjects_data_et['expert'],
        'et'
    )
    compare_demographics(
        all_subjects_data_eeg['novice'],
        all_subjects_data_eeg['expert'],
        'eeg'
    )

if __name__ == "__main__":
    main()
