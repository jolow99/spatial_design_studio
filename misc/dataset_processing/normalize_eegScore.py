import os
import pandas as pd

def normalize_eeg_score(subject_folder):
    # Define the input and output directories
    input_dir = os.path.join(subject_folder, 'et_eeg_mult')
    output_dir = os.path.join(subject_folder, 'eeg')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: The directory {input_dir} does not exist.")
        return

    # Iterate through all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(input_dir, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Find min and max values, excluding zeros
            min_eeg = df[df['EEGScore'] < 0]['EEGScore'].min()  # minimum negative value
            max_eeg = df[df['EEGScore'] > 0]['EEGScore'].max()  # maximum positive value
            
            # Create masks for zero, positive and negative values
            zero_mask = df['EEGScore'] == 0
            positive_mask = df['EEGScore'] > 0
            negative_mask = df['EEGScore'] < 0
            
            # Normalize positive values (0 to max → 0 to 1)
            df.loc[positive_mask, 'NormalizedEEGScore'] = df.loc[positive_mask, 'EEGScore'] / max_eeg
            
            # Normalize negative values (min to 0 → -1 to 0)
            df.loc[negative_mask, 'NormalizedEEGScore'] = df.loc[negative_mask, 'EEGScore'] / abs(min_eeg)
            
            # Keep zeros as zeros
            df.loc[zero_mask, 'NormalizedEEGScore'] = 0
            
            print(f"Processing: Min EEGScore = {min_eeg}, Max EEGScore = {max_eeg}")
            
            # Select the required columns
            df = df[['x', 'y', 'z', 'EEGScore', 'NormalizedEEGScore']]
            
            # Save the new CSV file in the output directory
            output_file_path = os.path.join(output_dir, filename)
            df.to_csv(output_file_path, index=False)

# Example usage
normalize_eeg_score('data/expert/subject_3')
