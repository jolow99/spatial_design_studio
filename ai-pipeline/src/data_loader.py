# src/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from src.geometric_features import compute_geometric_features

def convert_to_classes(normalized_scores, config_type):
    if config_type == 'eeg':
        class_ranges = [
            "Class 0: x ≤ -0.5 (Very negative)",
            "Class 1: -0.5 < x ≤ 0 (Slightly negative)",
            "Class 2: x = 0 (No attention)",
            "Class 3: 0 < x ≤ 0.5 (Slightly positive)",
            "Class 4: x > 0.5 (Very positive)"
        ]
        
        classes = np.zeros_like(normalized_scores, dtype=int)
        classes[normalized_scores <= -0.5] = 0
        classes[(-0.5 < normalized_scores) & (normalized_scores <= 0)] = 1
        classes[normalized_scores == 0] = 2
        classes[(0 < normalized_scores) & (normalized_scores <= 0.5)] = 3
        classes[normalized_scores > 0.5] = 4
    elif config_type == 'et':
        class_ranges = [
            "Class 0: x = 0 (No Attention)",
            "Class 1: 0 < x ≤ 0.025 (Very Low)",
            "Class 2: 0.025 < x ≤ 0.050 (Low)",
            "Class 3: 0.050 < x ≤ 0.1 (Medium)",
            "Class 4: x > 0.1 (High)"
        ]
        
        classes = np.zeros_like(normalized_scores, dtype=int)
        classes[normalized_scores == 0] = 0
        classes[(0 < normalized_scores) & (normalized_scores <= 0.025)] = 1
        classes[(0.025 < normalized_scores) & (normalized_scores <= 0.050)] = 2
        classes[(0.050 < normalized_scores) & (normalized_scores <= 0.1)] = 3
        classes[normalized_scores > 0.1] = 4
    
    return classes

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, subject_type='novice', subject_id='1', config_type='et'):
        self.point_clouds = []
        self.attention_classes = []
        self.geometric_features = []
        self.metadata = []
        
        # Construct path to specific subject directory
        subject_path = os.path.join(data_dir, subject_type, f"subject_{subject_id}", config_type)
        
        if not os.path.exists(subject_path):
            raise ValueError(f"Subject directory not found: {subject_path}")
            
        for file in os.listdir(subject_path):
            if not file.endswith('.csv'):
                continue
            
            # Extract form type and number from filename
            # Example: 'curved1.csv' or 'rect1.csv'
            filename = os.path.splitext(file)[0]  # Remove .csv extension
            if filename.startswith('curved'):
                form_type = 'curved'
                form_number = filename[6:]  # Get number after 'curved'
            elif filename.startswith('rect'):
                form_type = 'rect'
                form_number = filename[4:]  # Get number after 'rect'
            else:
                continue  # Skip files that don't match expected pattern
                
            df = pd.read_csv(os.path.join(subject_path, file))
            points = df[['x', 'y', 'z']].values.astype('float32')
            
            # Compute geometric features
            geometric_feats, normals = compute_geometric_features(points)
            
            # Convert continuous scores to classes
            # Use NormalizedCombinedScore for et_eeg_mult and et_eeg_sum configs
            score_column = 'NormalizedEEGScore' if 'eeg' in config_type else 'NormalizedScore'
            scores = df[score_column].values
            attention_classes = convert_to_classes(scores, config_type)
            
            self.point_clouds.append(points)
            self.attention_classes.append(attention_classes)
            self.geometric_features.append(geometric_feats)
            self.metadata.append({
                'subject_type': subject_type,
                'subject_id': subject_id,
                'config_type': config_type,
                'form_type': form_type,
                'form_number': form_number
            })

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = self.point_clouds[idx]
        features = self.geometric_features[idx]
        classes = self.attention_classes[idx]
        
        # Combine point coordinates with geometric features
        node_features = np.concatenate([points, features], axis=1)
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            y=torch.tensor(classes, dtype=torch.long),
            pos=torch.tensor(points, dtype=torch.float32),  # Original positions
            metadata=self.metadata[idx]
        )
        return data

def get_dataloader(dataset, batch_size=32):
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)