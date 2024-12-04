# src/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from src.geometric_features import compute_geometric_features

def convert_to_classes(normalized_scores):
    bins = [0, 0.001, 0.02, 0.045, 0.1]
    classes = np.digitize(normalized_scores, bins) - 1
    # Handle the zero case separately
    classes[normalized_scores == 0] = 0
    return classes

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, demographic='novice', subject='subject1'):
        self.point_clouds = []
        self.attention_classes = []
        self.geometric_features = []
        self.metadata = []
        
        # Construct path to specific subject directory
        subject_path = os.path.join(data_dir, demographic, subject)
        
        if not os.path.exists(subject_path):
            raise ValueError(f"Subject directory not found: {subject_path}")
            
        for file in os.listdir(subject_path):
            if not file.endswith('score.csv'):
                continue
                
            df = pd.read_csv(os.path.join(subject_path, file))
            points = df[['x', 'y', 'z']].values.astype('float32')
            
            # Compute geometric features
            geometric_feats, normals = compute_geometric_features(points)
            
            # Convert continuous scores to classes
            scores = df['NormalizedScore'].values
            attention_classes = convert_to_classes(scores)
            
            form_info = file.replace(f"{subject}_", "").replace("score.csv", "")
            form_type = 'curved' if 'curved' in form_info else 'rect'
            form_number = int(''.join(filter(str.isdigit, form_info)))
            
            self.point_clouds.append(points)
            self.attention_classes.append(attention_classes)
            self.geometric_features.append(geometric_feats)
            self.metadata.append({
                'subject': subject,
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

def get_train_test_dataloaders(dataset, test_models=[2, 14], batch_size=32):
    train_indices = []
    test_indices = []
    
    for idx, data in enumerate(dataset):
        form_number = dataset.metadata[idx]['form_number']
        if form_number in test_models:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = PyGDataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_subset, batch_size=batch_size)
    
    return train_loader, test_loader