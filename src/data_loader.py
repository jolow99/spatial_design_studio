# src/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np

def convert_to_classes(normalized_scores):
    bins = [0, 0.25, 0.50, 0.75, 1.0]
    classes = np.digitize(normalized_scores, bins) - 1
    # Handle the zero case separately
    classes[normalized_scores == 0] = 0
    return classes

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, demographic='novice', subject='subject1'):
        self.point_clouds = []
        self.attention_classes = []
        self.metadata = []
        
        # Construct path to specific subject directory
        subject_path = os.path.join(data_dir, demographic, subject)
        
        if not os.path.exists(subject_path):
            raise ValueError(f"Subject directory not found: {subject_path}")
            
        for file in os.listdir(subject_path):
            if not file.endswith('_score.csv'):
                continue
                
            df = pd.read_csv(os.path.join(subject_path, file))
            points = df[['x', 'y', 'z']].values.astype('float32')
            
            # Convert continuous scores to classes
            scores = df['NormalizedScore'].values
            attention_classes = convert_to_classes(scores)
            
            self.point_clouds.append(points)
            self.attention_classes.append(attention_classes)
            self.metadata.append({
                'subject': subject,
                'form_type': 'curved' if 'curved' in file else 'rect',
                'form_number': int(file.split('_')[0].replace('curved', '').replace('rect', ''))
            })

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = torch.tensor(self.point_clouds[idx])
        classes = torch.tensor(self.attention_classes[idx], dtype=torch.long)
        
        data = Data(
            x=points,        # [N, 3] Node features
            y=classes,       # [N] Node-wise class labels
            metadata=self.metadata[idx]
        )
        return data

def get_train_test_dataloaders(dataset, test_models=[1, 15], batch_size=32):
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