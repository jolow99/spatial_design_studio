# src/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, demographic='novice'):
        self.point_clouds = []
        self.saliency_scores = []
        self.metadata = []
        
        # Get all subjects for this demographic
        demographic_dir = os.path.join(data_dir, demographic)
        for subject_dir in os.listdir(demographic_dir):
            subject_path = os.path.join(demographic_dir, subject_dir)
            
            for file in os.listdir(subject_path):
                if not file.endswith('_score.csv'):
                    continue
                    
                df = pd.read_csv(os.path.join(subject_path, file))
                points = df[['x', 'y', 'z']].values.astype('float32')
                scores = df['NormalizedScore'].values.astype('float32')
                
                self.point_clouds.append(points)
                self.saliency_scores.append(scores)
                self.metadata.append({
                    'subject': subject_dir,
                    'form_type': 'curved' if 'curved' in file else 'rect',
                    'form_number': int(file.split('_')[0].replace('curved', '').replace('rect', ''))
                })

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = torch.tensor(self.point_clouds[idx])
        scores = torch.tensor(self.saliency_scores[idx]).reshape(-1, 1)
        
        data = Data(
            x=points,        # [N, 3] Node features
            y=scores,        # [N, 1] Node-wise scores
            metadata=self.metadata[idx]  # Store metadata for validation
        )
        return data

def get_train_test_dataloaders(dataset, test_models=[1, 15], batch_size=32):
    """
    Replace k-fold with architectural form-based split
    """
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