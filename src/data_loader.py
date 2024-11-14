# src/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, file_names):
        self.point_clouds = []
        self.saliency_scores = []
        for file in file_names:
            df = pd.read_csv(os.path.join(data_dir, file))
            points = df[['x', 'y', 'z']].values.astype('float32')
            scores = df['NormalizedScore'].values.astype('float32')
            self.point_clouds.append(points)
            self.saliency_scores.append(scores)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = torch.tensor(self.point_clouds[idx])
        scores = torch.tensor(self.saliency_scores[idx]).reshape(-1, 1)
        
        # Create PyG Data object
        data = Data(
            pos=points,      # [N, 3] Node positions
            x=points,        # [N, 3] Node features (using positions as features)
            y=scores,        # [N, 1] Node-wise scores
        )
        return data

def get_kfold_dataloaders(dataset, k_folds, batch_size):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    dataloaders = []
    for train_idx, val_idx in kfold.split(range(len(dataset))):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = PyGDataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_subset, batch_size=batch_size, shuffle=False)
        dataloaders.append((train_loader, val_loader))
    return dataloaders