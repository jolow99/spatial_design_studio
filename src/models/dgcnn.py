# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
import torch_geometric.nn

class DGCNN(nn.Module):
    def __init__(self, k=20, dropout=0.5, num_classes=5):
        super(DGCNN, self).__init__()
        self.k = k
        
        # Input features: 3 (xyz) + 6 (geometric features: curvature, density, height, normal_x, normal_y, normal_z)
        input_features = 9
        
        # Attention mechanism for geometric features
        self.attention = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, input_features),
            nn.Sigmoid()
        )
        
        # Edge convolution layers with batch normalization
        self.conv1 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(input_features * 2, 64),  # *2 because EdgeConv concatenates features
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ), k=k)
        
        self.conv2 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(64 * 2, 128),  # *2 because EdgeConv concatenates features
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        ), k=k)
        
        self.conv3 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(128 * 2, 256),  # *2 because EdgeConv concatenates features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        ), k=k)
        
        # Point-wise MLPs for per-point prediction
        self.point_mlp = nn.Sequential(
            nn.Linear(64 + 128 + 256, 512),  # Concatenated features from all conv layers
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        
        # Ensure batch is properly defined
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply attention to input features (both geometric and spatial)
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Edge convolution layers - maintain point-wise features
        x1 = self.conv1(x, batch)      # [num_points, 64]
        x2 = self.conv2(x1, batch)     # [num_points, 128]
        x3 = self.conv3(x2, batch)     # [num_points, 256]
        
        # Concatenate features from all levels for each point
        x = torch.cat([x1, x2, x3], dim=1)  # [num_points, 64+128+256]
        
        # Final point-wise predictions
        x = self.point_mlp(x)  # [num_points, num_classes]
        
        return x