# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
import torch_geometric.nn
from src.ordinal import OrdinalClassificationHead

class ModifiedDGCNN(torch.nn.Module):
    def __init__(self, num_classes, k=20):
        super().__init__()
        
        # DGCNN path for spatial features (xyz)
        self.k = k
        self.spatial_conv1 = DynamicEdgeConv(nn.Sequential(nn.Linear(6, 64)), k=k, aggr='max')    # input: xyz only
        self.spatial_conv2 = DynamicEdgeConv(nn.Sequential(nn.Linear(128, 128)), k=k)
        self.spatial_conv3 = DynamicEdgeConv(nn.Sequential(nn.Linear(256, 256)), k=k)
        
        # Enhanced geometric feature processing
        self.geom_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Separate attention branches for different geometric aspects
        self.local_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        self.global_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(512, 256),  # 448 (DGCNN) + 64 (geom)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Replace the final linear layer with OrdinalClassificationHead
        self.ordinal_head = OrdinalClassificationHead(
            in_features=num_classes,  # Changed from 1024 to num_classes
            num_classes=num_classes
        )

    def forward(self, data):
        # Split features
        xyz = data.x[:, :3]  # Spatial coordinates
        geom = data.x[:, 3:] # Geometric features
        batch = data.batch
        
        # DGCNN path (spatial learning)
        s1 = self.spatial_conv1(xyz, batch)
        s2 = self.spatial_conv2(s1, batch)
        s3 = self.spatial_conv3(s2, batch)
        spatial_features = torch.cat([s1, s2, s3], dim=1)  # 448 features
        
        # Enhanced geometric processing
        geom_features = self.geom_encoder(geom)
        local_weights = self.local_attention(geom_features)
        global_weights = self.global_attention(torch_geometric.nn.global_mean_pool(geom_features, batch))
        
        # Combine local and global geometric attention
        attended_geom = geom_features * (local_weights + global_weights[batch])
        
        # Combine both paths
        combined_features = torch.cat([spatial_features, attended_geom], dim=1)
        
        # Final classification
        x = self.fusion_layer(combined_features)
        x = self.ordinal_head(x)
        return x