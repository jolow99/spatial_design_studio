# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
import torch_geometric.nn

class DGCNN(nn.Module):
    def __init__(self, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ), k=k)
        
        self.conv2 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ), k=k)
        
        self.conv3 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ), k=k)
        
        self.conv4 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        ), k=k)
        
        self.fc1 = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.fc3 = nn.Linear(256, 1)

    def forward(self, data):
        x, batch = data.x, data.batch  # Assuming data.x has shape [N, 3]
        
        # Apply convolutions
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        
        # Concatenate all features
        x = torch.cat((x1, x2, x3, x4), dim=1)  # [N, C]
        
        # Remove global pooling and directly apply FC layers to each point
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # Shape: [N, 1] where N is total number of points
        x = torch.sigmoid(x)  # Add sigmoid activation
        return x