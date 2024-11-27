import torch
import torch.nn as nn

class OrdinalClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.ordinal_layers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes - 1)
        ])
        
    def forward(self, x):
        # Predict cumulative probabilities
        cumulative_probs = torch.sigmoid(
            torch.cat([layer(x) for layer in self.ordinal_layers], dim=1)
        )
        # Convert to class probabilities
        probs = torch.zeros(x.size(0), len(self.ordinal_layers) + 1, 
                          device=x.device)
        probs[:, 0] = 1 - cumulative_probs[:, 0]
        probs[:, 1:-1] = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
        probs[:, -1] = cumulative_probs[:, -1]
        return probs

def ordinal_focal_loss(pred, target, gamma=2.0, alpha=None):
    """
    Ordinal Focal Loss that considers the distance between predicted and true classes.
    """
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(pred, dim=1)
    
    # Create distance matrix between classes
    num_classes = pred.size(1)
    class_distances = torch.abs(
        torch.arange(num_classes, device=pred.device).unsqueeze(0) - 
        torch.arange(num_classes, device=pred.device).unsqueeze(1)
    )
    
    # Calculate loss with ordinal penalties
    target_onehot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
    focal_weight = (1 - probs) ** gamma
    
    # Add distance-based penalties
    ordinal_penalty = class_distances[target, :].unsqueeze(0)
    loss = -torch.log(probs + 1e-8) * focal_weight * (1 + ordinal_penalty)
    
    if alpha is not None:
        alpha = torch.tensor(alpha, device=pred.device)
        class_weights = alpha.gather(0, target)
        loss = loss * class_weights.unsqueeze(1)
    
    return loss.mean() 