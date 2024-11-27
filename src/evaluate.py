# src/evaluate.py
import torch
from src.utils import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from src.ordinal import ordinal_focal_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = ordinal_focal_loss(outputs, batch.y)
            total_loss += loss.item()
            
            # Get predicted classes
            pred_classes = outputs.argmax(dim=1)
            
            all_preds.extend(pred_classes.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
    
    # Calculate mean absolute error to account for ordinal nature
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return total_loss / len(dataloader), mae