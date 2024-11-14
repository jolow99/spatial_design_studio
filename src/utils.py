# src/utils.py
import torch.nn as nn
import torch
import os

def mean_squared_error(pred, target):
    return nn.MSELoss()(pred, target)

def mean_absolute_error(pred, target):
    return nn.L1Loss()(pred, target)

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    # Save the latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
    
    # If this is the best model, save it separately
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']