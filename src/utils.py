# src/utils.py
import torch.nn as nn
import torch
import os
from datetime import datetime
import yaml

def mean_squared_error(pred, target):
    return nn.MSELoss()(pred, target)

def mean_absolute_error(pred, target):
    return nn.L1Loss()(pred, target)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_timestamp_dir(checkpoint_dir):
    """Create a directory name with current date and timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(checkpoint_dir, timestamp)

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, is_best=False, config=None):
    """Save model checkpoint"""
    # Create timestamp directory if it doesn't exist
    if not hasattr(save_checkpoint, 'timestamp_dir'):
        save_checkpoint.timestamp_dir = get_timestamp_dir(checkpoint_dir)
        os.makedirs(save_checkpoint.timestamp_dir, exist_ok=True)
        print(f"Created new checkpoint directory: {save_checkpoint.timestamp_dir}")
        
        # Save config file if provided
        if config is not None:
            config_path = os.path.join(save_checkpoint.timestamp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            print(f"Saved configuration to {config_path}")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    # Save epoch checkpoint
    epoch_path = os.path.join(save_checkpoint.timestamp_dir, f'epoch_{epoch+1}.pt')
    torch.save(checkpoint, epoch_path)
    print(f"Saved epoch checkpoint to {epoch_path}")
    
    # If this is the best model, save it separately
    if is_best:
        best_path = os.path.join(save_checkpoint.timestamp_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model checkpoint to {best_path} with validation loss: {val_loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']