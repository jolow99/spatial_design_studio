# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import PointCloudDataset, get_kfold_dataloaders
from src.models.dgcnn import DGCNN
from src.train import train_one_epoch, validate, train_model
from src.evaluate import evaluate_model
from src.utils import load_config, load_checkpoint
import yaml
import os

def main():
    # Load configuration
    config = load_config("configs/config.yaml")
    
    # Device configuration
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load dataset
    dataset = PointCloudDataset(data_dir=config['data']['path'],
                                file_names=config['data']['file_names'])
    
    # Get K-fold dataloaders
    dataloaders = get_kfold_dataloaders(dataset, config['training']['k_folds'], config['training']['batch_size'])
    
    # Initialize model
    model = DGCNN(k=config['model']['k'], dropout=config['model']['dropout']).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    checkpoint_dir = config['training']['checkpoint_dir']
    
    # K-fold Cross Validation
    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        print(f"--- Fold {fold+1} ---")
        
        # Training loop
        best_val_loss = train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=config['training']['epochs'],
            checkpoint_dir=checkpoint_dir
        )
        
        # Optional: Load best model for final evaluation
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            _, _ = load_checkpoint(model, optimizer, best_model_path)
            final_mse, final_mae = evaluate_model(model, val_loader, device)
            print(f"Best Model Performance - MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")
        
        print(f"--- Fold {fold+1} Completed ---\n")
    
if __name__ == "__main__":
    main()