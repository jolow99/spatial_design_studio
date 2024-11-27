# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import PointCloudDataset, get_train_test_dataloaders
from src.models.dgcnn import ModifiedDGCNN
from src.train import train_model
from src.utils import load_config, load_checkpoint
import yaml
import os

def main():
    # Load configuration
    config = load_config("configs/config.yaml")
    print("loaded config")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with subject and demographic from config
    dataset = PointCloudDataset(
        data_dir=config['data']['path'],
        demographic=config['data']['demographic'],
        subject=config['data']['subject']
    )

    
    # Get train/test loaders
    train_loader, test_loader = get_train_test_dataloaders(
        dataset,
        test_models=[1, 15],  # Models to hold out for testing
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model and optimizer
    model = ModifiedDGCNN(
        num_classes=5,
        k=config['model']['k']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Train model
    best_loss = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=config['training']['epochs'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        config=config
    )
    
    print(f"Training complete! Best test loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()