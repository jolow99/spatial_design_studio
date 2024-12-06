# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import PointCloudDataset, get_dataloader
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

    # Get dataloader for the full dataset
    dataloader = get_dataloader(
        dataset,
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
    
    best_loss = train_model(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        num_epochs=config['training']['epochs'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        config=config
    )
    
    print(f"Training complete! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()