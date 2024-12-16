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
    print("\nConfiguration loaded:")
    print(f"Data Settings:")
    print(f"  • Path: {config['data']['path']}")
    print(f"  • Subject Type: {config['data']['subject_type']}")
    print(f"  • Subject ID: {config['data']['subject_id']}")
    print(f"  • Config Type: {config['data']['config_type']}")
    print(f"\nModel Settings:")
    print(f"  • k-nearest neighbors: {config['model']['k']}")
    print(f"  • Dropout Rate: {config['model']['dropout']}")
    print(f"\nTraining Settings:")
    print(f"  • Epochs: {config['training']['epochs']}")
    print(f"  • Batch Size: {config['training']['batch_size']}")
    print(f"  • Learning Rate: {config['training']['learning_rate']}")
    print(f"  • Checkpoint Directory: {config['training']['checkpoint_dir']}")
    print("\n" + "="*50 + "\n")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with updated parameters
    dataset = PointCloudDataset(
        data_dir=config['data']['path'],
        subject_type=config['data']['subject_type'],
        subject_id=config['data']['subject_id'],
        config_type=config['data']['config_type']
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