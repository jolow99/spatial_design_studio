# src/train.py
import torch
from tqdm import tqdm
from src.utils import mean_squared_error, mean_absolute_error, save_checkpoint

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        outputs = model(batch.to(device))
        loss = mean_squared_error(outputs, batch.y.to(device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, dataloader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            outputs = model(batch)
            total_mse += mean_squared_error(outputs, batch.y).item()
            total_mae += mean_absolute_error(outputs, batch.y).item()
    return total_mse / len(dataloader), total_mae / len(dataloader)

def train_model(model, optimizer, train_loader, val_loader, device, num_epochs, checkpoint_dir):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        
        # Validation
        val_mse, val_mae = validate(model, val_loader, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}')
        
        # Save checkpoint
        is_best = val_mse < best_val_loss
        if is_best:
            best_val_loss = val_mse
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_mse,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best
        )
    
    return best_val_loss