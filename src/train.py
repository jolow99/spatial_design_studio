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

def train_model(model, optimizer, train_loader, test_loader, device, num_epochs, checkpoint_dir, config=None):
    best_test_loss = float('inf')
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_mse = 0
        total_train_mae = 0
        total_samples = 0
        
        for i, batch in enumerate(train_loader):
            # Extract metadata for logging
            form_type = batch.metadata['form_type'][0]
            form_number = batch.metadata['form_number'][0]
            print(f"\nTraining on {form_type.capitalize()} Model {form_number}")
            
            optimizer.zero_grad()
            batch = batch.to(device)
            outputs = model(batch)
            
            # Calculate both MSE and MAE
            mse_loss = mean_squared_error(outputs, batch.y)
            mae = mean_absolute_error(outputs, batch.y)
            
            # Use MSE as training loss
            mse_loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_size = batch.x.size(0)
            total_train_mse += mse_loss.item() * batch_size
            total_train_mae += mae.item() * batch_size
            total_samples += batch_size
            
            print(f"Batch MSE: {mse_loss.item():.4f}, MAE: {mae.item():.4f}")
        
        # Testing
        model.eval()
        total_test_mse = 0
        total_test_mae = 0
        test_samples = 0
        
        print("\nEvaluating test models...")
        with torch.no_grad():
            for batch in test_loader:
                form_type = batch.metadata['form_type'][0]
                form_number = batch.metadata['form_number'][0]
                print(f"Testing on {form_type.capitalize()} Model {form_number}")
                
                batch = batch.to(device)
                outputs = model(batch)
                test_mse = mean_squared_error(outputs, batch.y).item()
                test_mae = mean_absolute_error(outputs, batch.y).item()
                
                batch_size = batch.x.size(0)
                total_test_mse += test_mse * batch_size
                total_test_mae += test_mae * batch_size
                test_samples += batch_size
                
                print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
        
        # Calculate averages
        avg_train_mse = total_train_mse / total_samples
        avg_train_mae = total_train_mae / total_samples
        avg_test_mse = total_test_mse / test_samples
        avg_test_mae = total_test_mae / test_samples
        
        print(f"\nEpoch Summary:")
        print(f"Average Training - MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}")
        print(f"Average Test     - MSE: {avg_test_mse:.4f}, MAE: {avg_test_mae:.4f}")
        
        # Save checkpoint for every epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=avg_test_mse,
            checkpoint_dir=checkpoint_dir,
            is_best=(avg_test_mse < best_test_loss),
            config=config
        )
        
        # Update best loss if needed
        if avg_test_mse < best_test_loss:
            best_test_loss = avg_test_mse
    
    return best_test_loss