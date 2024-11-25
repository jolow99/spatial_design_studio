# src/train.py
import torch
from src.utils import save_checkpoint
import torch.nn as nn

def train_model(model, optimizer, train_loader, test_loader, device, num_epochs, checkpoint_dir, config=None):
    best_test_loss = float('inf')
    criterion = nn.MSELoss()  # Define MSE loss directly
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        total_samples = 0
        
        for data in train_loader:
            # Extract metadata for logging
            form_type = data.metadata['form_type']
            form_number = data.metadata['form_number'][0]
            print(f"\nTraining on {form_type[0].capitalize()} Model {form_number}")
            
            optimizer.zero_grad()
            data = data.to(device)
            outputs = model(data)
            
            # Add debugging prints
            print(f"Outputs min: {outputs.min().item():.6f}, max: {outputs.max().item():.6f}")
            print(f"Target min: {data.y.min().item():.6f}, max: {data.y.max().item():.6f}")
            
            # Calculate MSE loss
            loss = criterion(outputs, data.y)
            
            # Add gradient debugging
            loss.backward()
            total_grad = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().sum()
            print(f"Total gradient magnitude: {total_grad:.6f}")
            
            optimizer.step()
            
            batch_size = data.x.size(0)
            total_train_loss += loss.item() * batch_size
            total_samples += batch_size
            
            print(f"Loss: {loss.item():.4f}")
        
        # Testing
        model.eval()
        total_test_loss = 0
        test_samples = 0
        
        print("\nEvaluating test models...")
        with torch.no_grad():
            for batch in test_loader:
                form_type = batch.metadata['form_type']
                form_number = batch.metadata['form_number'][0]
                print(f"Testing on {form_type[0].capitalize()} Model {form_number}")
                
                batch = batch.to(device)
                outputs = model(batch)
                test_loss = criterion(outputs, batch.y).item()
                
                batch_size = batch.x.size(0)
                total_test_loss += test_loss * batch_size
                test_samples += batch_size
                
                print(f"Test Loss: {test_loss:.4f}")
        
        # Calculate averages
        avg_train_loss = total_train_loss / total_samples
        avg_test_loss = total_test_loss / test_samples
        
        print(f"\nEpoch Summary:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        
        # Save checkpoint every epoch
        is_best = avg_test_loss < best_test_loss
        if is_best:
            best_test_loss = avg_test_loss
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=avg_test_loss,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best,
            config=config
        )
    
    return best_test_loss