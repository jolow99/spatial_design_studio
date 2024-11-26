# src/train.py
import torch
from src.utils import save_checkpoint
import torch.nn as nn
from tqdm import tqdm

def sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    Sigmoid Focal Loss as defined in the paper (equation 5)
    SFL(p,y) = -1/n ∑[αy(1-p)^γ logp + (1-α)(1-y)p^γ log(1-p)]
    
    Args:
        pred: predictions (logits)
        target: ground truth labels
        gamma: focusing parameter to reduce weight of easy examples
        alpha: balancing parameter for class imbalance
    """
    num_classes = pred.size(1)
    one_hot_target = torch.nn.functional.one_hot(target, num_classes=num_classes).float()
    
    # Apply sigmoid to get probabilities (equation 6)
    p = torch.sigmoid(pred)
    
    # Calculate focal loss for positive examples (y=1)
    pos_loss = -alpha * ((1 - p) ** gamma) * torch.log(p + 1e-8)
    
    # Calculate focal loss for negative examples (y=0)
    neg_loss = -(1 - alpha) * (p ** gamma) * torch.log(1 - p + 1e-8)
    
    # Combine losses based on target (equation 5)
    loss = one_hot_target * pos_loss + (1 - one_hot_target) * neg_loss
    
    # Average over all samples and classes
    return loss.mean()

def train_model(model, optimizer, train_loader, test_loader, device, num_epochs, checkpoint_dir, config=None):
    best_test_loss = float('inf')
    
    # Simplified criterion without class weights
    criterion = lambda pred, target: sigmoid_focal_loss(
        pred, 
        target,
        gamma=2.0,
        alpha=0.25
    )
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        total_samples = 0
        
        # Add progress bar
        train_pbar = tqdm(train_loader, desc='Training', total=len(train_loader))
        for data in train_pbar:
            form_type = data.metadata['form_type']
            form_number = data.metadata['form_number'][0]
            train_pbar.set_description(f"Training on {form_type[0].capitalize()} Model {form_number}")
            
            optimizer.zero_grad()
            data = data.to(device)
            outputs = model(data)
            
            # Add debugging prints
            pred_classes = outputs.argmax(dim=1)
            print(f"\nPredicted class distribution: {[int((pred_classes == i).sum()) for i in range(5)]}")
            print(f"Target class distribution: {[int((data.y == i).sum()) for i in range(5)]}")
            
            # Calculate cross entropy loss
            loss = criterion(outputs, data.y)
            
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
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Testing
        model.eval()
        total_test_loss = 0
        test_samples = 0
        
        print("\nEvaluating test models...")
        test_pbar = tqdm(test_loader, desc='Testing')
        with torch.no_grad():
            for batch in test_pbar:
                form_type = batch.metadata['form_type']
                form_number = batch.metadata['form_number'][0]
                test_pbar.set_description(f"Testing on {form_type[0].capitalize()} Model {form_number}")
                
                batch = batch.to(device)
                outputs = model(batch)
                test_loss = criterion(outputs, batch.y).item()
                
                batch_size = batch.x.size(0)
                total_test_loss += test_loss * batch_size
                test_samples += batch_size
                
                test_pbar.set_postfix({'Loss': f'{test_loss:.4f}'})
        
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