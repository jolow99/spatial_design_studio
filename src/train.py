# src/train.py
import torch
from src.utils import save_checkpoint
import torch.nn as nn
from tqdm import tqdm

def softmax_focal_loss(pred, target, gamma=2.0, alpha=None):
    """
    Softmax Focal Loss for multiclass classification (adapted for mutually exclusive classes).
    
    Args:
        pred: Predictions (logits) with shape [batch_size, num_classes].
        target: Ground truth labels with shape [batch_size].
        gamma: Focusing parameter to reduce weight of easy examples.
        alpha: Optional class balancing weights (list or tensor of shape [num_classes]).
               If None, all classes are treated equally.
    
    Returns:
        Loss value (scalar).
    """
    # Apply softmax to logits to get probabilities
    probs = torch.nn.functional.softmax(pred, dim=1)
    
    # Select the probabilities corresponding to the target class
    target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
    
    # Compute the focal loss modulation factor (1 - p_t)^γ
    focal_weight = (1 - target_probs) ** gamma
    
    # Compute the cross-entropy loss for the target class
    ce_loss = -torch.log(target_probs + 1e-8)  # Shape: [batch_size]

    # Apply alpha (class weighting) if provided
    if alpha is not None:
        alpha = torch.tensor(alpha, device=pred.device)  # Ensure alpha is on the same device
        class_weights = alpha.gather(0, target)  # Get weights for the true classes
        loss = focal_weight * class_weights * ce_loss
    else:
        loss = focal_weight * ce_loss

    # Return the mean loss over the batch
    return loss.mean()

def calculate_class_weights(dataset):
    """
    Calculate class weights based on inverse class frequencies
    
    Args:
        dataset: PyTorch Dataset object containing all training data
        
    Returns:
        List of class weights normalized to have minimum weight of 1.0
    """
    # Initialize class counts
    class_counts = [0] * 5  # 5 classes (0-4)
    total_points = 0
    
    # Count instances of each class
    for data in dataset:
        for class_idx in range(5):
            count = (data.y == class_idx).sum().item()
            class_counts[class_idx] += count
            total_points += count
    
    # Calculate inverse frequencies
    inverse_freqs = [total_points / (count + 1e-8) for count in class_counts]
    
    # Normalize weights so minimum weight is 1.0
    min_weight = min(inverse_freqs)
    normalized_weights = [freq / min_weight for freq in inverse_freqs]
    
    print("\nClass distribution and weights:")
    for i, (count, weight) in enumerate(zip(class_counts, normalized_weights)):
        print(f"Class {i}: {count} samples, weight = {weight:.2f}")
    
    return normalized_weights

def train_model(model, optimizer, train_loader, test_loader, device, num_epochs, checkpoint_dir, config=None):
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Add gradient clipping to handle large gradients
    max_grad_norm = 1.0
    
    best_test_loss = float('inf')
    
    # Calculate class weights from training data
    alpha = calculate_class_weights(train_loader.dataset.dataset)  # Access underlying dataset through Subset
    
    criterion = lambda pred, target: softmax_focal_loss(
        pred, 
        target,
        gamma=2.0,  # Keep the focusing parameter
        alpha=alpha  # Updated class weights
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
            
            # Calculate and print total gradient magnitude
            total_grad_magnitude = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_magnitude += param.grad.data.norm(2).item()
            print(f"Total gradient magnitude: {total_grad_magnitude:.6f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Optional: Print per-layer gradient magnitudes for debugging
            if epoch == 0 and len(train_pbar) < 5:  # Only for first few batches of first epoch
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Layer {name} gradient magnitude: {param.grad.data.norm(2).item():.6f}")
            
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
                
                # Add debugging prints
                pred_classes = outputs.argmax(dim=1)
                print(f"\nTest Predictions - Raw logits: {outputs[0][:5]}")  # Show first 5 logits
                print(f"Test Predicted class distribution: {[int((pred_classes == i).sum()) for i in range(5)]}")
                print(f"Test Target class distribution: {[int((batch.y == i).sum()) for i in range(5)]}")
                
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
        
        # Update learning rate
        scheduler.step(avg_test_loss)
    
    return best_test_loss