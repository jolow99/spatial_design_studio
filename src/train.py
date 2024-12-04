# src/train.py
import torch
from src.utils import save_checkpoint
import torch.nn as nn
from tqdm import tqdm
import csv
import os

def softmax_focal_loss(pred, target, gamma=2.0, alpha=None):
    """
    Ordinal Focal Loss for saliency classification with ordered classes (0-4).
    
    Args:
        pred: Predictions (logits) with shape [batch_size, num_classes].
        target: Ground truth labels with shape [batch_size].
        gamma: Focusing parameter to reduce weight of easy examples.
        alpha: Optional class balancing weights (list or tensor of shape [num_classes]).
    
    Returns:
        Loss value (scalar).
    """
    # Apply softmax to logits to get probabilities
    probs = torch.nn.functional.softmax(pred, dim=1)
    batch_size = pred.size(0)
    num_classes = pred.size(1)
    
    # Convert target to one-hot but with cumulative probabilities
    # For example, if target is 3, we want [1, 1, 1, 1, 0]
    target_one_hot = torch.zeros_like(pred)
    for i in range(num_classes):
        target_one_hot[:, i] = (target >= i).float()
    
    # Calculate the ordinal loss with distance penalty
    distances = torch.abs(torch.arange(num_classes, device=pred.device).unsqueeze(0) - 
                         target.unsqueeze(1)).float()
    # distance_weights = 1 + distances  # Linear penalty for distance from true class
    distance_weights = torch.exp(distances) # Exponential penalty for distance from true class
    
    # Compute focal weights for each class prediction
    focal_weights = torch.abs(target_one_hot - probs) ** gamma
    
    # Compute the binary cross entropy for each ordinal level
    bce_loss = -(target_one_hot * torch.log(probs + 1e-8) + 
                 (1 - target_one_hot) * torch.log(1 - probs + 1e-8))
    
    # Apply distance weighting and focal weighting
    weighted_loss = distance_weights * focal_weights * bce_loss
    
    # Apply alpha (class weighting) if provided
    if alpha is not None:
        alpha = torch.tensor(alpha, device=pred.device)
        class_weights = alpha.gather(0, target)
        weighted_loss = weighted_loss * class_weights.unsqueeze(1)
    
    return weighted_loss.mean()

def calculate_class_weights(dataset):
    """
    Calculate class weights based on inverse class frequencies, capped at 200
    
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
    normalized_weights = [min(200.0, freq / min_weight) for freq in inverse_freqs]  # Cap at 200
    
    print("\nClass distribution and weights:")
    for i, (count, weight) in enumerate(zip(class_counts, normalized_weights)):
        print(f"Class {i}: {count} samples, weight = {weight:.2f}")
    
    return normalized_weights

def save_losses_to_csv(train_losses, test_losses, save_dir):
    """
    Save training and testing losses to a CSV file.
    
    Args:
        train_losses: List of average training losses per epoch
        test_losses: List of average testing losses per epoch
        save_dir: Directory to save the CSV file
    """
    csv_path = os.path.join(save_dir, 'losses.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Testing Loss'])
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses), start=1):
            writer.writerow([epoch, train_loss, test_loss])
    print(f"Losses saved to {csv_path}")

def train_model(model, optimizer, train_loader, test_loader, device, num_epochs, checkpoint_dir, config=None):
    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    
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
        gamma=4.0,  # Keep the focusing parameter
        alpha=alpha  # Updated class weights
    )
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc='Training')
        for data in train_pbar:
            optimizer.zero_grad()
            data = data.to(device)
            
            # Print model info for each batch
            print(f"\nTraining on {data.metadata['form_type'][0].capitalize()} Model {data.metadata['form_number'][0]}")
            
            # Split input features
            points = data.x[:, :3]  # xyz coordinates
            geom_features = data.x[:, 3:]  # geometric features
            
            outputs = model(data)
        
            pred_classes = outputs.argmax(dim=1)
            print(f"Training Batch Statistics:")
            print(f"Raw logits (first sample): {outputs[0][:5]}")
            print(f"Predicted class distribution: {[int((pred_classes == i).sum()) for i in range(5)]}")
            print(f"Target class distribution: {[int((data.y == i).sum()) for i in range(5)]}")
            
            # Calculate main classification loss
            loss = criterion(outputs, data.y)
            loss.backward()
            
            # Gradient clipping and optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            batch_size = data.x.size(0)
            total_train_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar with only the classification loss
            train_pbar.set_postfix({
                'Class Loss': f'{loss.item():.4f}'
            })
        
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
                
                # Simplified test debugging prints
                pred_classes = outputs.argmax(dim=1)
                print(f"\nTest Batch Statistics:")
                print(f"Predicted class distribution: {[int((pred_classes == i).sum()) for i in range(5)]}")
                print(f"Target class distribution: {[int((batch.y == i).sum()) for i in range(5)]}")
                
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
        
        # After calculating averages, add these lines:
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
    
    # Save losses to CSV after training completes
    save_losses_to_csv(train_losses, test_losses, save_checkpoint.timestamp_dir)
    
    return best_test_loss