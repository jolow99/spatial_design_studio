# src/train.py
import torch
from src.utils import save_checkpoint
import torch.nn as nn
from tqdm import tqdm
import csv
import os
from tabulate import tabulate
import numpy as np
from sklearn.metrics import f1_score

def softmax_focal_loss(pred, target, gamma=2.0, alpha=None):
    """
    Modified Ordinal Focal Loss for saliency classification with ordered classes (0-4).
    Uses triangular encoding to better represent ordinal relationships.
    
    Args:
        pred: Predictions (logits) with shape [batch_size, num_classes]
        target: Ground truth labels with shape [batch_size]
        gamma: Focusing parameter
        alpha: Class balancing weights
    """
    probs = torch.nn.functional.softmax(pred, dim=1)
    batch_size = pred.size(0)
    num_classes = pred.size(1)
    
    # Create triangular ordinal encoding
    # For example, if target is 3:
    # [0.25, 0.5, 0.75, 1.0, 0.0] instead of [1, 1, 1, 1, 0]
    target_ordinal = torch.zeros_like(pred)
    for i in range(num_classes):
        # Points up to target get increasing probability
        target_ordinal[:, i] = torch.where(
            target >= i,
            (1.0 + i) / (target + 1.0),  # Increasing values up to 1.0 at target
            torch.where(
                target + 1 == i,
                0.25,  # Small probability for next class
                0.0    # Zero for classes far beyond target
            )
        )
    
    # Calculate distances with linear penalty
    distances = torch.abs(torch.arange(num_classes, device=pred.device).unsqueeze(0) - 
                         target.unsqueeze(1)).float()
    distance_weights = 1 + distances  # Linear penalty instead of exponential
    
    # Compute focal weights
    focal_weights = torch.abs(target_ordinal - probs) ** gamma
    
    # Binary cross entropy with ordinal targets
    bce_loss = -(target_ordinal * torch.log(probs + 1e-8) + 
                 (1 - target_ordinal) * torch.log(1 - probs + 1e-8))
    
    # Apply distance and focal weighting
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
    
    # Create initial checkpoint to establish timestamp_dir
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=-1,  # Initialize at -1 since training hasn't started
        val_loss=float('inf'),
        checkpoint_dir=checkpoint_dir,
        is_best=False,
        config=config
    )
    
    # Now we can safely access the timestamp_dir
    train_metrics_path = os.path.join(save_checkpoint.timestamp_dir, 'batch_metrics_train.csv')
    test_metrics_path = os.path.join(save_checkpoint.timestamp_dir, 'batch_metrics_test.csv')
    
    # Initialize CSV files with headers
    headers = ['Epoch', 'Form_Type', 'Form_Number', 'Class', 'True_Count', 'Predicted_Count', 'F1_Score', 'Accuracy']
    for path in [train_metrics_path, test_metrics_path]:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
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
            
            # Calculate metrics
            true_classes = data.y.cpu().numpy()
            pred_classes_np = pred_classes.cpu().numpy()
            
            # Calculate all metrics
            class_distribution = [int((pred_classes == i).sum()) for i in range(5)]
            target_distribution = [int((data.y == i).sum()) for i in range(5)]
            f1_scores = f1_score(true_classes, pred_classes_np, average=None, zero_division=0)
            
            # Create metrics table
            metrics_table = []
            headers = ['Class', 'True Count', 'Predicted', 'F1 Score', 'Accuracy']
            
            for i in range(5):
                total = target_distribution[i]
                correct = ((pred_classes == i) & (data.y == i)).sum().item()
                accuracy = f"{(correct/total)*100:.1f}%" if total > 0 else "N/A"
                
                metrics_table.append([
                    f"Class {i}",
                    target_distribution[i],
                    class_distribution[i],
                    f"{f1_scores[i]:.3f}",
                    accuracy
                ])
            
            print("\nBatch Statistics:")
            print(tabulate(metrics_table, headers=headers, tablefmt='grid'))
            
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
            
            # After calculating metrics_table, save to CSV
            with open(train_metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in metrics_table:
                    writer.writerow([
                        epoch,
                        data.metadata['form_type'][0],
                        data.metadata['form_number'][0],
                        row[0],  # Class
                        row[1],  # True Count
                        row[2],  # Predicted
                        row[3],  # F1 Score
                        row[4]   # Accuracy
                    ])
        
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
                
                # Calculate metrics
                pred_classes = outputs.argmax(dim=1)
                true_classes = batch.y.cpu().numpy()
                pred_classes_np = pred_classes.cpu().numpy()
                
                # Calculate all metrics
                class_distribution = [int((pred_classes == i).sum()) for i in range(5)]
                target_distribution = [int((batch.y == i).sum()) for i in range(5)]
                
                # Ensure f1_scores has entries for all classes
                f1_scores_raw = f1_score(true_classes, pred_classes_np, average=None, zero_division=0)
                f1_scores = np.zeros(5)  # Initialize array for all 5 classes
                f1_scores[:len(f1_scores_raw)] = f1_scores_raw  # Fill with calculated scores
                
                # Create metrics table
                metrics_table = []
                headers = ['Class', 'True Count', 'Predicted', 'F1 Score', 'Accuracy']
                
                for i in range(5):
                    total = target_distribution[i]
                    correct = ((pred_classes == i) & (batch.y == i)).sum().item()
                    accuracy = f"{(correct/total)*100:.1f}%" if total > 0 else "N/A"
                    
                    metrics_table.append([
                        f"Class {i}",
                        target_distribution[i],
                        class_distribution[i],
                        f"{f1_scores[i]:.3f}",
                        accuracy
                    ])
                
                print(f"\nTest Statistics for {form_type[0].capitalize()} Model {form_number}:")
                print(tabulate(metrics_table, headers=headers, tablefmt='grid'))
                
                test_loss = criterion(outputs, batch.y).item()
                
                batch_size = batch.x.size(0)
                total_test_loss += test_loss * batch_size
                test_samples += batch_size
                
                test_pbar.set_postfix({'Loss': f'{test_loss:.4f}'})
                
                # After calculating metrics_table, save to CSV
                with open(test_metrics_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for row in metrics_table:
                        writer.writerow([
                            epoch,
                            batch.metadata['form_type'][0],
                            batch.metadata['form_number'][0],
                            row[0],  # Class
                            row[1],  # True Count
                            row[2],  # Predicted
                            row[3],  # F1 Score
                            row[4]   # Accuracy
                        ])
        
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