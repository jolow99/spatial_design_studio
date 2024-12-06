import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import load_checkpoint, load_config
from src.models.dgcnn import ModifiedDGCNN
from src.data_loader import PointCloudDataset, get_dataloader

def test_and_visualize():
    # Find the latest checkpoint directory
    checkpoint_dir = "checkpoints"
    timestamp_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not timestamp_dirs:
        raise FileNotFoundError("No checkpoint directories found")
    latest_dir = max(timestamp_dirs)
    
    # Load config from the checkpoint directory
    config_path = os.path.join(checkpoint_dir, latest_dir, 'config.yaml')
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ModifiedDGCNN(num_classes=5, k=config['model']['k']).to(device)
    
    # Load best model from latest directory
    checkpoint_path = os.path.join(checkpoint_dir, latest_dir, 'best_model.pt')
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = PointCloudDataset(
        data_dir=config['data']['path'],
        demographic=config['data']['demographic'],
        subject=config['data']['subject']
    )
    
    # Get dataloader with batch size 1 for visualization
    dataloader = get_dataloader(dataset, batch_size=1)
    
    # Colors and class names for visualization
    colors = ['lightgray', 'green', 'yellow', 'orange', 'red']
    class_names = ['No attention', 'Low', 'Medium-low', 'Medium-high', 'High']
    
    # Create visualization directory with same timestamp as model
    vis_dir = os.path.join("visualizations", latest_dir)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Saving visualizations to: {vis_dir}")
    
    # Test and visualize each model
    for batch in dataloader:
        form_type = batch.metadata['form_type'][0]
        form_number = batch.metadata['form_number'][0]
        
        # Get predictions
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            # Add debugging prints
            print(f"\nTest Predictions - Raw logits: {outputs[0][:5]}")  # Show first 5 logits
            print(f"Logits min/max: {outputs.min().item():.4f}/{outputs.max().item():.4f}")
            
            # Print logits distribution
            print("\nLogits distribution:")
            for i in range(5):
                class_logits = outputs[:, i]
                print(f"Class {i}: min={class_logits.min().item():.4f}, "
                      f"max={class_logits.max().item():.4f}, "
                      f"mean={class_logits.mean().item():.4f}")
            
            pred_classes = outputs.argmax(dim=1).cpu().numpy()
            print(f"\nPredicted class distribution: {[int((pred_classes == i).sum()) for i in range(5)]}")
            print(f"Target class distribution: {[int((batch.y.cpu().numpy() == i).sum()) for i in range(5)]}")
        
        points = batch.x.cpu().numpy()
        ground_truth = batch.y.cpu().numpy()
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))  # Increased figure height to accommodate metrics
        
        # Ground truth plot
        ax1 = fig.add_subplot(121, projection='3d')
        point_colors_gt = [colors[int(c)] for c in ground_truth]
        point_sizes_gt = [2 if c == 0 else 8 for c in ground_truth]  # Smaller size for 'No attention'
        point_alphas_gt = [0.4 if c == 0 else 1.0 for c in ground_truth]  # More transparent for 'No attention'
        
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=point_colors_gt,
                               s=point_sizes_gt,
                               alpha=point_alphas_gt)
        ax1.set_title(f"{form_type.capitalize()} Model {form_number} - Ground Truth")
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=c, label=class_names[i],
                                      markersize=10,
                                      alpha=1.0 if i > 0 else 0.4)  # Match transparency
                           for i, c in enumerate(colors)]
        ax1.legend(handles=legend_elements)
        
        # Prediction plot
        ax2 = fig.add_subplot(122, projection='3d')
        point_colors_pred = [colors[int(c)] for c in pred_classes]
        point_sizes_pred = [2 if c == 0 else 8 for c in pred_classes]
        point_alphas_pred = [0.4 if c == 0 else 1.0 for c in pred_classes]
        
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=point_colors_pred,
                               s=point_sizes_pred,
                               alpha=point_alphas_pred)
        ax2.set_title(f"{form_type.capitalize()} Model {form_number} - Predicted")
        ax2.legend(handles=legend_elements)
        
        # Set consistent viewing angles and limits
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=30, azim=45)
            
            max_range = np.array([
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add metrics text below the plots
        accuracy = np.mean(pred_classes == ground_truth)
        metrics_text = f"Overall Accuracy: {accuracy:.2%}\n\nClass Distribution and Accuracies:\n"
        for i in range(5):
            class_mask = (ground_truth == i)
            class_accuracy = np.mean(pred_classes[class_mask] == ground_truth[class_mask]) if np.any(class_mask) else 0
            metrics_text += f"Class {class_names[i]}:\n"
            metrics_text += f"  Ground Truth: {np.sum(ground_truth == i)}, "
            metrics_text += f"Predicted: {np.sum(pred_classes == i)}, "
            metrics_text += f"Accuracy: {class_accuracy:.2%}\n"

        # Add text box with metrics
        plt.figtext(0.1, 0.02, metrics_text, fontsize=8, va='bottom', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout to make room for metrics
        plt.subplots_adjust(bottom=0.25)  # Adjust this value based on the amount of text
        
        # Remove the print statements for metrics
        save_path = os.path.join(vis_dir, f"{form_type}_model_{form_number}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    test_and_visualize()
