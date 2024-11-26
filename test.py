import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import load_checkpoint, load_config
from src.models.dgcnn import DGCNN
from src.data_loader import PointCloudDataset, get_train_test_dataloaders

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
    model = DGCNN(k=config['model']['k'], dropout=config['model']['dropout']).to(device)
    
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
    
    # Get test loader with specific test models
    _, test_loader = get_train_test_dataloaders(
        dataset,
        test_models=[1, 15],
        batch_size=1
    )
    
    # Colors and class names for visualization
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    class_names = ['No attention', 'Low', 'Medium-low', 'Medium-high', 'High']
    
    # Create visualization directory with same timestamp as model
    vis_dir = os.path.join("visualizations", latest_dir)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Saving visualizations to: {vis_dir}")
    
    # Test and visualize each model
    for batch in test_loader:
        form_type = batch.metadata['form_type'][0]
        form_number = batch.metadata['form_number'][0]
        
        # Get predictions
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            predictions = torch.softmax(outputs, dim=1)
            pred_classes = predictions.argmax(dim=1).cpu().numpy()
        
        points = batch.x.cpu().numpy()
        ground_truth = batch.y.cpu().numpy()
        
        # Create visualization
        fig = plt.figure(figsize=(15, 7))
        
        # Ground truth plot
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=[colors[int(c)] for c in ground_truth],
                             s=2)
        ax1.set_title(f"{form_type.capitalize()} Model {form_number} - Ground Truth")
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=c, label=class_names[i],
                                    markersize=10)
                         for i, c in enumerate(colors)]
        ax1.legend(handles=legend_elements)
        
        # Prediction plot
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=[colors[int(c)] for c in pred_classes],
                             s=2)
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
        
        # Add accuracy information
        accuracy = np.mean(pred_classes == ground_truth)
        plt.suptitle(f'Accuracy: {accuracy:.2%}', y=0.95)
        
        plt.tight_layout()
        
        # Update save path to use the timestamped directory
        save_path = os.path.join(vis_dir, f"{form_type}_model_{form_number}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        
        # Print classification report
        print(f"\nResults for {form_type.capitalize()} Model {form_number}:")
        print(f"Overall Accuracy: {accuracy:.2%}")
        print("\nClass Distribution:")
        for i in range(5):
            print(f"Class {i} ({class_names[i]}):")
            print(f"  Ground Truth: {np.sum(ground_truth == i)}")
            print(f"  Predicted: {np.sum(pred_classes == i)}")

if __name__ == "__main__":
    test_and_visualize()
