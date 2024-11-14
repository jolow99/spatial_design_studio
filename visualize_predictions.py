import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import load_checkpoint
from src.models.dgcnn import DGCNN
from src.data_loader import PointCloudDataset

def visualize_prediction(model, data, device, save_path=None):
    """
    Visualize ground truth saliency and predicted saliency scores
    
    Args:
        model: Trained DGCNN model
        data: Single PyG Data object containing the point cloud
        device: torch device
        save_path: Optional path to save the visualization
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        data = data.to(device)
        predictions = model(data).cpu().numpy()
    
    # Get point cloud coordinates and ground truth saliency
    points = data.x.cpu().numpy()
    ground_truth = data.y.cpu().numpy()
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    
    # Ground truth saliency (left)
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=ground_truth, cmap='viridis', s=1)
    ax1.set_title('Ground Truth Saliency')
    plt.colorbar(scatter1)
    
    # Predicted saliency (right)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=predictions, cmap='viridis', s=1)
    ax2.set_title('Predicted Saliency Scores')
    plt.colorbar(scatter2)
    
    # Set consistent viewing angles and limits
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=45)
        
        # Set equal aspect ratio
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
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # Load configuration
    from src.utils import load_config
    config = load_config("configs/config.yaml")
    
    # Device configuration
    device = torch.device("cpu")
    
    # Initialize model with same configuration as training
    model = DGCNN(k=config['model']['k'], dropout=config['model']['dropout']).to(device)
    
    # Load the best model - modified to skip optimizer loading
    checkpoint_path = f"{config['training']['checkpoint_dir']}/best_model.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    dataset = PointCloudDataset(
        data_dir=config['data']['path'],
        file_names=config['data']['file_names']
    )
    
    # Visualize predictions for each point cloud in the dataset
    for i in range(len(dataset)):
        data = dataset[i]
        visualize_prediction(
            model=model,
            data=data,
            device=device,
            save_path=f"visualization_sample_{i}.png"
        )

if __name__ == "__main__":
    main() 