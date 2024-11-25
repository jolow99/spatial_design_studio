import os
import torch
import pandas as pd
from src.models.dgcnn import DGCNN
from src.utils import load_config
from src.data_loader import PointCloudDataset

def get_latest_checkpoint():
    checkpoint_dir = "checkpoints"
    timestamp_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not timestamp_dirs:
        raise FileNotFoundError("No checkpoint directories found")
    latest_dir = max(timestamp_dirs)
    return os.path.join(checkpoint_dir, latest_dir)

def save_predictions(points, predictions, form_type, form_number):
    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Create DataFrame with original points and predictions
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'predicted_score': predictions.flatten()
    })
    
    # Save to CSV in predictions directory
    output_filename = os.path.join("predictions", f'predictions_{form_type}{form_number}.csv')
    df.to_csv(output_filename, index=False)
    print(f"Saved predictions to {output_filename}")

def main():
    # Get latest checkpoint directory
    checkpoint_dir = get_latest_checkpoint()
    
    # Load configuration from checkpoint directory
    config = load_config(os.path.join(checkpoint_dir, "config.yaml"))
    
    # Device configuration
    device = torch.device("cpu")
    
    # Initialize model
    model = DGCNN(k=config['model']['k'], dropout=config['model']['dropout']).to(device)
    
    # Load the best model
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
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
    
    # Test models to evaluate
    test_forms = [
        ('curved', 1),
        ('curved', 15),
        ('rect', 1),
        ('rect', 15)
    ]
    
    # Make predictions for each test form
    with torch.no_grad():
        for form_type, form_number in test_forms:
            # Find the corresponding dataset index
            target_idx = None
            for idx, data in enumerate(dataset):
                if (dataset.metadata[idx]['form_type'] == form_type and 
                    dataset.metadata[idx]['form_number'] == form_number):
                    target_idx = idx
                    break
            
            if target_idx is None:
                print(f"Could not find {form_type}{form_number} in dataset")
                continue
            
            # Get the data and make predictions
            data = dataset[target_idx].to(device)
            predictions = model(data).cpu().numpy()
            points = data.x.cpu().numpy()
            
            # Save predictions to CSV
            save_predictions(points, predictions, form_type, form_number)

if __name__ == "__main__":
    main()