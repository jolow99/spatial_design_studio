import matplotlib.pyplot as plt
import csv
import os

def plot_loss_curves_from_csv(csv_path):
    epochs = []
    train_losses = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_losses.append(float(row['Training Loss']))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Find the latest checkpoint directory
    checkpoint_dir = "checkpoints"
    timestamp_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not timestamp_dirs:
        raise FileNotFoundError("No checkpoint directories found")
    latest_dir = max(timestamp_dirs)
    
    # Construct path to losses.csv in the latest directory
    csv_path = os.path.join(checkpoint_dir, latest_dir, 'losses.csv')
    print(f"Loading losses from: {csv_path}")
    plot_loss_curves_from_csv(csv_path) 