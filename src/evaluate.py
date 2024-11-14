# src/evaluate.py
import torch
from src.utils import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            outputs = model(batch)
            total_mse += mean_squared_error(outputs, batch.y).item()
            total_mae += mean_absolute_error(outputs, batch.y).item()
    return total_mse / len(dataloader), total_mae / len(dataloader)