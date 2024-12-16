# src/evaluate.py
import torch
from src.utils import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import torch.nn as nn

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            total_loss += criterion(outputs, batch.y).item()
    return total_loss / len(dataloader)