"""
PyTorch model and dataset classes for time-series forecasting.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sequential time-series data.
    
    Creates overlapping sequences of `sequence_length` to predict a target value.
    Use when data fits in memory.
    """
    
    def __init__(
        self, 
        features: np.ndarray, 
        targets: np.ndarray,
        sequence_length: int = 30
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]  # Target at end of sequence
        return X, y


class StreamingTimeSeriesDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams from Parquet.
    
    Use when data doesn't fit in memory.
    Requires a generator function that yields DataFrames.
    """
    
    def __init__(
        self, 
        batch_generator,  # Generator yielding DataFrames
        feature_cols: list[str],
        target_col: str = "target",
        sequence_length: int = 30
    ):
        self.batch_generator = batch_generator
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
    
    def __iter__(self):
        for batch_df in self.batch_generator:
            features = batch_df[self.feature_cols].values
            targets = batch_df[self.target_col].values
            
            for i in range(len(features) - self.sequence_length):
                X = features[i:i + self.sequence_length]
                y = targets[i + self.sequence_length - 1]
                
                yield (
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32)
                )


class SimpleLSTM(nn.Module):
    """
    Basic LSTM for sequence-to-value regression.
    
    This is intentionally simple—a demonstration of the data pipeline,
    not a production forecasting model.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, (hidden, _) = self.lstm(x)
        # Use final hidden state
        out = self.fc(hidden[-1])
        return out.squeeze(-1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0


def evaluate(model, dataloader, criterion, device):
    """Evaluate model, return average loss."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0
