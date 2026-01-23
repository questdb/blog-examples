#!/usr/bin/env python3
"""
Full Training Pipeline

Loads data from Parquet (exported via Method 2), engineers features,
trains a simple LSTM model, and saves the result.

Run Method 2 first to create the training data:
    python steps/method2_parquet_export.py
    
Then run this:
    python steps/train.py
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from src.features import prepare_features, normalize_features, FEATURE_COLS
from src.model import TimeSeriesDataset, SimpleLSTM, train_epoch, evaluate
from config.settings import (
    DATASET_DIR, MODEL_DIR, 
    SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE
)


def main():
    print("=" * 60)
    print("PyTorch Training Pipeline")
    print("=" * 60)
    
    # Check for training data
    parquet_path = DATASET_DIR / "training_data.parquet"
    if not parquet_path.exists():
        print(f"\n⚠ Training data not found: {parquet_path}")
        print("Run Method 2 first to export the data:")
        print("    python steps/method2_parquet_export.py")
        return
    
    # Load data
    print(f"\n1. Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"   Loaded {len(df):,} rows")
    
    # Feature engineering
    print("\n2. Engineering features...")
    df = prepare_features(df)
    print(f"   After feature engineering: {len(df):,} rows")
    print(f"   Features: {FEATURE_COLS}")
    
    # Train/test split (respect time order)
    print("\n3. Splitting data (80/20, time-ordered)...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")
    
    # Normalize
    print("\n4. Normalizing features...")
    X_train, X_test, y_train, y_test, scaler = normalize_features(train_df, test_df)
    
    # Create PyTorch datasets
    print(f"\n5. Creating PyTorch datasets (sequence_length={SEQUENCE_LENGTH})...")
    train_dataset = TimeSeriesDataset(X_train, y_train, SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(X_test, y_test, SEQUENCE_LENGTH)
    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Test sequences:  {len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n6. Initializing model on {device}...")
    
    model = SimpleLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=32,
        num_layers=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n7. Training for {EPOCHS} epochs...")
    print("-" * 40)
    
    best_test_loss = float("inf")
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
    
    print("-" * 40)
    print(f"   Best test loss: {best_test_loss:.4f} (epoch {best_epoch})")
    
    # Save model
    print("\n8. Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "volume_forecast_lstm.pt"
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": FEATURE_COLS,
        "sequence_length": SEQUENCE_LENGTH,
        "test_loss": best_test_loss
    }, model_path)
    
    print(f"   ✓ Saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
