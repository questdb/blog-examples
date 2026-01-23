"""
Feature engineering for time-series forecasting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "volume_lag_1", 
    "volume_lag_5", 
    "volume_lag_15",
    "volume_ma_15", 
    "volume_std_15", 
    "trade_count", 
    "price_range"
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for volume forecasting.
    
    Returns DataFrame with features and target column.
    """
    df = df.sort_values("timestamp").copy()
    
    # Ensure numeric types
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")
    df["price_range"] = pd.to_numeric(df["price_range"], errors="coerce")
    
    # Lag features
    for lag in [1, 5, 15]:
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    
    # Rolling statistics
    df["volume_ma_15"] = df["volume"].rolling(15).mean()
    df["volume_std_15"] = df["volume"].rolling(15).std()
    
    # Target: next period's volume
    df["target"] = df["volume"].shift(-1)
    
    # Drop rows with NaN (from lags/rolling)
    df = df.dropna()
    
    return df


def normalize_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    feature_cols: list[str] = FEATURE_COLS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Fits on train, transforms both train and test.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    return X_train, X_test, y_train, y_test, scaler
