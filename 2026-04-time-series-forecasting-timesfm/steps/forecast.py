#!/usr/bin/env python3
"""
Zero-Shot Forecasting with TimesFM

This script demonstrates using Google's TimesFM foundation model
for time-series forecasting on QuestDB data.

We forecast two operationally useful metrics:
1. Volume - When will liquidity be available? Useful for execution timing.
2. Volatility - How much is price moving? Useful for risk management.

Note: We intentionally avoid price prediction. Forecasting "where will BTC
be in an hour" is essentially claiming to beat the market—not credible.
Volume and volatility forecasting, however, serve real operational needs.
"""
# Suppress HuggingFace warnings before any imports
import os
import logging
import warnings
os.environ["HF_HUB_VERBOSITY"] = "error"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import query_rest_api, get_ohlcv_query
from src.forecaster import TimesFMForecaster
from config.settings import (
    SYMBOL, DATASET_DIR, OUTPUT_DIR,
    MAX_CONTEXT, FORECAST_HORIZON
)


def load_data() -> pd.DataFrame:
    """
    Load OHLCV data from Parquet (preferred) or live query (fallback).
    
    Returns DataFrame with columns: timestamp, open, high, low, close, volume, trade_count
    """
    parquet_path = DATASET_DIR / "btc_ohlcv_7d.parquet"
    
    if parquet_path.exists():
        print(f"Loading from Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"Parquet not found, querying live data from QuestDB...")
        print(f"  (Run method2_parquet_export.py first for reproducible results)")
        query = get_ohlcv_query(symbol=SYMBOL, days=7)
        df = query_rest_api(query)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def compute_volatility(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-minute volatility as the high-low range.
    
    This is a simple measure of intra-minute price movement.
    Higher values = more volatile = higher risk/reward.
    """
    return (df["high"] - df["low"]).values


def main():
    print("=" * 70)
    print("Zero-Shot Forecasting with TimesFM")
    print("=" * 70)
    
    # --- 1. Load Data ---
    print("\n1. LOADING DATA")
    print("-" * 70)
    df = load_data()
    print(f"   Rows loaded:  {len(df):,}")
    print(f"   Time range:   {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Columns:      {', '.join(df.columns)}")
    
    # --- 2. Prepare Series ---
    print("\n2. PREPARING TIME SERIES")
    print("-" * 70)
    
    timestamps = pd.DatetimeIndex(df["timestamp"])
    
    # Volume: total amount traded per minute
    volume_series = df["volume"].values
    print(f"   Volume series:     {len(volume_series)} points")
    print(f"     Recent values:   {volume_series[-5:].round(3)}")
    print(f"     Mean:            {volume_series.mean():.3f}")
    
    # Volatility: high-low range per minute  
    volatility_series = compute_volatility(df)
    print(f"   Volatility series: {len(volatility_series)} points")
    print(f"     Recent values:   {volatility_series[-5:].round(2)}")
    print(f"     Mean:            ${volatility_series.mean():.2f}")
    
    print(f"\n   Using last {MAX_CONTEXT} points as context for forecasting")
    print(f"   Forecasting {FORECAST_HORIZON} minutes ahead")
    
    # --- 3. Load Model ---
    print("\n3. LOADING TIMESFM MODEL")
    print("-" * 70)
    print("   TimesFM is a foundation model trained on 100B+ time points.")
    print("   It provides zero-shot forecasting—no training on your data needed.")
    print()
    
    forecaster = TimesFMForecaster(
        max_context=MAX_CONTEXT,
        max_horizon=FORECAST_HORIZON,
        use_quantiles=True,
    )
    forecaster.load_model()
    
    # --- 4. Generate Forecasts ---
    print("\n4. GENERATING FORECASTS")
    print("-" * 70)
    
    print("   Forecasting volume (liquidity timing)...")
    volume_result = forecaster.forecast(
        series=volume_series,
        timestamps=timestamps,
        horizon=FORECAST_HORIZON,
        freq="1min",
    )
    
    print("   Forecasting volatility (risk assessment)...")
    volatility_result = forecaster.forecast(
        series=volatility_series,
        timestamps=timestamps,
        horizon=FORECAST_HORIZON,
        freq="1min",
    )
    
    # --- 5. Display Results ---
    print("\n" + "=" * 70)
    print("FORECAST RESULTS")
    print("=" * 70)
    
    # Volume forecast
    print("\n📊 VOLUME FORECAST (next 60 minutes)")
    print("   Use case: Execution timing—when will liquidity be available?")
    print()
    volume_df = volume_result.to_dataframe()
    # Show subset of columns for readability
    display_cols = ["timestamp", "forecast", "q10", "q50", "q90"]
    print(volume_df[display_cols].head(10).to_string(index=False))
    print("   ...")
    print(volume_df[display_cols].tail(5).to_string(index=False))
    
    # Volatility forecast
    print("\n📈 VOLATILITY FORECAST (next 60 minutes)")
    print("   Use case: Risk management—how much is price likely to move?")
    print()
    volatility_df = volatility_result.to_dataframe()
    print(volatility_df[display_cols].head(10).to_string(index=False))
    print("   ...")
    print(volatility_df[display_cols].tail(5).to_string(index=False))
    
    # --- 6. Save Results ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    volume_output = OUTPUT_DIR / "volume_forecast.csv"
    volatility_output = OUTPUT_DIR / "volatility_forecast.csv"
    
    volume_df.to_csv(volume_output, index=False)
    volatility_df.to_csv(volatility_output, index=False)
    
    print(f"\n✓ Saved forecasts to:")
    print(f"   - {volume_output}")
    print(f"   - {volatility_output}")
    
    # --- 7. Summary Statistics ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Volume summary
    last_volume = volume_series[-1]
    forecast_volume = volume_result.point_forecast
    
    print(f"\n📊 Volume:")
    print(f"   Last observed:       {last_volume:.3f}")
    print(f"   Forecast mean:       {forecast_volume.mean():.3f}")
    print(f"   Forecast range:      {forecast_volume.min():.3f} – {forecast_volume.max():.3f}")
    if volume_result.quantile_forecast is not None:
        q10_mean = volume_result.quantile_forecast[:, 0].mean()
        q90_mean = volume_result.quantile_forecast[:, -1].mean()
        print(f"   80% confidence band: {q10_mean:.3f} – {q90_mean:.3f}")
    
    # Volatility summary
    last_volatility = volatility_series[-1]
    forecast_volatility = volatility_result.point_forecast
    
    print(f"\n📈 Volatility (price range per minute):")
    print(f"   Last observed:       ${last_volatility:.2f}")
    print(f"   Forecast mean:       ${forecast_volatility.mean():.2f}")
    print(f"   Forecast range:      ${forecast_volatility.min():.2f} – ${forecast_volatility.max():.2f}")
    if volatility_result.quantile_forecast is not None:
        q10_mean = volatility_result.quantile_forecast[:, 0].mean()
        q90_mean = volatility_result.quantile_forecast[:, -1].mean()
        print(f"   80% confidence band: ${q10_mean:.2f} – ${q90_mean:.2f}")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print("""
   • Wide confidence intervals are expected—volume and volatility are
     inherently noisy and driven by unpredictable factors (news, large
     orders, market sentiment).
   
   • These forecasts provide a baseline. For production use, you'd want
     to backtest against historical data and compare with simpler methods
     like moving averages or ARIMA.
   
   • The value of foundation models like TimesFM is speed: you get a
     reasonable forecast immediately, without training or tuning.
    """)
    
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
