# src/feature_extraction.py
import pandas as pd

class FeatureEngineer:
    def __init__(self, window_seconds=300):
        """
        window_seconds: rolling window size (default = 5 minutes at 1-second intervals)
        """
        self.window = window_seconds

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling, trend, and volatility features"""
        # Rolling averages & std (5 min window)
        df['spread_ma_5m'] = df['spread_bps'].rolling(self.window, min_periods=1).mean()
        df['spread_std_5m'] = df['spread_bps'].rolling(self.window, min_periods=1).std()
        df['volume_ma_5m'] = df['total_volume'].rolling(self.window, min_periods=1).mean()
        df['imbalance_ma_5m'] = df['imbalance'].rolling(self.window, min_periods=1).mean()

        # Price change trends
        df['price_change_1m'] = df['mid_price'].diff(60)   # 1 min = 60 sec
        df['price_change_5m'] = df['mid_price'].diff(300)  # 5 min = 300 sec

        # Volatility (rolling std of pct changes)
        df['volatility_5m'] = df['mid_price'].pct_change().rolling(self.window, min_periods=1).std()

        # Spread trend (1 min difference)
        df['spread_trend'] = df['spread_bps'].diff(60)

        # Volume surge (current / avg of last 5 min)
        df['volume_surge'] = df['total_volume'] / df['volume_ma_5m']

        # Fill any NaNs
        df = df.ffill().fillna(0)
        return df

