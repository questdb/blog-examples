# src/label_creator.py
import pandas as pd

class LabelCreator:
    def __init__(self, horizon_minutes=10):
        """
        horizon_minutes: how far ahead to predict stress (default = 10 minutes)
        """
        self.horizon_minutes = horizon_minutes

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary stress labels"""

        # Thresholds
        spread_threshold = df['spread_bps'].quantile(0.95)
        volume_threshold = df['total_volume'].quantile(0.05)


        # Stress conditions
        stress_events = (
            (df['spread_bps'] > spread_threshold) |
            (df['total_volume'] < volume_threshold)
        ).astype(int)

        # Shift labels forward in time
        horizon_rows = self.horizon_minutes * 60  # assuming 1s intervals
        df['stress_label'] = stress_events.shift(-horizon_rows)

        # Drop last rows where future labels don't exist
        df = df[:-horizon_rows].copy()

        stress_count = df['stress_label'].sum()
        total_count = len(df)

        return df

