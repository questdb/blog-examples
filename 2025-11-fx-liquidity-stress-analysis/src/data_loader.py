# src/data_loader.py
import pandas as pd

class DataLoader:
   def __init__(self, file_path):
       self.file_path = file_path

   def load_data(self):
       """Load and prepare raw Parquet data"""
       print("Loading data...")
       df = pd.read_parquet(self.file_path)
       df['timestamp'] = pd.to_datetime(df['timestamp'])
       df = df.sort_values('timestamp').reset_index(drop=True)
       return df

