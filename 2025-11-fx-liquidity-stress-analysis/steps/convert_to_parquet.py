#steps.convert_to_parquet.py

import requests
import pandas as pd
from config.settings import QUESTDB_PATH, PARQUET_OUTPUT_PATH, TABLE_NAME, ROWS_LIMIT

query = f"""
WITH training_data AS (
  SELECT timestamp, symbol,
    asks[1][1] as ask_price, bids[1][1] as bid_price,
    asks[2][1] as ask_volume, bids[2][1] as bid_volume
    FROM {TABLE_NAME}
    WHERE symbol = 'EURUSD'
    AND timestamp BETWEEN dateadd('d', -2, now()) AND dateadd('d', -1, now())
    LIMIT {ROWS_LIMIT}
)
SELECT
   timestamp,
   symbol,
   avg(ask_price - bid_price) AS spread,
   avg((ask_price - bid_price) / ((ask_price + bid_price) / 2) * 10000) AS spread_bps,
   avg(bid_volume + ask_volume) AS total_volume,
   avg((bid_volume - ask_volume) / (bid_volume + ask_volume)) AS imbalance,
   avg(bid_price) AS bid_price,
   avg(ask_price) AS ask_price,
   avg((bid_price + ask_price) / 2) AS mid_price
FROM training_data
SAMPLE BY 500T;
"""

print(f"Exporting rows")

response = requests.get(
   url=f"{QUESTDB_PATH.rstrip('/')}/exec",
   params={'query': query}
)
response.raise_for_status()
data = response.json()

print(f"Converting rows to Parquet")
columns = [col['name'] for col in data['columns']]
df = pd.DataFrame(data['dataset'], columns=columns)

df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
