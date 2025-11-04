#steps.convert_to_parquet.py

import requests
import pandas as pd
from config.settings import QUESTDB_PATH, PARQUET_OUTPUT_PATH, TABLE_NAME, ROWS_LIMIT

query = f"""
SELECT
   timestamp,
   symbol,
   asks[1][1] - bids[1][1] AS spread,
   (asks[1][1] - bids[1][1]) / ((asks[1][1] + bids[1][1]) / 2) * 10000 AS spread_bps,
   bids[2][1] + asks[2][1] AS total_volume,
   (bids[2][1] - asks[2][1]) / (bids[2][1] + asks[2][1]) AS imbalance,
   bids[1][1] AS bid_price,
   asks[1][1] AS ask_price,
   (bids[1][1] + asks[1][1]) / 2 AS mid_price
FROM {TABLE_NAME}
WHERE symbol = 'EURUSD'
 AND dateadd('d', -1, now()) < timestamp
LIMIT {ROWS_LIMIT};
"""

print(f"Exporting {ROWS_LIMIT} rows")

response = requests.get(
   url=f"{QUESTDB_PATH.rstrip('/')}/exec",
   params={'query': query}
)
response.raise_for_status()
data = response.json()

print(f"Converting {ROWS_LIMIT} rows to Parquet")
columns = [col['name'] for col in data['columns']]
df = pd.DataFrame(data['dataset'], columns=columns)

df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
