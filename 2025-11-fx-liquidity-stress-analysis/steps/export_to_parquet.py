#steps.export_to_parquet.py

import requests
from pathlib import Path
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

# Ensure output folder exists
Path(PARQUET_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with requests.get(
    url=f"{QUESTDB_PATH.rstrip('/')}/exp",
    params={"query": query, "fmt": "parquet"},
    stream=True,
) as r:
    r.raise_for_status()
    with open(PARQUET_OUTPUT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

print(f"Wrote Parquet to {PARQUET_OUTPUT_PATH}")
