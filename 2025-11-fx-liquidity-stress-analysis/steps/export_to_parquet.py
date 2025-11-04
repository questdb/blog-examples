#steps.convert_to_parquet.py

import requests
from pathlib import Path
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
 AND dateadd('d', -7, now()) < timestamp
LIMIT {ROWS_LIMIT};
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
