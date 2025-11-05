# steps/inference.py
from io import BytesIO
from pathlib import Path
import requests
import pandas as pd
from src.feature_extraction import FeatureEngineer
from src.model import LiquidityStressModel
from config.settings import QUESTDB_PATH, TABLE_NAME, MODEL_PATH, INFERENCE_OUTPUT

# 1) Fetch the latest 50 rows (CSV via /exp), same colums as in training
EXP_URL = f"{QUESTDB_PATH.rstrip('/')}/exp"
sql = f"""
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
LIMIT -50
"""

r = requests.get(EXP_URL, params={"fmt": "csv", "query": sql})
r.raise_for_status()
df = pd.read_csv(BytesIO(r.content))

# 2) Features
df["timestamp"] = pd.to_datetime(df["timestamp"])
engineer = FeatureEngineer()
df = engineer.create_features(df)

feature_cols = [
    "spread_bps", "total_volume", "imbalance",
    "spread_ma_5m", "spread_std_5m",
    "price_change_1m", "price_change_5m",
    "volatility_5m", "spread_trend", "volume_surge",
]

# 3) Model + predict
model = LiquidityStressModel()
model.load_model(MODEL_PATH)
model.feature_columns = feature_cols

def align_features(df, cols):
    aligned = {c: (df[c] if c in df.columns else 0) for c in cols}
    return pd.DataFrame(aligned)[cols]

X = align_features(df, feature_cols).values
preds = [model.predict_stress(row) for row in X]
pred_df = pd.DataFrame(preds)

# 4) Save results
Path(INFERENCE_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
result = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
result.to_csv(INFERENCE_OUTPUT, index=False)
print(f"Wrote inference results → {INFERENCE_OUTPUT}")

# Display sample
print(result[["timestamp", "stress_probability", "alert_level"]].to_string(index=False))
