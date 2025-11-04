# config/settings.py

# QuestDB REST endpoint for SQL
QUESTDB_PATH = "https://demo.questdb.io"

# Source table to query
TABLE_NAME = "market_data"

# Rows to export
ROWS_LIMIT = 5000000

# Where to write the Parquet file
PARQUET_OUTPUT_PATH = "dataset/output.parquet"

#Path to store the training model data
MODEL_PATH="model_registry/fx_liquidity_model.json"

#Path to store the inference output
INFERENCE_OUTPUT = "results/inference_latest.csv"
