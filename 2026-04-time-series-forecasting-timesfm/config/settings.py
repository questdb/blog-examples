from pathlib import Path

# QuestDB connection settings
# REST API - points to public demo instance (works out of the box)
QUESTDB_HTTP = "https://demo.questdb.io"

# PostgreSQL wire protocol - points to local QuestDB (for ConnectorX)
# The demo instance doesn't expose port 8812 externally, so this requires local QuestDB
QUESTDB_PG = "postgresql://admin:quest@localhost:8812/qdb"

# For local QuestDB with Parquet partitions (Method 3)
QUESTDB_DATA_DIR = Path("/var/lib/questdb/db")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data settings
TABLE_NAME = "trades"
SYMBOL = "BTC-USDT"

# TimesFM settings
TIMESFM_MODEL = "google/timesfm-2.5-200m-pytorch"
MAX_CONTEXT = 512      # How much history to use
FORECAST_HORIZON = 60  # How far ahead to forecast (in minutes)
