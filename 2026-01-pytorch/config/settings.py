from pathlib import Path

# QuestDB connection
QUESTDB_HTTP = "https://demo.questdb.io"
QUESTDB_PG = "postgresql://admin:quest@demo.questdb.io:8812/qdb"

# For local QuestDB with Parquet partitions (Method 3)
QUESTDB_DATA_DIR = Path("/var/lib/questdb/db")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"

# Data settings
TABLE_NAME = "trades"
SYMBOL = "BTC-USDT"

# Model settings
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
