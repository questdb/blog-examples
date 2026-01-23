# PyTorch + QuestDB Data Pipeline Tutorial

This tutorial demonstrates three ways to load time-series data from QuestDB into PyTorch:

1. **Direct SQL queries** (REST API or ConnectorX)
2. **Parquet export** via REST API
3. **Direct Parquet partition access** (for local QuestDB)

The example task is volume forecasting, but the data loading patterns apply to any model.

## Quick Start

```bash
# Install dependencies
pip install pandas pyarrow requests torch scikit-learn

# Or with uv:
uv sync

# Run the demos in order:
python steps/method1_direct_query.py   # See direct query methods
python steps/method2_parquet_export.py # Export data to Parquet
python steps/train.py                  # Train the model
```

## What Each Script Does

| Script | Description |
|--------|-------------|
| `steps/method1_direct_query.py` | Queries QuestDB demo instance, shows REST API and ConnectorX |
| `steps/method2_parquet_export.py` | Exports 7 days of data to `dataset/training_data.parquet` |
| `steps/method3_parquet_partitions.py` | Shows direct partition access (requires local QuestDB) |
| `steps/train.py` | Loads Parquet, engineers features, trains LSTM, saves model |

## Project Structure

```
├── config/
│   └── settings.py          # Configuration (URLs, paths, hyperparameters)
├── src/
│   ├── data_loader.py       # Three data loading methods
│   ├── features.py          # Feature engineering
│   └── model.py             # PyTorch Dataset and LSTM model
├── steps/
│   ├── method1_direct_query.py
│   ├── method2_parquet_export.py
│   ├── method3_parquet_partitions.py
│   └── train.py
├── dataset/                 # Exported Parquet files
├── models/                  # Saved model checkpoints
└── pyproject.toml
```

## Data Source

Uses the `trades` table from QuestDB's public demo: `https://demo.questdb.io`

The query aggregates raw trades into 1-minute bars:

```sql
SELECT 
    timestamp,
    symbol,
    sum(amount) as volume,
    count() as trade_count,
    avg(price) as avg_price,
    max(price) - min(price) as price_range
FROM trades
WHERE symbol = 'BTC-USD'
SAMPLE BY 1m
ALIGN TO CALENDAR
```

## Method Comparison

| Method | Best For | Requires |
|--------|----------|----------|
| REST API | Exploration, simple queries | Network access |
| ConnectorX | Larger result sets | `pip install connectorx` |
| Parquet Export | Reproducible training data | Network access |
| Direct Partitions | Terabyte-scale data | Local QuestDB with Parquet format |

## Note on the Model

The LSTM model is intentionally simple—this tutorial focuses on data pipeline patterns, not forecasting accuracy. In production, you'd benchmark against simpler methods (moving averages, ARIMA) and tune hyperparameters properly.
