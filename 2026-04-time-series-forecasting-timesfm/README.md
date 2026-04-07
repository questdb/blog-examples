# QuestDB → PyTorch: Time-Series Data Loading & Forecasting

This tutorial demonstrates how to efficiently load time-series data from QuestDB into Python, then run forecasts using Google's TimesFM foundation model.

**Focus**: Data pipeline patterns. The forecasting is just to show the complete workflow.

## What's Inside

| File | Purpose |
|------|---------|
| `steps/method1_direct_query.py` | Load data via REST API and ConnectorX |
| `steps/method2_parquet_export.py` | Export data to Parquet files |
| `steps/method3_parquet_partitions.py` | Read QuestDB partitions directly |
| `steps/forecast.py` | Run TimesFM forecasts on volume & volatility |

## The Use Case

We forecast two operationally useful metrics from crypto trade data:

1. **Volume** – When will liquidity be available? (execution timing)
2. **Volatility** – How much is price moving? (risk management)

We intentionally avoid price prediction—forecasting "where will BTC be in an hour" isn't credible. Volume and volatility forecasting, however, serve real needs.

## Quick Start

> **Requires Python 3.10+**

```bash
# 1. Clone and setup
git clone https://github.com/questdb/blog-examples.git
cd blog-examples/2025-pytorch-questdb-timesfm

# 2. Create environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install base dependencies
pip install pandas pyarrow requests torch

# 4. Install TimesFM from source (pip package is outdated)
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e ".[torch]"
cd ..

# 5. Run the pipeline
python steps/method1_direct_query.py   # Test QuestDB connection
python steps/method2_parquet_export.py # Export data to Parquet
python steps/forecast.py               # Run forecasts
```

## Data Source

Uses the `trades` table from QuestDB's public demo: `https://demo.questdb.io`

Raw trades are aggregated into 1-minute OHLCV bars:

```sql
SELECT 
    timestamp,
    first(price) as open,
    max(price) as high,
    min(price) as low,
    last(price) as close,
    sum(amount) as volume,
    count() as trade_count
FROM trades
WHERE symbol = 'BTC-USDT'
SAMPLE BY 1m
ALIGN TO CALENDAR
```

## Three Data Loading Methods

| Method | Best For | How |
|--------|----------|-----|
| **REST API** | Exploration, small data | `requests.get("/exp?fmt=csv")` |
| **ConnectorX** | Large results (local QuestDB) | PostgreSQL wire protocol |
| **Parquet Export** | Reproducible pipelines | `requests.get("/exp?fmt=parquet")` |
| **Direct Partitions** | Terabyte-scale | Read QuestDB's Parquet files directly |

> **Note**: ConnectorX requires PostgreSQL port access (8812). The public demo only exposes REST API, so ConnectorX works with local/self-hosted QuestDB only.

## Project Structure

```
├── config/settings.py       # Configuration (URLs, symbols, params)
├── src/
│   ├── data_loader.py       # Three data loading methods
│   └── forecaster.py        # TimesFM wrapper
├── steps/
│   ├── method1_direct_query.py
│   ├── method2_parquet_export.py
│   ├── method3_parquet_partitions.py
│   └── forecast.py
├── dataset/                 # Exported Parquet files
├── outputs/                 # Forecast results (CSV)
└── BLOG_POST.md             # Full tutorial writeup
```

## Notes

- **Python 3.10+ required** – TimesFM doesn't support older versions
- **TimesFM must be installed from source** – The PyPI package is outdated
- **Model downloads ~900MB on first run** – Cached in `~/.cache/huggingface/`
- **GPU optional** – CPU inference works fine, just slower

## Honest Caveats

1. **Foundation models aren't magic.** TimesFM gives a quick baseline, but tuned domain-specific models may perform better.

2. **Volume is hard to forecast.** It's driven by news, large orders, and sentiment—all unpredictable. Wide confidence intervals are expected.

3. **This is a data pipeline tutorial.** Whether these forecasts are useful for your application requires proper backtesting.
