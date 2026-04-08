# QuestDB → TimesFM: time-series data loading & zero-shot forecasting

Companion code for the blog post at
https://questdb.com/blog/zero-shot-forecasting-questdb-timesfm/

This tutorial demonstrates four ways to load time-series data from QuestDB
into Python, then runs zero-shot forecasts using Google's TimesFM foundation
model on BTC-USDT trading volume and volatility.

**Focus**: data pipeline patterns. The forecasting is just there to show the
complete workflow end to end.

## What's inside

| File | Purpose |
|------|---------|
| `steps/method1_direct_query.py` | Load data via REST API and ConnectorX |
| `steps/method2_parquet_export.py` | Export data to Parquet via REST |
| `steps/method3_parquet_partitions.py` | Read QuestDB's Parquet partitions directly |
| `steps/forecast.py` | Run TimesFM forecasts on volume & volatility |
| `steps/visualize.py` | Render fan charts (PNG) from the forecast CSVs |
| `src/data_loader.py` | The four loading functions used by the steps |
| `src/forecaster.py` | TimesFM wrapper |
| `config/settings.py` | URLs, symbol, model parameters |

## The use case

We forecast two operationally useful metrics from crypto trade data:

1. **Volume** - when will liquidity be available? (execution timing)
2. **Volatility** - how much is price moving? (risk management)

We intentionally avoid price prediction. Forecasting "where will BTC be in
an hour" is not credible. Volume and volatility forecasting, however, serve
real needs.

## Quick start

> Requires Python 3.10+ (TimesFM requirement).

```bash
# 1. Clone and enter the example
git clone https://github.com/questdb/blog-examples.git
cd blog-examples/2026-04-time-series-forecasting-timesfm

# 2. Create a virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install base dependencies
pip install -r requirements.txt

# 4. Install TimesFM from source (the PyPI package is outdated)
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e ".[torch]"
cd ..

# 5. Run the pipeline
python steps/method1_direct_query.py    # Test the QuestDB connection
python steps/method2_parquet_export.py  # Export 7 days of data to Parquet
python steps/forecast.py                # Run TimesFM forecasts (downloads ~900MB on first run)
python steps/visualize.py               # Render fan charts to outputs/*.png
```

## Data source

Uses the `trades` table from QuestDB's public demo:
[demo.questdb.io](https://demo.questdb.io)

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
    AND timestamp IN '$now - 7d..$now'
SAMPLE BY 1m;
```

## The four data loading methods

The four methods are not sorted by data size - they are sorted by **how much
of the QuestDB engine you use** and **how the data flows into your training
loop**.

| Method | Uses query engine | SQL transforms | Includes latest data | Reusable across runs | Natural fit |
|--------|-------------------|----------------|----------------------|----------------------|-------------|
| REST API          | Yes | Yes | Yes              | No (re-query)       | Exploration, notebooks               |
| ConnectorX        | Yes | Yes | Yes              | No (re-query)       | Arrow-native Python workflows        |
| Parquet export    | Yes | Yes | Yes (at export)  | Yes (file on disk)  | Training pipelines, reproducibility  |
| Direct partitions | No  | No  | No (converted only) | Yes (existing files) | Lakehouse-style ML, Spark/Dask/Ray |

The first three all use QuestDB's query engine and support arbitrary SQL
(filters, joins, aggregations, `SAMPLE BY`, window joins). REST is the
lowest-friction option but you parse text and handle pagination yourself.
ConnectorX talks PostgreSQL wire and returns Arrow/pandas directly with no
manual paging - the right default beyond exploration. Parquet export still
runs the query once but produces a file you can iterate over across many
training runs.

The fourth method bypasses the query engine entirely and reads QuestDB's
Parquet partitions directly, lakehouse-style. Trade-offs: no SQL transforms
at read time, and the active partition is never in Parquet, so this method
always lags real time.

> **Note**: ConnectorX needs the PostgreSQL wire protocol (port 8812). The
> public demo instance does not expose it externally, so ConnectorX works
> against local or self-hosted QuestDB only. Method 1's REST path uses the
> demo by default; ConnectorX falls back to a clear error message.

## Project structure

```
├── config/settings.py             # Configuration (URLs, symbols, params)
├── src/
│   ├── data_loader.py             # All four loading methods
│   └── forecaster.py              # TimesFM wrapper
├── steps/
│   ├── method1_direct_query.py
│   ├── method2_parquet_export.py
│   ├── method3_parquet_partitions.py
│   ├── forecast.py
│   └── visualize.py
├── dataset/                       # Exported Parquet files (gitignored)
├── outputs/                       # Forecast CSVs and chart PNGs (gitignored)
└── requirements.txt
```

## Notes

- **Python 3.10+ required.** TimesFM does not support older versions.
- **TimesFM must be installed from source.** The PyPI package is outdated.
- **Model downloads ~900MB on first run.** Cached in `~/.cache/huggingface/`.
- **GPU optional.** CPU inference works fine, just slower.

## Honest caveats

1. **Foundation models are not magic.** TimesFM gives a quick baseline, but a
   tuned domain-specific model may perform better.
2. **Volume is hard to forecast.** It is driven by news, large orders, and
   sentiment - all unpredictable. Wide confidence intervals are expected.
3. **This is a data pipeline tutorial.** Whether these forecasts are useful
   for your application requires proper backtesting.
