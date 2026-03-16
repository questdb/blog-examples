# Coinbase Market Data with QuestDB UNNEST

Real-time crypto market data ingestion from the Coinbase Exchange public API
into QuestDB, demonstrating query-time JSON processing:

- **`DOUBLE[][]` cast**: Order book payloads stored as raw JSON, with
  `json_extract` and `::DOUBLE[][]` cast at query time for native array
  indexing (`bids[1][1]` for best bid price).
- **JSON `UNNEST` with `COLUMNS()`**: Trade payloads stored as raw JSON arrays
  of objects, expanded into typed rows with named field mapping and zero
  extraction boilerplate.

No client-side JSON parsing. The Python script stores the raw API response
as a single `VARCHAR` column. All transformation happens in SQL.

## Architecture

```
Coinbase Exchange REST API
    |
    ├── /products/BTC-USD/book?level=2
    |       └── cb_orderbook_raw (payload VARCHAR)
    |               └── VIEW cb_orderbook
    |                     json_extract → DOUBLE[][] cast
    |
    └── /products/BTC-USD/trades
            └── cb_trades_raw (payload VARCHAR)
                    └── VIEW cb_trades
                          UNNEST(payload COLUMNS(...))
```

## Prerequisites

- [QuestDB](https://questdb.io/get-questdb/) running with HTTP enabled
  (port 9000) and PG wire (port 8812)
- Python 3.9+

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Start ingesting:

```bash
python ingest.py
```

The script polls Coinbase Exchange for BTC-USD every 0.75 seconds. Both order
book snapshots and trade batches are stored as raw JSON payloads. Timestamps
are assigned by the QuestDB server at nanosecond resolution.

Tables are created automatically on first run. No API key is needed.

## Queries

Once data is flowing, open the QuestDB Web Console at
[localhost:9000](http://localhost:9000) and try the queries in
[`queries.sql`](queries.sql).

### Order book: native array access

```sql
SELECT
    timestamp,
    symbol,
    bids[1][1] AS best_bid,
    asks[1][1] AS best_ask,
    asks[1][1] - bids[1][1] AS spread
FROM orderbook
WHERE symbol = 'BTC-USD';
```

### Trades: JSON UNNEST

```sql
SELECT t.symbol, u.trade_id, u.price, u.size, u.side, u.time
FROM cb_trades_raw t,
    UNNEST(t.payload COLUMNS(
        trade_id LONG,
        price DOUBLE,
        size DOUBLE,
        side VARCHAR,
        time TIMESTAMP
    )) u;
```

## Configuration

Edit `ingest.py` to change the product or polling interval. Adding more
products requires adjusting the interval to stay within rate limits.

## Rate limits

Coinbase Exchange allows 10,000 public requests per hour. With 1 product and
2 endpoints, polling every 0.75 seconds uses roughly 9,600 requests per hour.

## How it works

1. **Ingestion**: A polling loop hits two Coinbase Exchange REST endpoints
   every 0.75 seconds. The raw JSON response body is stored as-is in a
   `VARCHAR` column. No parsing, no transformation. QuestDB's server assigns
   the timestamp at nanosecond resolution via ILP.

2. **Order book**: The payload contains bids, asks, sequence number, and an
   exchange timestamp. At query time, `json_extract(payload, '$.bids')` pulls
   the array, and `::DOUBLE[][]` casts it to a native 2D array for direct
   indexing.

3. **Trades**: The payload is a JSON array of objects with named fields
   (`trade_id`, `price`, `size`, `side`, `time`). At query time,
   `UNNEST(payload COLUMNS(price DOUBLE, size DOUBLE, ...))` expands each
   batch into typed rows, mapping JSON field names directly to columns.

4. **Views**: `orderbook` and `trades` views apply the transformations.
   Materialized views can aggregate these into BBO snapshots and OHLC candles.
