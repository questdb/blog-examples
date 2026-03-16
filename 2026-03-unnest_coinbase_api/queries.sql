-- ============================================================================
-- UNNEST demo: Coinbase Exchange market data into QuestDB
--
-- Two tables, both storing raw JSON payloads with no client-side parsing.
-- All transformation happens at query time:
--
--   cb_orderbook_raw  ->  json_extract + DOUBLE[][] cast  ->  native array access
--   cb_trades_raw     ->  JSON UNNEST with COLUMNS()      ->  typed rows
-- ============================================================================


-- ---------------------------------------------------------------------------
-- 1. Tables (also created by the Python ingestion script)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cb_orderbook_raw (
    timestamp TIMESTAMP_NS,
    symbol SYMBOL,
    payload VARCHAR
) TIMESTAMP(timestamp) PARTITION BY HOUR;

CREATE TABLE IF NOT EXISTS cb_trades_raw (
    timestamp TIMESTAMP_NS,
    symbol SYMBOL,
    payload VARCHAR
) TIMESTAMP(timestamp) PARTITION BY HOUR;


-- ---------------------------------------------------------------------------
-- 2. Orderbook: json_extract + DOUBLE[][] cast
--
-- The raw payload from Coinbase looks like:
--   {"bids":[["73577.91","0.05",2],...], "asks":[...],
--    "sequence":124211219720, "time":"2026-03-16T15:44:23.376849802Z"}
--
-- json_extract pulls out the bids/asks arrays, then ::DOUBLE[][] gives
-- native array access with direct indexing.
-- ---------------------------------------------------------------------------

-- Ad-hoc: extract and cast on the fly
SELECT
    timestamp,
    symbol,
    json_extract(payload, '$.bids')::DOUBLE[][] AS bids,
    json_extract(payload, '$.asks')::DOUBLE[][] AS asks,
    json_extract(payload, '$.time')::TIMESTAMP AS exchange_time
FROM cb_orderbook_raw;

-- View: orderbook with native arrays
CREATE OR REPLACE VIEW cb_orderbook AS
SELECT
    timestamp,
    symbol,
    json_extract(payload, '$.bids')::DOUBLE[][] AS bids,
    json_extract(payload, '$.asks')::DOUBLE[][] AS asks,
    json_extract(payload, '$.time')::TIMESTAMP AS exchange_time
FROM cb_orderbook_raw;

-- Best bid/ask using array indexing
-- bids[1] is the best (highest) bid level
-- bids[1][1] = price, bids[1][2] = size, bids[1][3] = num_orders
SELECT
    timestamp,
    symbol,
    bids[1][1] AS best_bid,
    bids[1][2] AS best_bid_size,
    asks[1][1] AS best_ask,
    asks[1][2] AS best_ask_size,
    asks[1][1] - bids[1][1] AS spread
FROM cb_orderbook
WHERE symbol = 'BTC-USD';


-- ---------------------------------------------------------------------------
-- 3. Trades: JSON UNNEST with COLUMNS()
--
-- The raw payload from Coinbase is a JSON array of objects:
--   [{"trade_id":123,"price":"82150.30","size":"0.5",
--     "time":"2026-03-16T15:44:23.860Z","side":"sell"}, ...]
--
-- COLUMNS() maps directly to JSON field names, producing typed columns
-- with zero extraction boilerplate.
-- ---------------------------------------------------------------------------

-- Ad-hoc: expand trade batches into individual trades
SELECT
    t.timestamp,
    t.symbol,
    u.trade_id,
    u.price,
    u.size,
    u.side,
    u.time AS trade_time
FROM cb_trades_raw t,
    UNNEST(t.payload COLUMNS(
        trade_id LONG,
        price DOUBLE,
        size DOUBLE,
        side VARCHAR,
        time TIMESTAMP
    )) u
WHERE t.symbol = 'BTC-USD';

-- View: structured trades (may contain duplicates from overlapping polls)
CREATE OR REPLACE VIEW cb_trades AS
SELECT
    t.timestamp,
    t.symbol,
    u.trade_id,
    u.price,
    u.size,
    u.side,
    u.time AS trade_time
FROM cb_trades_raw t,
    UNNEST(t.payload COLUMNS(
        trade_id LONG,
        price DOUBLE,
        size DOUBLE,
        side VARCHAR,
        time TIMESTAMP
    )) u;

-- Materialized view: trades as a QuestDB table
-- Overlapping REST polls mean the same trade_id may appear across batches.
-- This matview converts the UNNEST output into a native table for fast queries.
CREATE MATERIALIZED VIEW cb_trades_mat AS
SELECT
    timestamp,
    symbol,
    trade_id,
    side,
    first(price) AS price,
    first(size) AS size,
    first(trade_time) AS trade_time
FROM cb_trades
SAMPLE BY 1s;

-- View: deduplicated trades (for queries where uniqueness matters)
-- Uses row_number() partitioned by trade_id, keeping only the first occurrence.
-- trade_time becomes the designated timestamp in this view.
CREATE OR REPLACE VIEW cb_trades_dedup AS
WITH ranked AS (
    SELECT *,
        row_number() OVER (PARTITION BY trade_id ORDER BY timestamp) AS rn
    FROM cb_trades_mat
)
SELECT trade_time, timestamp AS batch_timestamp, symbol, trade_id, side, price, size
FROM ranked
WHERE rn = 1
ORDER BY trade_time;
