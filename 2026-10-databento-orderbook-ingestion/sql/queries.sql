-- Queries over the tables the connector creates.
-- Run with: python3 run_sql.py sql/queries.sql [name ...]

-- name: row_counts
-- How much arrived, and over which time range.
SELECT 'equs_orderbook' AS table_name, count() AS row_count,
       min(ts_recv) AS first_ts, max(ts_recv) AS last_ts,
       count_distinct(symbol) AS symbols
FROM equs_orderbook
UNION ALL
SELECT 'equs_trades', count(), min(ts_recv), max(ts_recv), count_distinct(symbol)
FROM equs_trades;

-- name: latest_quotes
-- Current top of book per symbol. Book levels are DOUBLE[][] arrays of shape
-- [2, N]: row 1 holds prices, row 2 sizes, best level first.
SELECT ts_recv, symbol,
       bids[1][1] AS bid_px, bids[2][1] AS bid_sz,
       asks[1][1] AS ask_px, asks[2][1] AS ask_sz
FROM equs_orderbook
WHERE symbol IN ('AAPL', 'MSFT', 'NVDA', 'TSLA', 'SPY', 'QQQ')
LATEST ON ts_recv PARTITION BY symbol;

-- name: most_traded
-- Most traded symbols by notional value.
SELECT symbol, count() AS trades, sum(size) AS shares,
       sum(price * size) AS notional
FROM equs_trades
ORDER BY notional DESC
LIMIT 10;

-- name: ohlcv_1m
-- One-minute bars per symbol, built from raw trades.
SELECT ts_recv, symbol,
       first(price) AS open, max(price) AS high,
       min(price) AS low, last(price) AS close,
       sum(size) AS volume, count() AS trades
FROM equs_trades
SAMPLE BY 1m
ORDER BY trades DESC
LIMIT 10;

-- name: spread_bps
-- Average quoted spread in basis points of the mid price, per minute.
SELECT ts_recv, symbol,
       avg((asks[1][1] - bids[1][1]) / ((asks[1][1] + bids[1][1]) / 2) * 10000) AS spread_bps,
       count() AS quote_updates
FROM equs_orderbook
WHERE symbol IN ('AAPL', 'NVDA', 'TSLA', 'SPY', 'QQQ')
  AND bids[1][1] > 0 AND asks[1][1] > 0
SAMPLE BY 1m;

-- name: trades_vs_quote
-- Each trade next to the quote prevailing when it printed. ASOF JOIN matches
-- the latest quote at or before the trade's ts_recv for the same symbol. An
-- empty side of the book is NULL, so one-sided quotes get their own label.
SELECT t.ts_recv, t.symbol, t.price, t.size,
       q.bids[1][1] AS bid_px, q.asks[1][1] AS ask_px,
       CASE
           WHEN q.bids[1][1] = NULL OR q.asks[1][1] = NULL THEN 'no two-sided quote'
           WHEN t.price >= q.asks[1][1] THEN 'at or above ask'
           WHEN t.price <= q.bids[1][1] THEN 'at or below bid'
           ELSE 'inside spread'
       END AS trade_location
FROM equs_trades t
ASOF JOIN equs_orderbook q ON (symbol)
LIMIT -10;

-- name: arrival_timeline
-- Rows per 30 seconds. After a restart with the same state directory, the
-- buckets covering the downtime fill in from Databento intraday replay.
SELECT ts_recv, count() AS book_updates
FROM equs_orderbook
SAMPLE BY 30s;

-- name: duplicate_check
-- Rows that are identical in every field. Recovery replays from slightly
-- before the last checkpoint; replay_dedup removes those replayed rows. A
-- few identical records can still be genuine, such as two equal-sized fills
-- at the same price in the same nanosecond.
SELECT 'equs_orderbook' AS table_name, count() AS row_count,
       count() - count_distinct(concat(ts_recv::long, '|', instrument_id, '|',
           ts_event::long, '|', action, '|', side, '|', price, '|', size)) AS identical_rows
FROM equs_orderbook
UNION ALL
SELECT 'equs_trades', count(),
       count() - count_distinct(concat(ts_recv::long, '|', instrument_id, '|',
           ts_event::long, '|', side, '|', price, '|', size))
FROM equs_trades;

-- name: spread_distribution
-- Spread distribution in basis points for the most actively quoted symbols,
-- using QuestDB's native spread() and mid() functions. Precision 3 keeps
-- approx_percentile accurate for small values.
SELECT symbol,
       min(spread(bids[1][1], asks[1][1]) / mid(bids[1][1], asks[1][1]) * 10000) AS min_bps,
       approx_percentile(spread(bids[1][1], asks[1][1]) / mid(bids[1][1], asks[1][1]) * 10000, 0.5, 3) AS p50_bps,
       max(spread(bids[1][1], asks[1][1]) / mid(bids[1][1], asks[1][1]) * 10000) AS max_bps,
       count() AS updates
FROM equs_orderbook
WHERE bids[1][1] > 0 AND asks[1][1] > 0
ORDER BY updates DESC
LIMIT 8;

-- name: event_to_receive
-- Gap between each record's event timestamp and Databento's receive
-- timestamp, in microseconds. Both are stored with nanosecond precision.
SELECT approx_percentile((ts_recv - ts_event) / 1000.0, 0.5, 3) AS p50_us,
       approx_percentile((ts_recv - ts_event) / 1000.0, 0.99, 3) AS p99_us,
       approx_percentile((ts_recv - ts_event) / 1000.0, 0.999, 3) AS p999_us,
       max((ts_recv - ts_event) / 1000.0) AS max_us,
       count() AS updates
FROM equs_orderbook
WHERE ts_recv >= ts_event;

-- name: burstiness
-- Per-second update rates. Averages hide bursts; size buffers for the peak.
SELECT max(n) AS peak_per_second, avg(n) AS avg_per_second,
       approx_percentile(n, 0.99, 3) AS p99_per_second, count() AS seconds
FROM (SELECT ts_recv, count() AS n FROM equs_orderbook SAMPLE BY 1s);

-- name: busiest_milliseconds
-- The busiest single milliseconds, using SAMPLE BY 1T (one millisecond).
SELECT ts_recv, count() AS updates, count_distinct(symbol) AS symbols
FROM equs_orderbook
SAMPLE BY 1T
ORDER BY updates DESC
LIMIT 5;

-- name: trade_location_mix
-- Where trades print relative to the prevailing quote, across all symbols.
WITH located AS (
    SELECT t.size,
           CASE
               WHEN q.bids[1][1] = NULL OR q.asks[1][1] = NULL THEN 'no two-sided quote'
               WHEN t.price >= q.asks[1][1] THEN 'at or above ask'
               WHEN t.price <= q.bids[1][1] THEN 'at or below bid'
               ELSE 'inside spread'
           END AS trade_location
    FROM equs_trades t
    ASOF JOIN equs_orderbook q ON (symbol)
)
SELECT trade_location, count() AS trades, sum(size) AS shares
FROM located
ORDER BY trades DESC;
