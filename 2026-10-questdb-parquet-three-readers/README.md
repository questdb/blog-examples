# One table, three readers: QuestDB, Polars and DuckDB on the same Parquet

Companion code for the QuestDB blog post
[One table, three readers: QuestDB, Polars and DuckDB on the same Parquet](https://questdb.com/blog/questdb-parquet-polars-duckdb-one-table-three-readers/).

The walkthrough loads 1.5 million synthetic FX trades into QuestDB from a Polars
DataFrame, keeps one-minute bars current with a materialized view, converts the
three oldest days to Parquet, and then reads that history three ways: with
QuestDB SQL, with Polars and with DuckDB opening the partition files directly.
Every reader is checked against the DataFrame that was sent. The run stops at
the first mismatch.

Everything here runs on QuestDB open source. No licence, no market data
subscription, and no Java toolchain: QuestDB runs in Docker.

## What's inside

| File | Purpose |
| --- | --- |
| `walkthrough.py` | The eight steps below, with a check after each one |
| `requirements.txt` | Pinned Python dependencies |

## Requirements

- Docker
- Python 3.10 or later

## Quick start

```bash
docker run -d --name qdb-parquet-readers -p 9000:9000 \
  -v "$PWD/qdb_root:/var/lib/questdb" questdb/questdb:10.1.0

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python walkthrough.py
```

The run takes a few seconds and ends with `All checks passed.` To run it
again, add `--reset`, which drops the two tables first.

Open the Web Console at http://localhost:9000 to query `trades` and
`trades_ohlc_1m` yourself.

## What the walkthrough does

1. Creates `trades`, partitioned by day, with WAL and
   `DEDUP UPSERT KEYS(ts, symbol, trade_id)`.
2. Sends five days of trades, 300,000 a day, from a Polars DataFrame over the
   QuestDB Wire Protocol (QWP), and waits until SQL sees every row.
3. Creates the `trades_ohlc_1m` materialized view and waits for it to catch up.
   Checks daily statistics from SQL against the DataFrame.
4. Runs `ALTER TABLE trades CONVERT PARTITION TO PARQUET WHERE ts < '2026-10-22'`
   and prints `SHOW PARTITIONS`.
5. Runs the same SQL over three Parquet days and two native days, and checks it
   again.
6. Opens the three `data.parquet` files under `qdb_root/db/trades~N/` with
   Polars and with DuckDB, and checks both. Then checks that a QWP query into
   Polars returns exactly the rows in one of the files.
7. Sends a late trade into a day that is already Parquet, and replays an
   existing trade into another. The late trade shows up in SQL, in the file and
   in the one-minute bar. The replay adds nothing.
8. Exports the table with `COPY trades TO 'trades_export' WITH FORMAT PARQUET`
   and checks the export holds every row.

The checks compare row counts, volumes, lows and highs exactly, and VWAP to a
relative tolerance of 1e-12, because engines add floating point numbers in
different orders.

## Reading the files yourself

After a run, the converted partitions are ordinary Parquet files on your disk:

```python
import polars as pl

trades = pl.read_parquet("qdb_root/db/trades~*/2026-10-19.*/data.parquet")
```

```sql
-- DuckDB
SELECT symbol, count(*), sum(price * quantity) / sum(quantity) AS vwap
FROM read_parquet('qdb_root/db/trades~*/2026-10-*/data.parquet')
GROUP BY symbol;
```

These files belong to a live table. QuestDB rewrites them when late rows
arrive, and it can replace one with a compacted copy in a new versioned
directory, such as `2026-10-21.10` replacing `2026-10-21.4`. The old directory
lingers until QuestDB purges it, so a glob over `2026-10-21.*` can briefly match
both. That's fine for exploring, but build pipelines on the `COPY` export in
`qdb_root/export/trades_export/`, or on QuestDB Enterprise cold storage, which
writes a stable, Hive-partitioned layout to object storage.

## Clean up

```bash
docker rm -f qdb-parquet-readers
rm -rf qdb_root
```

On Linux the container writes `qdb_root` as UID 10001, so the `rm` needs
`sudo`.

## Tested with

- QuestDB 10.1.0 (`questdb/questdb:10.1.0`) on Docker Desktop, macOS on Apple
  Silicon
- Python 3.14 with the versions in `requirements.txt`
