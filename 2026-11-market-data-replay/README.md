# Replaying market data events from QuestDB

Companion code for the QuestDB blog post on replaying market data events.

The example loads one full trading day of US equities top-of-book quotes
(`mbp-1`) and trades from Databento's `EQUS.MINI` dataset into QuestDB, using
the Databento connector in Historical mode.

## What's inside

| File | Purpose |
| --- | --- |
| `backfill_day.py` | Loads one full trading day from Databento's Historical API into QuestDB |

`backfill_day.py` uses only the Python standard library.

## Requirements

- QuestDB Enterprise 4.0.0 or later. The connector writes with the QuestDB
  Wire Protocol (QWP) over WebSocket on port `9000`.
- The `databento-questdb` connector binary on your `PATH`, or set
  `DATABENTO_QUESTDB_BIN`. Get it through your QuestDB Enterprise contact; see
  [installing the connector](https://questdb.com/docs/connect/connectors/databento/#1-install-the-connector).
- A QuestDB service account token with `HTTP` and `CREATE TABLE` grants; see
  [authentication and TLS](https://questdb.com/docs/connect/connectors/databento/#questdb-authentication-and-tls).
- A Databento API key with Historical access to `EQUS.MINI`.
- Python 3.9 or later.

## Load a trading day

```bash
export DATABENTO_API_KEY='db-your-key-here'
export QDB_CLIENT_CONF='ws::addr=localhost:9000;token=<your-token>;'

python3 backfill_day.py                  # NVDA and AAPL, 2 to 3 million rows
python3 backfill_day.py --all-symbols    # every symbol, 350 to 450 million rows
```

The script picks the latest full trading day, skipping weekends, market
holidays and early-close days, and covers the whole `EQUS.MINI` session from
04:00 to 20:00 New York time.

Before downloading, it prints the rows, size and cost of the day from
Databento's metadata endpoints, which are not billed. If Databento would bill
the download, it asks first; `--yes` skips the question. `--dry-run` stops
after the estimate.

A full day of every symbol is about 30 GB from Databento and takes about
75 GiB in QuestDB, plus the same again for staging tables while it loads, so
run it against a server with room for it.

| Option | Effect |
| --- | --- |
| `--symbols AMZN,MSFT` | Symbols to load (default: `NVDA,AAPL`) |
| `--all-symbols` | Every symbol in `EQUS.MINI` |
| `--date 2026-10-23` | A specific trading day instead of the latest |
| `--table-prefix equs` | Tables are `<prefix>_orderbook` and `<prefix>_trades` (default: `equs_hist`) |
| `--batch-size 2000` | Symbols per Databento request, at most 2,000 (default: 2000) |
| `--split-batch 5` | Rerun batch 5 as smaller requests of `--split-size` symbols (default: 250), replacing its partial staging tables |
| `--merge-minutes 10` | Time window per merge statement for batched loads (default: 10) |
| `--bin PATH` | Connector binary, when it is not on `PATH` |
| `--tls-skip-verify` | Accept QuestDB's TLS certificate without verifying it, for test servers with a self-signed certificate |

`--tls-skip-verify` adds `tls_verify=unsafe_off` to `QDB_CLIENT_CONF`. The
connector accepts it only when built with the QuestDB client's
`insecure-skip-verify` feature:

```bash
cargo build --release --features questdb-rs/insecure-skip-verify
```

For servers with a private CA, verify instead: add `tls_roots=/path/to/ca.pem;`
to `QDB_CLIENT_CONF`.

## How it works

The script writes a Historical connector configuration to
`state/<prefix>-<day>-<symbols>/connector.toml` and runs the connector, which
exits when the day is loaded. Credentials stay in the environment and never
reach the generated file.

| Table | Schema | Contents |
| --- | --- | --- |
| `equs_hist_orderbook` | `mbp-1` | Every change to the best bid or offer, plus trade events |
| `equs_hist_trades` | `trades` | Trades only |

### Loading every symbol

Databento streams a request for an explicit list of symbols far faster than
one for `ALL_SYMBOLS`, and accepts up to 2,000 symbols per request. So with
`--all-symbols`, or any list longer than `--batch-size`, the script:

1. Fetches the day's symbols from Databento's `definition` schema, about
   13,000 for `EQUS.MINI`.
2. Downloads them in parallel batches, one connector task and one staging
   table per batch, `<prefix>_stageNN_orderbook` and `<prefix>_stageNN_trades`.
   Each Databento request returns rows in time order, so each staging table is
   written in order.
3. Merges the staging tables into the final tables one time window at a time,
   in order, with `INSERT INTO ... SELECT ... ORDER BY ts_recv`. Every window
   lands after the previous one, so QuestDB only appends and never rewrites a
   partition for out-of-order data.
4. Checks that the final row count matches the staging tables, then drops
   them.

### Tables

Tables are partitioned by day, with nanosecond
`ts_event` and `ts_recv` columns and `ts_recv` as the designated timestamp.
Book levels are stored as `DOUBLE[][]` arrays of shape `[2, N]`, best level
first: row 1 holds prices and row 2 sizes. `bids[1][1]` is the best bid price
and `bids[2][1]` its size. An empty side of the book is `NULL`.

## Interruptions

Ctrl+C stops the connector cleanly, and running the same command again
resumes. The connector downloads `mbp-1` first, then `trades`, and skips
whichever is already complete. A query that was interrupted starts again
from 04:00, so its rows from the first attempt are written twice. For a
clean load after an interruption, drop the tables it writes, including any
staging tables, and delete the run's `state/<prefix>-<day>-<symbols>/`
directory before running again.

For batched loads, an interrupted merge resumes after the last merged window
when you run the same command again. Completed downloads are not repeated.

Databento returns a request's rows in time order across all its symbols, so
one slow symbol can hold back a whole batch. If one batch lags far behind the
others, stop the run and rerun it with `--split-batch N`. The script drops
that batch's partial staging tables and checkpoint and downloads its symbols
as smaller parallel requests; finished batches are skipped.
