"""One table, three readers: QuestDB SQL, Polars and DuckDB on the same Parquet.

Runs against the QuestDB container from docker-compose.yml. Every step prints
what it did, and every check compares a reader against the DataFrame the script
sent, so a mismatch stops the run.

    docker compose up -d
    python walkthrough.py
"""
import argparse
import glob
import os
import re
import time
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import polars as pl
import questdb

HERE = os.path.dirname(os.path.abspath(__file__))
DAYS = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]
PARQUET_DAYS = DAYS[:3]
CUTOFF = DAYS[3]
SYMBOLS = {"EURUSD": (1.0850, 5), "GBPUSD": (1.2650, 5), "USDJPY": (151.20, 3)}
TRADES_PER_SYMBOL_PER_DAY = 100_000

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_width_chars(120)
pl.Config.set_tbl_rows(20)


# --- helpers -----------------------------------------------------------------

def step(title):
    print(f"\n=== {title} ===")


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"check failed: {name}")


def sql(db, query):
    return db.query(query).to_polars()


def scalar(db, query):
    return sql(db, query).item()


def wait_for(db, query, expected, what, timeout_s=120):
    """WAL apply and view refresh are asynchronous: poll until they catch up."""
    t0 = time.time()
    while (value := scalar(db, query)) != expected:
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"timed out waiting for {what}: got {value}, want {expected}")
        time.sleep(0.2)
    print(f"  {what}: ready after {time.time() - t0:.1f}s")


def connect(conf, timeout_s=60):
    """Retry while a freshly started container is still booting."""
    t0 = time.time()
    while True:
        try:
            return questdb.connect(conf)
        except questdb.QuestDBError:
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(1)


def daily_stats(frame):
    """Per day and symbol: the numbers every reader must agree on."""
    return (frame.group_by(pl.col("ts").dt.date().cast(pl.String).alias("day"),
                           pl.col("symbol").cast(pl.String))
                 .agg(pl.len().alias("trades"),
                      pl.col("quantity").sum().alias("volume"),
                      pl.col("price").min().alias("low"),
                      pl.col("price").max().alias("high"),
                      ((pl.col("price") * pl.col("quantity")).sum()
                       / pl.col("quantity").sum()).alias("vwap"))
                 .sort("day", "symbol"))


def same_stats(a, b):
    """Counts, volumes, lows and highs must match exactly. VWAP is a ratio of
    floating-point sums, and engines add in different orders, so it gets a
    relative tolerance of 1e-12."""
    if a.height != b.height:
        return False
    a, b = a.sort("day", "symbol"), b.sort("day", "symbol")
    exact = ["day", "symbol", "trades", "volume", "low", "high"]
    if not a.select(exact).cast({"trades": pl.Int64}).equals(
            b.select(exact).cast({"trades": pl.Int64})):
        return False
    rel = ((a["vwap"] - b["vwap"]).abs() / b["vwap"]).max()
    return rel < 1e-12


def partition_files(table_dir, days):
    """Current data.parquet for each day. A rewrite creates a new versioned
    directory (2026-10-20.7 replaces 2026-10-20.3), and the old one lingers
    until QuestDB purges it, so pick the highest version per day."""
    current = {}
    for path in glob.glob(os.path.join(table_dir, "*", "data.parquet")):
        name = os.path.basename(os.path.dirname(path))
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.(\d+)", name)
        if m and m.group(1) in days:
            if m.group(1) not in current or int(m.group(2)) > current[m.group(1)][0]:
                current[m.group(1)] = (int(m.group(2)), path)
    return {day: path for day, (_, path) in sorted(current.items())}


def generate_trades():
    """Seeded synthetic FX trades: a random walk per symbol, 24 hours a day."""
    rng = np.random.default_rng(20261019)
    frames, next_id = [], 1
    for day in DAYS:
        start_us = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) * 1_000_000
        for symbol, (price0, decimals) in SYMBOLS.items():
            n = TRADES_PER_SYMBOL_PER_DAY
            offsets = np.sort(rng.choice(86_400_000_000, n, replace=False))
            tick = 10.0 ** -(decimals - 1)
            walk = price0 + np.cumsum(rng.choice([-1, 0, 1], n)) * tick
            frames.append(pl.DataFrame({
                "ts": pl.Series(start_us + offsets).cast(pl.Datetime("us", "UTC")),
                "symbol": symbol,
                "side": rng.choice(["buy", "sell"], n),
                "price": np.round(walk, decimals),
                "quantity": rng.choice([100_000.0, 250_000.0, 500_000.0,
                                        1_000_000.0, 2_000_000.0, 5_000_000.0], n),
                "trade_id": np.arange(next_id, next_id + n, dtype=np.int64),
            }))
            next_id += n
    return pl.concat(frames).sort("ts", "trade_id")


# --- the walkthrough ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="ws::addr=localhost:9000;",
                    help="QuestDB client configuration string")
    ap.add_argument("--root", default=os.path.join(HERE, "qdb_root"),
                    help="host directory mounted as the QuestDB root")
    ap.add_argument("--reset", action="store_true",
                    help="drop the tables first, to run the walkthrough again")
    args = ap.parse_args()

    with connect(args.conf) as db:
        print(scalar(db, "SELECT build()"))
        if args.reset:
            db.execute("DROP MATERIALIZED VIEW IF EXISTS trades_ohlc_1m")
            db.execute("DROP TABLE IF EXISTS trades")
        existing = scalar(db, "SELECT count() FROM tables() "
                              "WHERE table_name IN ('trades', 'trades_ohlc_1m')")
        if existing:
            raise SystemExit("trades already exists: rerun with --reset")

        step("1. Create the table")
        db.execute("""
            CREATE TABLE trades (
              ts TIMESTAMP,
              symbol SYMBOL,
              side SYMBOL,
              price DOUBLE,
              quantity DOUBLE,
              trade_id LONG
            ) TIMESTAMP(ts) PARTITION BY DAY WAL
            DEDUP UPSERT KEYS(ts, symbol, trade_id)""")
        print("  trades: partitioned by day, WAL, deduplicated on (ts, symbol, trade_id)")

        step("2. Send a Polars DataFrame over QWP")
        trades = generate_trades()
        reference = daily_stats(trades)
        t0 = time.time()
        db.dataframe(trades, table_name="trades", symbols=["symbol", "side"], at="ts")
        print(f"  {trades.height:,} rows acknowledged in {time.time() - t0:.2f}s")
        wait_for(db, "SELECT count() FROM trades", trades.height, "rows visible to SQL")

        step("3. Keep one-minute bars current with a materialized view")
        db.execute("""
            CREATE MATERIALIZED VIEW trades_ohlc_1m AS (
              SELECT ts, symbol,
                     first(price) AS open, max(price) AS high,
                     min(price) AS low, last(price) AS close,
                     sum(quantity) AS volume
              FROM trades
              SAMPLE BY 1m
            ) PARTITION BY DAY""")
        total_volume = float(trades["quantity"].sum())
        wait_for(db, "SELECT sum(volume) FROM trades_ohlc_1m", total_volume,
                 "view caught up with the table")

        daily_sql = """
            SELECT ts, symbol,
                   count() AS trades, sum(quantity) AS volume,
                   min(price) AS low, max(price) AS high,
                   sum(price * quantity) / sum(quantity) AS vwap
            FROM trades
            SAMPLE BY 1d"""

        def daily_from_sql():
            return (sql(db, daily_sql)
                    .with_columns(pl.col("ts").dt.date().cast(pl.String).alias("day"),
                                  pl.col("symbol").cast(pl.String))
                    .drop("ts"))

        native = daily_from_sql()
        check("SQL daily stats over native partitions match the DataFrame sent",
              same_stats(native, reference))

        step(f"4. Convert the days before {CUTOFF} to Parquet")
        db.execute(f"ALTER TABLE trades CONVERT PARTITION TO PARQUET WHERE ts < '{CUTOFF}'")
        wait_for(db, "SELECT count() FROM (SHOW PARTITIONS FROM trades) WHERE isParquet",
                 len(PARQUET_DAYS), "partitions converted")
        print(sql(db, """
            SELECT name, numRows, isParquet, active, diskSizeHuman
            FROM (SHOW PARTITIONS FROM trades)"""))

        step("5. Reader one: the same SQL across both formats")
        mixed = daily_from_sql()
        check("SQL daily stats over Parquet + native partitions match the DataFrame sent",
              same_stats(mixed, reference))
        print(sql(db, f"""
            SELECT symbol,
                   sum(price * quantity) / sum(quantity) AS vwap,
                   count() AS trades
            FROM trades
            WHERE ts >= '{DAYS[1]}' AND ts < '{DAYS[4]}'
            ORDER BY symbol"""))
        print(f"  ^ VWAP over {DAYS[1]}..{DAYS[3]}: two Parquet days and one native day")

        step("6. Readers two and three: Polars and DuckDB open the files directly")
        table_dir = next(p for p in glob.glob(os.path.join(args.root, "db", "trades~*"))
                         if os.path.isdir(p))
        files = partition_files(table_dir, PARQUET_DAYS)
        for day, path in files.items():
            print(f"  {day}: {os.path.relpath(path, HERE)}")
        check("one data.parquet per converted day", list(files) == PARQUET_DAYS)
        from_files = pl.read_parquet(list(files.values()))
        print(f"  schema as Polars reads it: {dict(from_files.schema)}")
        expected = reference.filter(pl.col("day").is_in(PARQUET_DAYS))
        check("Polars over the partition files matches the DataFrame sent",
              same_stats(daily_stats(from_files), expected))

        con = duckdb.connect()
        file_list = ", ".join(f"'{p}'" for p in files.values())
        duck = con.sql(f"""
            SELECT strftime(ts AT TIME ZONE 'UTC', '%Y-%m-%d') AS day, symbol,
                   count(*) AS trades, sum(quantity) AS volume,
                   min(price) AS low, max(price) AS high,
                   sum(price * quantity) / sum(quantity) AS vwap
            FROM read_parquet([{file_list}])
            GROUP BY ALL ORDER BY day, symbol""").pl()
        check("DuckDB over the same files matches the DataFrame sent",
              same_stats(duck, expected))

        wire = sql(db, f"SELECT * FROM trades WHERE ts IN '{DAYS[0]}' ORDER BY ts, trade_id")
        file_day = pl.read_parquet(files[DAYS[0]]).sort("ts", "trade_id")
        check(f"QWP query into Polars equals the {DAYS[0]} file, row for row",
              wire.with_columns(pl.col("symbol", "side").cast(pl.String))
                  .equals(file_day.with_columns(pl.col("symbol", "side").cast(pl.String))))

        step("7. A late trade and a replay, into a day that is already Parquet")
        late_ts = datetime.fromisoformat(f"{DAYS[1]}T15:00:00.5+00:00")
        before = partition_files(table_dir, [DAYS[1]])[DAYS[1]]
        before_size = os.path.getsize(before)
        with db.sender() as s:
            s.row("trades", symbols={"symbol": "EURUSD", "side": "buy"},
                  columns={"price": 1.0900, "quantity": 7_000_000.0, "trade_id": 900_000_001},
                  at=late_ts)
        wait_for(db, "SELECT count() FROM trades", trades.height + 1, "late trade visible")
        after = partition_files(table_dir, [DAYS[1]])[DAYS[1]]
        print(f"  before: {os.path.relpath(before, table_dir)} ({before_size:,} bytes)")
        print(f"  after:  {os.path.relpath(after, table_dir)} ({os.path.getsize(after):,} bytes)")
        check(f"{DAYS[1]} is still Parquet",
              scalar(db, f"SELECT isParquet FROM (SHOW PARTITIONS FROM trades) "
                         f"WHERE name = '{DAYS[1]}'"))
        check("the rewritten file holds the late trade",
              pl.read_parquet(after).filter(pl.col("trade_id") == 900_000_001).height == 1)
        bar_before = float(trades.filter(
            (pl.col("symbol") == "EURUSD")
            & (pl.col("ts").dt.truncate("1m") == late_ts.replace(second=0, microsecond=0))
        )["quantity"].sum())
        wait_for(db, f"SELECT volume FROM trades_ohlc_1m WHERE symbol = 'EURUSD' "
                     f"AND ts = '{DAYS[1]}T15:00:00.000000Z'",
                 bar_before + 7_000_000.0, "15:00 EURUSD bar includes the late trade")

        replay = trades.filter(pl.col("ts").dt.date().cast(pl.String) == DAYS[2]).head(1)
        txn = scalar(db, "SELECT max(sequencerTxn) FROM wal_transactions('trades')")
        db.dataframe(replay, table_name="trades", symbols=["symbol", "side"], at="ts")
        wait_for(db, f"SELECT max(sequencerTxn) > {txn} FROM wal_transactions('trades')",
                 True, "replay committed")
        wait_for(db, f"SELECT writerTxn = sequencerTxn FROM wal_tables() "
                     f"WHERE name = 'trades'", True, "replay applied")
        check("the replayed trade did not add a row",
              scalar(db, "SELECT count() FROM trades") == trades.height + 1)

        step("8. A stable copy for tools outside QuestDB")
        export_id = scalar(db, "COPY trades TO 'trades_export' WITH FORMAT PARQUET")
        export_dir = os.path.join(args.root, "export", "trades_export")
        wait_for(db, f"SELECT count() FROM sys.copy_export_log "
                     f"WHERE id = '{export_id}' AND phase = 'success'",
                 1, "export finished")
        exported = sorted(glob.glob(os.path.join(export_dir, "*.parquet")))
        for path in exported:
            print(f"  {os.path.relpath(path, HERE)}")
        check("the export holds every row, late trade included",
              pl.scan_parquet(exported).select(pl.len()).collect().item() == trades.height + 1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
