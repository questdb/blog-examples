#!/usr/bin/env python3
"""Replay market data from QuestDB as fast as possible, in order.

Streams rows from one or more tables over QuestDB's QWP protocol as Arrow
batches and hands each batch to publish(). Each stream runs on one connection
in its own thread, because queries on a connection run strictly one at a time.

A stream is a sequence of consecutive one-hour time windows rather than one long
query, so that no query outlives the server's query.timeout (60 seconds by
default). Windows run in order and do not overlap, and rows arrive in timestamp
order within each one, so the whole stream is in timestamp order.

By default each table is one stream, which keeps that table in strict timestamp
order. --split-by symbol --splits N trades global order for throughput: it lists
the symbols with LATEST ON, divides them into N groups, and runs N streams per
table. Order then holds within each group, so per symbol, but not across groups.

    QDB_CLIENT_CONF  QuestDB client configuration string

Usage:
    python3 replay.py equs_all_orderbook equs_all_trades
    python3 replay.py equs_all_orderbook --limit 1000000
    python3 replay.py equs_all_orderbook --start 2026-10-27T13:30:00Z --end 2026-10-27T14:00:00Z
    python3 replay.py equs_all_orderbook equs_all_trades --split-by symbol --splits 8

Requires the questdb client (5.0 or later), pyarrow and numpy.
"""

import argparse
import dataclasses
import os
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import questdb

NANOS = 1_000_000_000


def ns_to_literal(ns):
    """Epoch nanoseconds as a SQL timestamp literal with nanosecond precision."""
    sec, frac = divmod(int(ns), NANOS)
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:09d}Z"


def quoted(value):
    return "'" + value.replace("'", "''") + "'"


def to_ns(column):
    """A timestamp Arrow array as int64 epoch nanoseconds."""
    return column.cast(pa.timestamp("ns", tz=column.type.tz)).cast(pa.int64()).to_numpy(zero_copy_only=False)


@dataclasses.dataclass
class Stream:
    """One connection's replay of a table or a group, with its running totals."""

    table: str
    select: str  # SELECT ... WHERE, ready for the time window to be appended
    first_ts: int
    last_ts: int
    limit: int = 0
    rows: int = 0
    bytes: int = 0
    windows: int = 0
    out_of_order: int = 0
    position: int = 0
    done: bool = False


def publish(table, batch):
    """Hand a batch of rows to whatever consumes the replay.

    This is the integration point: send to Kafka, feed a strategy, update a
    book. Called from the stream's thread with a pyarrow.RecordBatch, in
    timestamp order for that stream. Whatever runs here sets the pace, so a
    slow consumer slows its stream down through QWP flow control.
    """


def replay_window(reader, stream, ts_col, lo, hi, stop):
    """Run one time window and publish its rows. Returns False to end the stream."""
    sql = f"{stream.select} {ts_col} >= '{ns_to_literal(lo)}' AND {ts_col} < '{ns_to_literal(hi)}'"
    # Every window reads the same SYMBOL columns, so keep the dictionary warm.
    with reader.query(sql, reset_symbol_dict=False) as result:
        for batch in result.iter_arrow():
            if batch.num_rows == 0:  # a result can open with a schema-only batch
                continue
            if stream.limit and stream.rows + batch.num_rows >= stream.limit:
                batch = batch.slice(0, stream.limit - stream.rows)
            timestamps = to_ns(batch.column(ts_col))
            stream.out_of_order += int(np.count_nonzero(np.diff(timestamps) < 0))
            stream.out_of_order += int(stream.position > timestamps[0])
            stream.position = int(timestamps[-1])
            publish(stream.table, batch)
            stream.rows += batch.num_rows
            stream.bytes += batch.nbytes
            if stop.is_set() or stream.rows == stream.limit:
                result.cancel()
                return False
    return True


def run_stream(stream, conf, ts_col, window_ns, stop, errors):
    """Replay one stream's time windows in order on a single connection."""
    try:
        with questdb.connect(conf) as db, db.reader() as reader:
            lo = stream.first_ts
            while lo <= stream.last_ts:
                hi = lo + window_ns
                if not replay_window(reader, stream, ts_col, lo, hi, stop):
                    return
                stream.windows += 1
                lo = hi
    except Exception as e:  # noqa: BLE001
        errors.append(f"{stream.table}: {e}")
        stop.set()
    finally:
        stream.done = True


def time_bounds(db, table, args):
    """First and last timestamp in range as nanoseconds, or None for an empty range."""
    clauses = []
    if args.start:
        clauses.append(f"{args.ts} >= {quoted(args.start)}")
    if args.end:
        clauses.append(f"{args.ts} < {quoted(args.end)}")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    bounds = db.query(f"SELECT min({args.ts}) lo, max({args.ts}) hi FROM {table}{where}").to_arrow()
    if bounds.num_rows == 0 or bounds.column(0).null_count:
        return None
    return int(to_ns(bounds.column(0).combine_chunks())[0]), int(to_ns(bounds.column(1).combine_chunks())[0])


def split_groups(db, table, args, lo, hi):
    """Every distinct value of the split column, dealt into at most --splits groups."""
    sql = (f"SELECT {args.split_by} FROM {table} "
           f"WHERE {args.ts} >= '{ns_to_literal(lo)}' AND {args.ts} <= '{ns_to_literal(hi)}' "
           f"LATEST ON {args.ts} PARTITION BY {args.split_by}")
    values = sorted(v for v in db.query(sql).to_arrow().column(0).to_pylist() if v is not None)
    return [group for group in (values[i::args.splits] for i in range(args.splits)) if group]


def build_streams(db, args):
    streams = []
    for table in args.tables:
        bounds = time_bounds(db, table, args)
        if bounds is None:
            print(f"{table}: no rows in that range")
            continue
        lo, hi = bounds
        if args.split_by:
            groups = split_groups(db, table, args, lo, hi)
            limit = max(args.limit // len(groups), 1) if args.limit else 0
            for group in groups:
                values = ", ".join(quoted(v) for v in group)
                select = f"SELECT * FROM {table} WHERE {args.split_by} IN ({values}) AND"
                streams.append(Stream(table, select, lo, hi, limit))
        else:
            streams.append(Stream(table, f"SELECT * FROM {table} WHERE", lo, hi, args.limit or 0))
    return streams


def reporter(streams):
    """Return a function that prints one line per table: position, rows and rates."""
    last = {s.table: (0, 0) for s in streams}
    last_time = time.monotonic()

    def report():
        nonlocal last_time
        now = time.monotonic()
        seconds = max(now - last_time, 1e-9)
        for table, (last_rows, last_bytes) in last.items():
            group = [s for s in streams if s.table == table]
            rows, size = sum(s.rows for s in group), sum(s.bytes for s in group)
            if rows == last_rows and all(s.done for s in group):
                continue
            position = ns_to_literal(max(s.position for s in group))[11:23]
            print(f"[{position}] {table:<24} {rows:>14,} rows {(rows - last_rows) / seconds:>12,.0f} rows/s "
                  f"{(size - last_bytes) / seconds / 1e6:>9,.1f} MB/s", flush=True)
            last[table] = (rows, size)
        last_time = now

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", nargs="+", help="tables to replay")
    parser.add_argument("--ts", default="ts_recv", help="designated timestamp column (default: ts_recv)")
    parser.add_argument("--start", help="first timestamp, inclusive, for example 2026-10-27T13:30:00Z")
    parser.add_argument("--end", help="last timestamp, exclusive")
    parser.add_argument("--limit", type=int, help="stop after this many rows per table, for a quick test")
    parser.add_argument("--split-by", help="column to divide each table by, for example symbol")
    parser.add_argument("--splits", type=int, default=4, help="streams per table with --split-by (default: 4)")
    parser.add_argument("--window-seconds", type=float, default=3600.0,
                        help="event time per query; each must finish within the server's "
                             "query.timeout (default: 3600)")
    parser.add_argument("--report-seconds", type=float, default=1.0, help="seconds between progress lines")
    parser.add_argument("--tls-skip-verify", action="store_true",
                        help="accept QuestDB's TLS certificate without verifying it, for self-signed test servers")
    args = parser.parse_args()
    if args.splits < 1 or args.window_seconds <= 0:
        parser.error("--splits must be at least 1 and --window-seconds positive")

    conf = os.environ.get("QDB_CLIENT_CONF")
    if not conf:
        sys.exit("Set QDB_CLIENT_CONF, for example ws::addr=localhost:9000;")
    if args.tls_skip_verify:
        conf = conf.rstrip(";") + ";tls_verify=unsafe_off;"

    with questdb.connect(conf) as db:
        streams = build_streams(db, args)
    if not streams:
        sys.exit("Nothing to replay")

    stop = threading.Event()
    errors = []
    window_ns = int(args.window_seconds * NANOS)
    threads = [threading.Thread(target=run_stream, args=(s, conf, args.ts, window_ns, stop, errors), daemon=True)
               for s in streams]
    print(f"Replaying {', '.join(args.tables)}: {len(streams)} stream(s), {args.window_seconds:g}s windows",
          flush=True)
    started = time.monotonic()
    for thread in threads:
        thread.start()

    report = reporter(streams)
    try:
        while not all(s.done for s in streams):
            for thread in threads:
                thread.join(args.report_seconds / len(threads))
            report()
    except KeyboardInterrupt:
        print("\nStopping", flush=True)
        stop.set()
        for thread in threads:
            thread.join(10)

    elapsed = time.monotonic() - started
    for error in errors:
        print(f"Stream failed: {error}", file=sys.stderr)
    for table in dict.fromkeys(s.table for s in streams):
        group = [s for s in streams if s.table == table]
        rows, size = sum(s.rows for s in group), sum(s.bytes for s in group)
        print(f"{table}: {rows:,} rows, {sum(s.out_of_order for s in group):,} out of order, "
              f"{rows / elapsed:,.0f} rows/s, {size / elapsed / 1e6:,.1f} MB/s")
    total = sum(s.rows for s in streams)
    print(f"Replayed {total:,} rows over {len(streams)} stream(s) in {elapsed:.1f}s ({total / elapsed:,.0f} rows/s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
