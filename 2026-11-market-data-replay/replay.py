#!/usr/bin/env python3
"""Replay market data from QuestDB as fast as possible.

Streams rows from one or more tables over QuestDB's QWP protocol as Arrow
batches and hands each batch to publish(). Each connection runs in its own
thread, because queries on a connection run strictly one at a time.

A table is read as a sequence of consecutive time slices of about --slice-rows
rows each, rather than as one long query, so that no query outlives the server's
query.timeout (60 seconds by default). Rows arrive in timestamp order within
each slice.

By default each table uses one connection, which reads the slices in order, so
the whole table arrives in strict timestamp order. --connections N trades that
order for throughput: N connections take the next slice as they become free, so
order holds within each slice but not between them.

    QDB_CLIENT_CONF  QuestDB client configuration string

Usage:
    python3 replay.py equs_all_orderbook equs_all_trades
    python3 replay.py equs_all_orderbook --limit 1000000
    python3 replay.py equs_all_orderbook --start 2026-10-27T13:30:00Z --end 2026-10-27T14:00:00Z
    python3 replay.py equs_all_orderbook equs_all_trades --connections 8

Requires the questdb client (5.0 or later), pyarrow and numpy.
"""

import argparse
import collections
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
    """One table's replay: the slices still to read, and its running totals."""

    table: str
    ts_col: str
    slices: collections.deque
    workers: int
    limit: int = 0
    rows: int = 0
    bytes: int = 0
    out_of_order: int = 0
    position: int = 0
    last_row_at: float = 0.0
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    @property
    def done(self):
        return self.workers == 0

    def take(self, batch):
        """Claim a batch's rows, cut down to what --limit still allows."""
        with self.lock:
            if self.limit:
                batch = batch.slice(0, self.limit - self.rows)
            self.rows += batch.num_rows
        return batch

    def record(self, batch, out_of_order, position):
        with self.lock:
            self.bytes += batch.nbytes
            self.out_of_order += out_of_order
            self.position = max(self.position, position)
            self.last_row_at = time.monotonic()


def publish(table, batch):
    """Hand a batch of rows to whatever consumes the replay.

    This is the integration point: send to Kafka, feed a strategy, update a
    book. Called from a connection's thread with a pyarrow.RecordBatch, in
    timestamp order within its slice. Whatever runs here sets the pace, so a
    slow consumer slows its connection down through QWP flow control.
    """


def replay_slice(reader, stream, lo, hi, stop):
    """Read one time slice and publish its rows. Returns False to stop reading."""
    sql = (f"SELECT * FROM {stream.table} "
           f"WHERE {stream.ts_col} >= '{ns_to_literal(lo)}' AND {stream.ts_col} < '{ns_to_literal(hi)}'")
    # Every slice reads the same SYMBOL columns, so keep the dictionary warm.
    with reader.query(sql, reset_symbol_dict=False) as result:
        previous = lo
        for batch in result.iter_arrow():
            if batch.num_rows == 0:  # a result can open with a schema-only batch
                continue
            batch = stream.take(batch)
            if batch.num_rows == 0:  # another connection reached --limit first
                result.cancel()
                return False
            timestamps = to_ns(batch.column(stream.ts_col))
            out_of_order = int(np.count_nonzero(np.diff(timestamps) < 0)) + int(previous > timestamps[0])
            previous = int(timestamps[-1])
            publish(stream.table, batch)
            stream.record(batch, out_of_order, previous)
            if stop.is_set() or stream.rows == stream.limit:
                result.cancel()
                return False
    return True


def run_connection(stream, conf, stop, errors):
    """Read slices on a single connection until none are left."""
    try:
        with questdb.connect(conf) as db, db.reader() as reader:
            while not stop.is_set():
                try:
                    lo, hi = stream.slices.popleft()
                except IndexError:
                    return
                if not replay_slice(reader, stream, lo, hi, stop):
                    return
    except Exception as e:  # noqa: BLE001
        errors.append(f"{stream.table}: {e}")
        stop.set()
    finally:
        with stream.lock:
            stream.workers -= 1


def time_slices(db, table, args):
    """Consecutive (lo, hi) ranges of about --slice-rows rows, cut on minute boundaries."""
    clauses = []
    if args.start:
        clauses.append(f"{args.ts} >= {quoted(args.start)}")
    if args.end:
        clauses.append(f"{args.ts} < {quoted(args.end)}")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    bounds = db.query(f"SELECT min({args.ts}) lo, max({args.ts}) hi FROM {table}{where}").to_arrow()
    if bounds.num_rows == 0 or bounds.column(0).null_count:
        return []
    lo = int(to_ns(bounds.column(0).combine_chunks())[0])
    last = int(to_ns(bounds.column(1).combine_chunks())[0])

    minutes = db.query(f"SELECT {args.ts} minute, count() rows FROM {table}{where} SAMPLE BY 1m").to_arrow()
    slices = []
    rows = 0
    for minute, count in zip(to_ns(minutes.column(0).combine_chunks()), minutes.column(1).to_pylist()):
        if rows and rows + count > args.slice_rows:
            slices.append((lo, int(minute)))
            lo = int(minute)
            rows = 0
        rows += count
    slices.append((lo, last + 1))
    return slices


def build_streams(db, args):
    streams = []
    for table in args.tables:
        slices = time_slices(db, table, args)
        if not slices:
            print(f"{table}: no rows in that range")
            continue
        workers = min(args.connections, len(slices))
        streams.append(Stream(table, args.ts, collections.deque(slices), workers, args.limit or 0))
    return streams


def reporter(streams):
    """Return a function that prints one line per table: position, rows and rates."""
    last = {s.table: (0, 0) for s in streams}
    last_time = time.monotonic()

    def report():
        nonlocal last_time
        now = time.monotonic()
        seconds = max(now - last_time, 1e-9)
        for stream in streams:
            last_rows, last_bytes = last[stream.table]
            if stream.rows == last_rows:
                continue
            position = ns_to_literal(stream.position)[11:23]
            print(f"[{position}] {stream.table:<24} {stream.rows:>14,} rows "
                  f"{(stream.rows - last_rows) / seconds:>12,.0f} rows/s "
                  f"{(stream.bytes - last_bytes) / seconds / 1e6:>9,.1f} MB/s", flush=True)
            last[stream.table] = (stream.rows, stream.bytes)
        last_time = now

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", nargs="+", help="tables to replay")
    parser.add_argument("--ts", default="ts_recv", help="designated timestamp column (default: ts_recv)")
    parser.add_argument("--start", help="first timestamp, inclusive, for example 2026-10-27T13:30:00Z")
    parser.add_argument("--end", help="last timestamp, exclusive")
    parser.add_argument("--limit", type=int, help="stop after this many rows per table, for a quick test")
    parser.add_argument("--connections", type=int, default=1,
                        help="connections per table; above 1, rows are in order only within a slice (default: 1)")
    parser.add_argument("--slice-rows", type=int, default=5_000_000,
                        help="rows per query, roughly; each must finish within the server's query.timeout "
                             "(default: 5000000)")
    parser.add_argument("--report-seconds", type=float, default=1.0, help="seconds between progress lines")
    parser.add_argument("--tls-skip-verify", action="store_true",
                        help="accept QuestDB's TLS certificate without verifying it, for self-signed test servers")
    args = parser.parse_args()
    if args.connections < 1 or args.slice_rows < 1:
        parser.error("--connections and --slice-rows must be at least 1")

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
    threads = [threading.Thread(target=run_connection, args=(s, conf, stop, errors), daemon=True)
               for s in streams for _ in range(s.workers)]
    print(f"Replaying {', '.join(args.tables)}: {sum(len(s.slices) for s in streams)} slice(s), "
          f"{len(threads)} connection(s)", flush=True)
    started = time.monotonic()
    for thread in threads:
        thread.start()

    report = reporter(streams)
    try:
        while not all(s.done for s in streams):
            deadline = time.monotonic() + args.report_seconds
            for thread in threads:
                thread.join(max(deadline - time.monotonic(), 0))
            report()
    except KeyboardInterrupt:
        print("\nStopping", flush=True)
        stop.set()
        for thread in threads:
            thread.join(10)

    for error in errors:
        print(f"Stream failed: {error}", file=sys.stderr)
    # Time to the last row, so a table that ends early or a slow cancel does not skew its rate.
    for stream in streams:
        elapsed = max(stream.last_row_at - started, 1e-9)
        print(f"{stream.table}: {stream.rows:,} rows, {stream.out_of_order:,} out of order, {elapsed:.1f}s, "
              f"{stream.rows / elapsed:,.0f} rows/s, {stream.bytes / elapsed / 1e6:,.1f} MB/s")
    total = sum(s.rows for s in streams)
    elapsed = max(max(s.last_row_at for s in streams) - started, 1e-9)
    print(f"Replayed {total:,} rows over {len(threads)} connection(s) in {elapsed:.1f}s ({total / elapsed:,.0f} rows/s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
