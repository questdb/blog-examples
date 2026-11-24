#!/usr/bin/env python3
"""
Bounds on per-row ingest latency when rows reach a table through several writers in no
particular ts order (for example the Kafka connector: one writer per task, each fed by
partitions that interleave many producers).

Every WAL transaction records the min and max ts of its rows (sys.telemetry_wal, commit
event 109) and when it was applied (event 105). A row lies inside its own transaction's
[minTimestamp, maxTimestamp], so among all transactions whose range covers the row's ts:
  lower bound: the earliest apply time
  upper bound: the latest apply time
No ordering or writer mapping is assumed. Late duplicates widen every range they land in
(their ts is older), so run without duplicates for tight bounds. Rows are sampled with
<id_col> % sample = 0.

Usage: cover_latency.py <table> [--id-col event_id] [--sample 50] [--from ts] [--to ts]
"""
import argparse
import bisect

from common import sql, table_id


def pct(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--id-col", default="event_id")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--from", dest="from_ts")
    ap.add_argument("--to", dest="to_ts")
    args = ap.parse_args()
    tid = table_id(args.table)

    txns = sql(f"""
WITH c AS (SELECT seqTxn, minTimestamp lo, maxTimestamp hi FROM 'sys.telemetry_wal'
           WHERE tableId = {tid} AND event = 109),
a AS (SELECT seqTxn, created applied FROM 'sys.telemetry_wal' WHERE tableId = {tid} AND event = 105)
SELECT c.lo::long, c.hi::long, a.applied::long FROM c JOIN a ON a.seqTxn = c.seqTxn
ORDER BY c.lo""")["dataset"]
    los = [t[0] for t in txns]
    widest = max(t[1] - t[0] for t in txns)

    window = ""
    if args.from_ts:
        window += f" AND ts >= '{args.from_ts}'"
    if args.to_ts:
        window += f" AND ts < '{args.to_ts}'"
    rows = sql(f"SELECT ts::long FROM {args.table} "
               f"WHERE {args.id_col} % {args.sample} = 0{window}")["dataset"]

    lower, upper, uncovered = [], [], 0
    for (ts,) in rows:
        # candidates: transactions starting no later than ts and no earlier than ts - widest
        i = bisect.bisect_left(los, ts - widest)
        j = bisect.bisect_right(los, ts)
        applied = [txns[k][2] for k in range(i, j) if txns[k][1] >= ts]
        if not applied:
            uncovered += 1
            continue
        lower.append((min(applied) - ts) / 1000.0)
        upper.append((max(applied) - ts) / 1000.0)
    lower.sort()
    upper.sort()
    print(f"{args.table}: {len(txns)} txns (widest {widest / 1000:.0f} ms), "
          f"{len(lower)} sampled rows (1/{args.sample}), {uncovered} not covered")
    for name, vals in (("lower", lower), ("upper", upper)):
        print(f"  {name}: p50={pct(vals, 0.5):.1f} p95={pct(vals, 0.95):.1f} "
              f"p99={pct(vals, 0.99):.1f} max={vals[-1]:.1f} ms")


if __name__ == "__main__":
    main()
