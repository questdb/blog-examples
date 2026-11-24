#!/usr/bin/env python3
"""
Bounds on per-row latency into an aggregating materialized view (S1), from WAL telemetry.

A view's WAL transactions do not record which base transaction they include, so for
each base transaction k (applied at A_k, covering row timestamps (lo_k, hi_k]):
  lower bound: the first view apply at or after A_k contains k
  upper bound: the second view apply after A_k contains k (a refresh already running
               when k was applied may have missed it; the next one cannot)
Row latency = view apply time - row ts. Rows are taken as uniform over (lo_k, hi_k],
sampled every 1 ms, which is enough for percentiles. Ranges are tracked per WAL writer,
so the bounds also hold with several producers writing the base table at once.

Usage: view_latency.py <base_table> <view_name> [--from ts] [--to ts]
  --from/--to  only base transactions whose rows end in [from, to) (e.g. one load phase)
"""
import argparse
import bisect

from common import sql, table_id


def micros(col):
    return f"{col}::long"


def pct(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("view")
    ap.add_argument("--from", dest="from_ts")
    ap.add_argument("--to", dest="to_ts")
    args = ap.parse_args()
    base, view = args.base, args.view
    bid, vid = table_id(base), table_id(view)
    phase = ""
    if args.from_ts:
        phase += f" AND hi >= '{args.from_ts}'"
    if args.to_ts:
        phase += f" AND hi < '{args.to_ts}'"
    base_rows = sql(f"""
WITH c0 AS (
  SELECT seqTxn, maxTimestamp hi,
         max(maxTimestamp) OVER (PARTITION BY walId ORDER BY seqTxn
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_hi
  FROM 'sys.telemetry_wal' WHERE tableId = {bid} AND event = 109
),
c AS (SELECT * FROM c0 WHERE hi > prev_hi{phase}),
a AS (SELECT seqTxn, created applied FROM 'sys.telemetry_wal' WHERE tableId = {bid} AND event = 105)
SELECT {micros('c.prev_hi')}, {micros('c.hi')}, {micros('a.applied')}
FROM c JOIN a ON a.seqTxn = c.seqTxn ORDER BY c.seqTxn
""")["dataset"]
    view_applies = sorted(r[0] for r in sql(
        f"SELECT {micros('created')} FROM 'sys.telemetry_wal' WHERE tableId = {vid} AND event = 105"
    )["dataset"])

    lower, upper = [], []
    missing = 0
    for lo, hi, applied in base_rows:
        i = bisect.bisect_left(view_applies, applied)
        if i + 1 >= len(view_applies):
            missing += 1
            continue
        v1, v2 = view_applies[i], view_applies[i + 1]
        t = hi
        while t > lo:
            lower.append((v1 - t) / 1000.0)
            upper.append((v2 - t) / 1000.0)
            t -= 1000
    lower.sort()
    upper.sort()
    print(f"{view} over {base}: {len(base_rows)} base txns, {len(view_applies)} view applies, "
          f"{missing} base txns without two later view applies")
    for name, vals in (("lower", lower), ("upper", upper)):
        print(f"  {name}: p50={pct(vals, 0.5):.1f} p95={pct(vals, 0.95):.1f} "
              f"p99={pct(vals, 0.99):.1f} max={vals[-1]:.1f} ms")


if __name__ == "__main__":
    main()
