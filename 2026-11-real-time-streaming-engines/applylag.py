#!/usr/bin/env python3
"""
WAL commit-to-apply lag over time for one table, in buckets, from sys.telemetry_wal.
Shows whether apply falls behind (a backlog) or the lag is spread evenly.

Usage: applylag.py <table> [bucket, default 30s]
"""
import sys

from common import sql, table_id


def main():
    t = sys.argv[1]
    bucket = sys.argv[2] if len(sys.argv) > 2 else "30s"
    tid = table_id(t)
    res = sql(f"""
WITH c AS (SELECT seqTxn, created commit_ts FROM 'sys.telemetry_wal'
           WHERE tableId = {tid} AND event = 109),
a AS (SELECT seqTxn, created applied_ts, latency apply_ms FROM 'sys.telemetry_wal'
      WHERE tableId = {tid} AND event = 105),
j AS ((SELECT c.commit_ts, (a.applied_ts - c.commit_ts) / 1000.0 lag_ms, a.apply_ms
       FROM c JOIN a ON a.seqTxn = c.seqTxn ORDER BY c.commit_ts) timestamp(commit_ts))
SELECT commit_ts, count() txns, avg(lag_ms) avg_lag, max(lag_ms) max_lag, avg(apply_ms) avg_apply
FROM j SAMPLE BY {bucket}""")
    print(f"{t}: commit-to-apply lag per {bucket}")
    for ts, n, avg, mx, ap in res["dataset"]:
        print(f"  {ts}  txns={n:>5}  avg_lag={avg:8.1f} ms  max_lag={mx:8.1f} ms  avg_apply={ap:6.2f}")


if __name__ == "__main__":
    main()
