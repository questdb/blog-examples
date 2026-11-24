#!/usr/bin/env python3
"""
Per-row ingest latency for one run, computed from QuestDB's own WAL telemetry.

Every original row is produced with ts = wall clock, and the producer sends rows in
ts order, so each row belongs to the first WAL transaction whose maxTimestamp reaches
its ts. sys.telemetry_wal records, per transaction (seqTxn):
  event 109  commit to the WAL     -> commit latency  = created - ts
  event 105  applied to the table  -> visible latency = created - ts (readable by queries)
Duplicates carry an older ts, so a transaction's minTimestamp is not used: a transaction
covers (previous transaction's maxTimestamp, its own maxTimestamp].

With several producers writing one table at once, each connection commits through its own
WAL writer (walId) in its own ts order, so the rule holds per writer: pass --wal-id and
the id range of the producer that used it (walids.py lists the writers of a table). Only
commit events (109) are filtered by walId; apply events (105) are matched on seqTxn alone,
because their walId does not always name the writer that committed the transaction.

The server returns 0.1 ms latency histograms, one query per id window (--windows), and
the percentiles are computed from the merged histogram, so long runs stay well inside the
server's query timeout and each window is reported too.

Usage: latency.py <table> <first_id> <last_id> [--id-col event_id] [--wal-id N] [--windows N]
"""
import argparse

from common import sql
from common import table_id as lookup_table_id

INVALID = -1.0   # bucket for rows with no transaction to assign them to (first txn)


def histogram(table, table_id, id_col, lo_id, hi_id, wal):
    """{(commit_bucket_ms, visible_bucket_ms): rows} for rows with ids in [lo_id, hi_id]."""
    res = sql(f"""
WITH c0 AS (
  SELECT seqTxn, created commit_ts, maxTimestamp hi,
         max(maxTimestamp) OVER (ORDER BY seqTxn
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_hi
  FROM 'sys.telemetry_wal' WHERE tableId = {table_id} AND event = 109{wal}
),
c AS (SELECT * FROM c0 WHERE hi > prev_hi),
a AS (
  SELECT seqTxn, created applied_ts FROM 'sys.telemetry_wal'
  WHERE tableId = {table_id} AND event = 105
),
tx AS ((
  SELECT c.seqTxn, c.commit_ts, a.applied_ts, c.prev_hi lo, c.hi
  FROM c JOIN a ON a.seqTxn = c.seqTxn
  WHERE c.prev_hi IS NOT NULL
  ORDER BY lo
) timestamp(lo)),
r AS (
  SELECT e.ts, t.hi,
         (t.commit_ts - e.ts) / 1000.0 commit_ms,
         (t.applied_ts - e.ts) / 1000.0 visible_ms
  FROM {table} e ASOF JOIN tx t
  WHERE e.{id_col} BETWEEN {lo_id} AND {hi_id}
),
h AS (
  SELECT CASE WHEN hi IS NULL OR ts > hi OR commit_ms < 0 THEN {INVALID}
              ELSE round(commit_ms, 1) END cb,
         CASE WHEN hi IS NULL OR ts > hi OR commit_ms < 0 THEN {INVALID}
              ELSE round(visible_ms, 1) END vb
  FROM r
)
SELECT cb, vb, count() n FROM h GROUP BY cb, vb""")
    return {(cb, vb): n for cb, vb, n in res["dataset"]}


def stats(hist):
    """n, unassigned and percentiles (ms) from a {(commit, visible): rows} histogram."""
    unassigned = sum(n for (cb, _), n in hist.items() if cb == INVALID)
    commit, visible = {}, {}
    for (cb, vb), n in hist.items():
        if cb != INVALID:
            commit[cb] = commit.get(cb, 0) + n
            visible[vb] = visible.get(vb, 0) + n

    def pct(h, p):
        total, seen = sum(h.values()), 0
        for value in sorted(h):
            seen += h[value]
            if seen >= p * total:
                return value
        return float("nan")

    return {"n": sum(visible.values()), "unassigned": unassigned,
            "commit_p50": pct(commit, 0.5), "commit_p99": pct(commit, 0.99),
            "visible_p50": pct(visible, 0.5), "visible_p95": pct(visible, 0.95),
            "visible_p99": pct(visible, 0.99), "visible_max": max(visible, default=float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("lo_id", type=int)
    ap.add_argument("hi_id", type=int)
    ap.add_argument("--id-col", default="event_id", help="row id column the range applies to")
    ap.add_argument("--wal-id", type=int, help="only transactions of this WAL writer")
    ap.add_argument("--windows", type=int, default=1, help="split the id range into N windows")
    args = ap.parse_args()
    table_id = lookup_table_id(args.table)
    wal = f" AND walId = {args.wal_id}" if args.wal_id is not None else ""

    span = args.hi_id - args.lo_id + 1
    merged = {}
    print(f"{args.table} {args.id_col} {args.lo_id}..{args.hi_id}"
          + (f" walId {args.wal_id}" if args.wal_id is not None else ""))
    for w in range(args.windows):
        lo = args.lo_id + span * w // args.windows
        hi = args.lo_id + span * (w + 1) // args.windows - 1
        h = histogram(args.table, table_id, args.id_col, lo, hi, wal)
        for k, n in h.items():
            merged[k] = merged.get(k, 0) + n
        if args.windows > 1:
            s = stats(h)
            print(f"  window {w + 1}/{args.windows} ids {lo}..{hi}: n={s['n']} "
                  f"visible p50={s['visible_p50']:.1f} p95={s['visible_p95']:.1f} "
                  f"p99={s['visible_p99']:.1f} max={s['visible_max']:.1f} ms")
    for k, v in stats(merged).items():
        print(f"  {k:>12} = {v:.1f}" if isinstance(v, float) else f"  {k:>12} = {v}")


if __name__ == "__main__":
    main()
