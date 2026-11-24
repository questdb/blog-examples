#!/usr/bin/env python3
"""
S3 sessions (5 s gap per user): check the incremental "sessions closed in the last slot"
query against a full recompute, then time both.

A streaming engine emits each session once, when it closes. The query equivalent, run every
`slot` seconds, returns the sessions whose last event lies in (end - gap - slot,
end - gap]: no event for `gap` seconds after it, so the session is closed. It first finds
the users with an event in that slot (a short scan), then runs the gap/session window
functions on their events only, over a bounded `lookback` (sessions longer than that are
cut at its start).

Usage: s3check.py <table> <end_ts> [lookback_s, default 30] [slot_s, default 1] [runs, default 5]
  end_ts: e.g. 2026-11-19T09:32:50, or now() on a table being written
"""
import statistics
import sys

from common import sql

GAP_S = 5


def sessions(table, end, lookback, users_filter):
    """Sessions per user over (end - lookback, end], optionally only for users in a CTE."""
    src = f"(SELECT ts, user_id FROM {table} WHERE ts IN interval(dateadd('s', -{lookback}, {end}), {end}))"
    if users_filter:
        src = f"(SELECT e.ts, e.user_id FROM {src} e JOIN closing c ON e.user_id = c.user_id)"
    return f"""
p AS (
  SELECT ts, user_id, lag(ts) OVER (PARTITION BY user_id ORDER BY ts) prev_ts FROM {src}
), s AS (
  SELECT ts, user_id,
         sum(CASE WHEN ts > dateadd('s', {GAP_S}, prev_ts) THEN 1 ELSE 0 END)
           OVER (PARTITION BY user_id ORDER BY ts) sid
  FROM p
), g AS (
  SELECT user_id, sid, min(ts) session_start, max(ts) session_end, count() events
  FROM s GROUP BY user_id, sid
)"""


def queries(table, end, lookback, slot):
    lo = f"dateadd('s', -{GAP_S + slot}, {end})"
    hi = f"dateadd('s', -{GAP_S}, {end})"
    slot_filter = f"session_end > {lo} AND session_end <= {hi}"
    incremental = f"""
WITH closing AS (
  SELECT DISTINCT user_id FROM {table} WHERE ts > {lo} AND ts <= {hi}
),{sessions(table, end, lookback, True)}
SELECT user_id, session_start, session_end, events FROM g WHERE {slot_filter}"""
    full = f"""
WITH{sessions(table, end, lookback, False)}
SELECT user_id, session_start, session_end, events FROM g WHERE {slot_filter}"""
    return incremental, full


def summary(q):
    """Count, total events and a checksum of the sessions a query returns."""
    res = sql(f"SELECT count() n, sum(events) ev, sum(user_id) uid, "
              f"sum(session_start::long) st FROM ({q})")
    return tuple(res["dataset"][0])


def exec_ms(q, runs):
    wrapped = f"SELECT count() FROM ({q})"
    sql(wrapped)   # warm-up
    return [sql(wrapped, timings=True)["timings"]["execute"] / 1e6 for _ in range(runs)]


def main():
    table, end = sys.argv[1], sys.argv[2]
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    slot = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    end = end if end == "now()" else f"'{end}'"
    incremental, full = queries(table, end, lookback, slot)

    a, b = summary(incremental), summary(full)
    print(f"sessions closed in the {slot}s slot ending {GAP_S}s before {end}, {lookback}s lookback")
    print(f"  incremental: n={a[0]} events={a[1]}   full recompute: n={b[0]} events={b[1]}   "
          f"{'MATCH' if a == b else 'MISMATCH'}")
    for name, q in (("incremental", incremental), ("full recompute", full)):
        t = exec_ms(q, runs)
        print(f"  {name:<15} execute min={min(t):7.1f} ms  median={statistics.median(t):7.1f} ms")


if __name__ == "__main__":
    main()
