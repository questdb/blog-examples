#!/usr/bin/env python3
"""
Time the query-time workloads on a loaded table: S1 live window, S3 sessions closed in the
last second (incremental, see s3check.py), the stateless transform through a plain view,
and S4 enrichment.

Each query is wrapped in SELECT count() so the server computes the full result without
shipping it, and run several times; the report is QuestDB's own execute time from the
REST timings, min and median over the measured runs after one warm-up.

Usage: qtime.py <table> <window_end_ts> [runs] [s3_lookback_s] [delay_s]
  window_end_ts: the newest ts in the data, e.g. 2026-11-17T17:48:49, or now() while the
                 table is being written
  s3_lookback_s: S3 lookback in seconds, default 30
  delay_s:       wait this long before the first query (start mid-run), default 0
The plain-view query runs only when <table>_clean_v exists (mktable.py --clean-view).
"""
import statistics
import sys
import time

from common import sql
from s3check import queries as s3_queries


def run(query):
    res = sql(query, timings=True)
    return res["dataset"][0][0], res["timings"]["execute"] / 1e6


def build(t, end, lookback=30):
    """The timed queries by name; end is an SQL timestamp expression such as now()."""
    has_view = sql(f"SELECT count() FROM tables() WHERE table_name = '{t}_clean_v'")["dataset"][0][0]
    s3_incremental, _ = s3_queries(t, end, lookback, 1)

    queries = {
        "S1 live, current 10s window": f"""
SELECT count() FROM (
  SELECT ts, user_id, count() cnt, sum(amount) sum_amount, max(amount) max_amount, max(ts) max_ts
  FROM {t} WHERE ts IN interval(dateadd('s', -10, {end}), {end})
  SAMPLE BY 10s
)""",
        f"S3 sessions closed in last 1s, {lookback}s lookback":
            f"SELECT count() FROM ({s3_incremental})",
        "S4 enrich last 1s": f"""
SELECT count() FROM (
  SELECT e.ts, e.user_id, e.amount, r.tier, r.region
  FROM (SELECT * FROM {t} WHERE ts IN interval(dateadd('s', -1, {end}), {end})) e
  JOIN bench_users_ref r ON e.user_id = r.user_id
)""",
        "S4 enrich last 100ms": f"""
SELECT count() FROM (
  SELECT e.ts, e.user_id, e.amount, r.tier, r.region
  FROM (SELECT * FROM {t} WHERE ts IN interval(dateadd('T', -100, {end}), {end})) e
  JOIN bench_users_ref r ON e.user_id = r.user_id
)""",
    }
    if has_view:
        for label, unit, n in (("1s", "s", 1), ("100ms", "T", 100)):
            # Aggregate the projected columns so the transform is computed, not skipped.
            queries[f"stateless plain view, last {label}"] = f"""
SELECT count(), sum(amount_with_tax), count_distinct(country) FROM {t}_clean_v
WHERE ts IN interval(dateadd('{unit}', -{n}, {end}), {end})"""
    return queries


def main():
    t, end = sys.argv[1], sys.argv[2]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    lookback = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    delay = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    if delay:
        time.sleep(delay)   # start mid-run, while the producer is still writing
    end = end if end == "now()" else f"'{end}'"
    queries = build(t, end, lookback)

    for name, q in queries.items():
        rows, _ = run(q)   # warm-up
        times = [run(q)[1] for _ in range(runs)]
        print(f"{name:<46} rows={rows:>8}  execute min={min(times):7.1f} ms  "
              f"median={statistics.median(times):7.1f} ms")


if __name__ == "__main__":
    main()
