#!/usr/bin/env python3
"""
Run the query-time workloads the way a dashboard or downstream job would: each query once a
second, on its own thread, for a fixed time while the producer writes, against now().
Reports the execute-time distribution per query (QuestDB's own REST timings, as qtime.py).
A query that takes longer than a second skips the ticks it missed.

Kinds:
  events  on an events table: S1 live window, S3 sessions closed in the last 1 s, S4
          enrichment (last 1 s and 100 ms), the plain view (last 1 s and 100 ms, when
          <table>_clean_v exists); same queries as qtime.py
  view    on an events table: only the stateless plain view (last 1 s and 100 ms)
  s5      on <prefix>_orders and <prefix>_payments: the 1 s S5 poll as the interval join and
          as the ASOF JOIN (s5check.py)

Usage: poll.py <events|view|s5> <table or s5 prefix> <seconds> [delay_s, default 0]
"""
import statistics
import sys
import threading
import time

from common import sql
import qtime
import s5check


def pct(vals, p):
    return vals[min(len(vals) - 1, int(p * len(vals)))]


def loop(q, seconds, out):
    # qtime.py's queries already compute the full result inside SELECT count(); wrapping
    # them again would let the server skip that work.
    wrapped = q if q.lstrip().upper().startswith("SELECT COUNT()") else f"SELECT count() FROM ({q})"
    start = time.time()
    k = 0
    while True:
        now = time.time()
        if now - start >= seconds:
            return
        res = sql(wrapped, timings=True)
        out.append((res["timings"]["execute"] / 1e6, res["dataset"][0][0], now - start))
        k = int((time.time() - start) // 1) + 1   # next whole second; skips missed ticks
        time.sleep(max(0.0, start + k - time.time()))


def main():
    kind, target, seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
    delay = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    if delay:
        time.sleep(delay)   # start mid-run, while the producer is still writing
    if kind == "events":
        queries = qtime.build(target, "now()")
    elif kind == "view":
        queries = {k: q for k, q in qtime.build(target, "now()").items() if "plain view" in k}
    elif kind == "s5":
        queries = {f"S5 poll {name}": q for name, q in
                   s5check.polls(f"{target}_orders", f"{target}_payments", "now()").items()}
    else:
        sys.exit(f"unknown kind {kind}")

    results = {name: [] for name in queries}
    threads = [threading.Thread(target=loop, args=(q, seconds, results[name]))
               for name, q in queries.items()]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for name, r in results.items():
        t = sorted(x[0] for x in r)
        rows = statistics.median(x[1] for x in r)
        print(f"POLL {name:<44} n={len(t):>4} rows_median={rows:>9.0f}  execute "
              f"p50={pct(t, 0.5):7.1f} p95={pct(t, 0.95):7.1f} p99={pct(t, 0.99):7.1f} "
              f"max={t[-1]:7.1f} ms", flush=True)
        if seconds > 600:   # long runs: also per 5 minutes, to show any drift as tables grow
            for w in range(0, seconds, 300):
                tw = sorted(x[0] for x in r if w <= x[2] < w + 300)
                if tw:
                    print(f"  {name:<44} {(delay + w) // 60:>3}-{(delay + w + 300) // 60:<3} min "
                          f"n={len(tw):>4} p50={pct(tw, 0.5):7.1f} p99={pct(tw, 0.99):7.1f} "
                          f"max={tw[-1]:7.1f} ms", flush=True)


if __name__ == "__main__":
    main()
