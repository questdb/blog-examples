#!/usr/bin/env python3
"""
S5 as a polling consumer would see it: every <interval_ms>, read the joined pairs whose payment
arrived in the last <overlap_ms> (the ASOF JOIN of s5check.py), and record each pair the first
time it appears. End-to-end latency per pair = the wall clock when the response arrived minus
the payment's ts, which s5producer.py stamps from the wall clock at send. Run this on the
producer host, so both clocks are the same clock.

A payment sent just before a poll can become visible after it, so each poll reads an
overlapping window and skips the payment ids already seen; a pair that becomes visible more
than <overlap_ms> after it was sent would be missed, and the report counts them (complete =
pairs seen / pairs in the database for the same payment-time range).

The completeness check is one large ASOF JOIN over the whole run; it waits <settle_s> after
the last poll so it runs after the producer has stopped. On a server with a small network
pool, a long REST query holds a network worker, and QWP ingestion on the same port waits for
it (seen with shared.network.worker.count=2: a 20 s ingest stall).

Usage: s5consumer.py <prefix> <interval_ms> <seconds> [delay_s, default 0] [overlap_ms, default 1500]
                     [settle_s, default 0]
"""
import statistics
import sys
import time
from array import array

from common import sql


def pct(vals, p):
    return vals[min(len(vals) - 1, int(p * len(vals)))]


def main():
    prefix, interval_ms, seconds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    delay = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    overlap_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
    settle_s = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    fast_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    sweep_every = int(sys.argv[8]) if len(sys.argv) > 8 else 10
    full = len(sys.argv) > 9 and sys.argv[9] == "full"
    orders, payments = f"{prefix}_orders", f"{prefix}_payments"
    # "full" also fetches the columns of the post's S5 query (what a real consumer would use);
    # otherwise only the id and timestamp the measurement needs.
    extra = (", o.ts order_ts, o.order_id, o.user_id, o.amount order_amount, p.amount pay_amount"
             if full else "")

    def query(window_ms):
        return f"""
SELECT /*+ asof_dense(p o) */ p.payment_id, p.ts::long pay_us{extra}
FROM {payments} p
ASOF JOIN {orders} o ON (order_id) TOLERANCE 10s
WHERE p.ts IN '$now-{window_ms}T..$now' AND o.ts IS NOT NULL"""

    # Two-tier mode (fast_ms > 0): most polls read only the last fast_ms, and every
    # sweep_every-th poll reads the full overlap to pick up the rare pair that became visible
    # late. Each pair is then re-read a few times instead of overlap_ms / interval_ms times.
    q_full, q_fast = query(overlap_ms), query(fast_ms) if fast_ms else None
    if delay:
        time.sleep(delay)

    seen = {}                       # payment_id -> pay_us, pruned to the overlap window
    lat_ms, seen_ts, elapsed = array("d"), array("q"), array("d")
    exec_ms, gaps = [], []
    start = time.time()
    last = None
    k = 0
    lo_us = int(start * 1e6)
    while time.time() - start < seconds:
        t0 = time.time()
        if last is not None:
            gaps.append((t0 - last) * 1000)
        last = t0
        q = q_fast if q_fast and k % sweep_every else q_full
        res = sql(q, timings=True)
        now_us = time.time_ns() / 1000
        exec_ms.append(res["timings"]["execute"] / 1e6)
        if k == 0 and res["dataset"]:
            # Count completeness only for payments newer than the first poll's backlog.
            lo_us = max(row[1] for row in res["dataset"]) + 1
        for row in res["dataset"]:
            pid, pay_us = row[0], row[1]
            if pid not in seen:
                seen[pid] = pay_us
                if k == 0:
                    continue   # the first poll returns a backlog of overlap_ms, not new pairs
                lat_ms.append((now_us - pay_us) / 1000)
                seen_ts.append(pay_us)
                elapsed.append(t0 - start)
        cutoff = now_us - 2 * overlap_ms * 1000
        if k % 10 == 0:
            for pid in [p for p, ts in seen.items() if ts < cutoff]:
                del seen[pid]
        k += 1
        nxt = start + k * interval_ms / 1000
        time.sleep(max(0.0, nxt - time.time()))

    # Completeness over payments newer than the first poll's backlog, up to 2 s before the last
    # poll.
    hi_us = int((last - 2) * 1e6)
    got = sum(1 for ts in seen_ts if lo_us <= ts <= hi_us)
    time.sleep(settle_s)   # let the producer finish before the large completeness query
    want = sql(f"""
SELECT count() FROM (
  SELECT /*+ asof_dense(p o) */ p.payment_id FROM {payments} p
  ASOF JOIN {orders} o ON (order_id) TOLERANCE 10s
  WHERE p.ts >= {lo_us}::timestamp AND p.ts <= {hi_us}::timestamp AND o.ts IS NOT NULL)""")["dataset"][0][0]

    lat = sorted(lat_ms)
    ex = sorted(exec_ms)
    tier = (f" fast={fast_ms}ms sweep_every={sweep_every}" if fast_ms else "") + (" cols=full" if full else "")
    print(f"CONSUMER interval={interval_ms}ms overlap={overlap_ms}ms{tier} polls={len(exec_ms)} "
          f"mean_gap={statistics.mean(gaps):.1f}ms query p50={pct(ex, 0.5):.1f} p99={pct(ex, 0.99):.1f} ms "
          f"complete={got}/{want}", flush=True)
    print(f"CONSUMER interval={interval_ms}ms end-to-end p50={pct(lat, 0.5):.1f} p95={pct(lat, 0.95):.1f} "
          f"p99={pct(lat, 0.99):.1f} max={lat[-1]:.1f} ms pairs={len(lat)}", flush=True)
    if seconds > 600:
        for w in range(0, seconds, 300):
            lw = sorted(lat_ms[i] for i in range(len(lat_ms)) if w <= elapsed[i] < w + 300)
            if lw:
                print(f"  CONSUMER interval={interval_ms}ms {(delay + w) // 60:>3}-{(delay + w + 300) // 60:<3} min "
                      f"p50={pct(lw, 0.5):.1f} p99={pct(lw, 0.99):.1f} max={lw[-1]:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
