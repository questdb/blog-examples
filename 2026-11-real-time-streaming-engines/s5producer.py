#!/usr/bin/env python3
"""
S5 producer: orders and payments for the stream-stream interval join, straight to QuestDB.

Port of the S5 mode of fmorillo7694/spark-rtm-vs-flink common/producer.py: --rate orders/s;
a --match-rate fraction of orders gets one payment U(0, --max-delay-ms) after the order
(inside the 10 s join bound, with 0.5 s margin), a further --late-rate fraction gets one
payment U(--late-min-ms, --late-max-ms) later (outside the bound: must not join), the rest
get none. A payment's ts is its send wall clock, the latency anchor for the joined pair.

Creates <prefix>_orders and <prefix>_payments (refuses if they exist) and writes run
metadata, including the expected join counts, to runs/<prefix>.json.

Usage: s5producer.py <prefix> [--rate 27000] [--seconds 300] [--batch 2000] [--addr host:port]
"""
import argparse
import heapq
import json
import os
import random
import sys
import time

import questdb
from questdb import TimestampNanos

from common import QDB_ADDR, RUNS_DIR, sql


def create_tables(prefix):
    for t in (f"{prefix}_orders", f"{prefix}_payments"):
        if sql(f"SELECT count() FROM tables() WHERE table_name = '{t}'")["dataset"][0][0]:
            sys.exit(f"{t} already exists; pick a new prefix")
    sql(f"CREATE TABLE {prefix}_orders (ts TIMESTAMP, order_id LONG, user_id LONG, amount DOUBLE) "
        f"TIMESTAMP(ts) PARTITION BY HOUR DEDUP UPSERT KEYS(ts, order_id)")
    sql(f"CREATE TABLE {prefix}_payments (ts TIMESTAMP, payment_id LONG, order_id LONG, "
        f"user_id LONG, amount DOUBLE) TIMESTAMP(ts) PARTITION BY HOUR "
        f"DEDUP UPSERT KEYS(ts, payment_id)")
    print(f"created {prefix}_orders, {prefix}_payments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--addr", default=QDB_ADDR)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--rate", type=int, default=27000, help="orders/s")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keys", type=int, default=1000000)
    ap.add_argument("--match-rate", type=float, default=0.8)
    ap.add_argument("--late-rate", type=float, default=0.05)
    ap.add_argument("--max-delay-ms", type=int, default=9500)
    ap.add_argument("--late-min-ms", type=int, default=11000)
    ap.add_argument("--late-max-ms", type=int, default=14000)
    ap.add_argument("--create-only", action="store_true",
                    help="only create the two tables (for the Kafka path, where the upstream "
                         "producer sends the orders and payments)")
    args = ap.parse_args()
    orders_t, payments_t = f"{args.prefix}_orders", f"{args.prefix}_payments"
    create_tables(args.prefix)
    if args.create_only:
        return

    conf = f"ws::addr={args.addr};auto_flush_rows={args.batch};auto_flush_interval=1000;"
    rng = random.Random(args.seed)
    slice_s = 0.01
    per_slice = max(1, round(args.rate * slice_s))
    pending = []   # (due wall clock, order_id, user_id, amount, in_bound)
    seq = orders = payments = matched = late = behind = 0
    print(f"[s5] {conf} orders={orders_t} payments={payments_t} rate={args.rate} orders/s "
          f"seconds={args.seconds}", flush=True)

    with questdb.connect(conf) as db:
        with db.sender() as sender:
            def send_due(now):
                nonlocal payments
                while pending and pending[0][0] <= now:
                    _, oid, uid, amount = heapq.heappop(pending)
                    sender.row(payments_t,
                               columns={"payment_id": oid, "order_id": oid, "user_id": uid,
                                        "amount": amount},
                               at=TimestampNanos(time.time_ns()))
                    payments += 1

            started = time.time()
            next_slice = started
            while time.time() - started < args.seconds:
                for _ in range(per_slice):
                    seq += 1
                    uid = rng.randint(1, args.keys)
                    amount = round(rng.uniform(0.0, 500.0), 2)
                    ts_ns = time.time_ns()
                    sender.row(orders_t, columns={"order_id": seq, "user_id": uid, "amount": amount},
                               at=TimestampNanos(ts_ns))
                    orders += 1
                    r = rng.random()
                    if r < args.match_rate:
                        delay = rng.uniform(0.0, args.max_delay_ms / 1000.0)
                        matched += 1
                    elif r < args.match_rate + args.late_rate:
                        delay = rng.uniform(args.late_min_ms / 1000.0, args.late_max_ms / 1000.0)
                        late += 1
                    else:
                        continue
                    heapq.heappush(pending, (ts_ns / 1e9 + delay, seq, uid, amount))
                send_due(time.time())
                next_slice += slice_s
                wait = next_slice - time.time()
                if wait > 0:
                    time.sleep(wait)
                else:
                    behind += 1
                    next_slice = time.time()
            # Send every scheduled payment at its due time.
            while pending:
                wait = pending[0][0] - time.time()
                if wait > 0:
                    time.sleep(min(wait, 0.01))
                send_due(time.time())
            sender.flush(wait=True)
            ended = time.time()

    meta = {"run_id": args.prefix, "orders_table": orders_t, "payments_table": payments_t,
            "conf": conf, "rate": args.rate, "seconds": args.seconds, "seed": args.seed,
            "started": started, "ended": ended, "orders": orders, "payments": payments,
            "expected_pairs": matched, "late_payments": late, "behind_slices": behind}
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{args.prefix}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[s5] DONE orders={orders} payments={payments} expected_pairs={matched} late={late} "
          f"in {ended - started:.1f}s ({(orders + payments) / (ended - started):.0f} rows/s) "
          f"behind_slices={behind} -> {path}", flush=True)


if __name__ == "__main__":
    main()
