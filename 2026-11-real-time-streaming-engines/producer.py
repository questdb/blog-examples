#!/usr/bin/env python3
"""
Event producer for the QuestDB arm of the spark-rtm-vs-flink benchmark.

Port of fmorillo7694/spark-rtm-vs-flink common/producer.py: same event shape, same
user/type/country/amount distributions, same duplicate rule (an exact copy with the
same event_id and ts, re-sent after a uniform 0..dup_max_delay_ms delay). The only
change is the transport: rows go straight to QuestDB over QWP instead of to Kafka.

ts is stamped at produce time, so ts is the latency anchor. QuestDB's own WAL
telemetry records when each transaction was committed and applied, so latency is
computed server side after the run (see latency.py); nothing polls.

Writes run metadata (start, end, counts, settings) to runs/<run_id>.json.
"""
import argparse
import heapq
import json
import os
import logging
import random
import shutil
import signal
import sys
import time

import psutil
import questdb
from questdb import TimestampNanos

from common import QDB_ADDR, RUNS_DIR

EVENT_TYPES = ["click", "view", "purchase", "add_to_cart", "logout", "login"]
COUNTRIES = ["us", "es", "de", "fr", "br", "in", "jp", "gb"]

_stop = False

# Hard floor for free disk space on the volume holding QuestDB's data. Below it the
# producer stops at once: no duplicate drain, no further rows.
MIN_FREE_BYTES = 10 * 1024 ** 3


def free_bytes(path):
    return shutil.disk_usage(path).free


def _handle_stop(_sig, _frame):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default=QDB_ADDR, help="QuestDB host:port (default: QDB_ADDR)")
    ap.add_argument("--batch", type=int, help="rows per flush (auto_flush_rows); client default "
                                               "when omitted")
    ap.add_argument("--sf-dir", help="enable store-and-forward: parent dir for the on-disk "
                                     "buffer (absolute, must exist); sender_id is the table name")
    ap.add_argument("--conf", help="full connect string; overrides --addr, --batch and --sf-dir")
    ap.add_argument("--conf-extra", default="",
                    help="extra connect-string keys appended to the built string, for example "
                         "reconnect_max_backoff_millis=500;")
    ap.add_argument("--delay", type=float, default=0,
                    help="seconds to wait before connecting (stagger concurrent producers, or "
                         "start a load step partway through another producer's run)")
    ap.add_argument("--table", default="bench_events")
    ap.add_argument("--rate", type=int, default=50000, help="original events/s; dups are extra")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keys", type=int, default=1000000)
    ap.add_argument("--dup-rate", type=float, default=0.1)
    ap.add_argument("--dup-max-delay-ms", type=int, default=3000)
    ap.add_argument("--id-base", type=int, default=0)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%S"))
    ap.add_argument("--disk-path", default=os.path.expanduser("~"),
                    help="a path on the volume holding QuestDB's data, for the free space check")
    ap.add_argument("--quiet", action="store_true", help="no per-second progress lines")
    ap.add_argument("--qdb-pid", type=int, default=0,
                    help="QuestDB server PID; when set, its CPU and RSS are sampled every second")
    args = ap.parse_args()
    if not args.conf:
        args.conf = f"ws::addr={args.addr};"
        if args.batch:
            args.conf += f"auto_flush_rows={args.batch};auto_flush_interval=1000;"
        if args.sf_dir:
            args.conf += f"sf_dir={args.sf_dir};sender_id={args.table};"
        args.conf += args.conf_extra
    if not args.quiet:
        # The client logs connection events (disconnects, reconnect attempts) through logging.
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")

    qdb_proc = psutil.Process(args.qdb_pid) if args.qdb_pid else None
    self_proc = psutil.Process()
    if qdb_proc:
        qdb_proc.cpu_percent(None)
    self_proc.cpu_percent(None)
    samples = []   # (elapsed_s, qdb_cpu_cores, qdb_rss_gib, producer_cpu_cores, free_disk_gib)

    if args.delay:
        time.sleep(args.delay)
    if free_bytes(args.disk_path) < MIN_FREE_BYTES:
        sys.exit(f"[producer] ABORT: free disk below 10 GB on {args.disk_path}, not starting")
    aborted = False

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    rng = random.Random(args.seed)
    slice_s = 0.01
    per_slice = max(1, round(args.rate * slice_s))
    pending = []   # duplicate heap: (due_wallclock, event_id, row kwargs)
    seq = args.id_base
    sent = originals = dups = 0
    behind_slices = 0

    print(f"[producer] {args.conf} table={args.table} rate={args.rate}/s seconds={args.seconds} "
          f"keys={args.keys} dup={args.dup_rate}<={args.dup_max_delay_ms}ms", flush=True)

    with questdb.connect(args.conf) as db:
        with db.sender() as sender:
            started = time.time()
            next_slice = started
            last_report = started

            def send(row):
                sender.row(args.table,
                           symbols={"event_type": row["event_type"], "country": row["country"]},
                           columns={"event_id": row["event_id"], "user_id": row["user_id"],
                                    "amount": row["amount"]},
                           at=TimestampNanos(row["ts_ns"]))

            while not _stop:
                now = time.time()
                if now - started >= args.seconds:
                    break
                for _ in range(per_slice):
                    seq += 1
                    row = {
                        "event_id": seq,
                        "user_id": rng.randint(1, args.keys),
                        "event_type": rng.choice(EVENT_TYPES),
                        "country": rng.choice(COUNTRIES),
                        "amount": round(rng.uniform(0.0, 500.0), 2),
                        "ts_ns": time.time_ns(),
                    }
                    send(row)
                    sent += 1
                    originals += 1
                    if args.dup_rate > 0 and rng.random() < args.dup_rate:
                        delay = rng.uniform(0.0, args.dup_max_delay_ms / 1000.0)
                        heapq.heappush(pending, (row["ts_ns"] / 1e9 + delay, seq, row))

                now = time.time()
                while pending and pending[0][0] <= now:
                    _, _, row = heapq.heappop(pending)
                    send(row)
                    sent += 1
                    dups += 1

                if now - last_report >= 1.0:
                    free = free_bytes(args.disk_path)
                    if free < MIN_FREE_BYTES:
                        print(f"[producer] ABORT: free disk {free / 1024 ** 3:.1f} GB is below 10 GB",
                              flush=True)
                        aborted = True
                        break
                    elapsed = now - started
                    samples.append((
                        round(elapsed, 2),
                        round(qdb_proc.cpu_percent(None) / 100.0, 3) if qdb_proc else None,
                        round(qdb_proc.memory_info().rss / 1024 ** 3, 3) if qdb_proc else None,
                        round(self_proc.cpu_percent(None) / 100.0, 3),
                        round(free / 1024 ** 3, 2),
                    ))
                    if not args.quiet:
                        print(f"[producer] {time.strftime('%H:%M:%S', time.gmtime(now))} "
                              f"sent={sent} originals={originals} dups={dups} "
                              f"elapsed={elapsed:5.1f}s avg_rate={sent / elapsed:8.1f}/s "
                              f"behind_slices={behind_slices}", flush=True)
                    last_report = now

                next_slice += slice_s
                delay = next_slice - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    behind_slices += 1
                    next_slice = time.time()

            # Send every scheduled duplicate at its due time.
            while pending and not _stop and not aborted:
                wait = pending[0][0] - time.time()
                if wait > 0:
                    time.sleep(min(wait, 0.01))
                    continue
                _, _, row = heapq.heappop(pending)
                send(row)
                sent += 1
                dups += 1
            sender.flush(wait=True)
            ended = time.time()

    meta = {
        "run_id": args.run_id, "table": args.table, "conf": args.conf,
        "rate": args.rate, "seconds": args.seconds, "keys": args.keys,
        "dup_rate": args.dup_rate, "dup_max_delay_ms": args.dup_max_delay_ms,
        "seed": args.seed, "id_base": args.id_base,
        "started": started, "ended": ended,
        "sent": sent, "originals": originals, "dups": dups,
        "first_event_id": args.id_base + 1, "last_event_id": seq,
        "behind_slices": behind_slices, "aborted_low_disk": aborted,
        "samples_cols": ["elapsed_s", "qdb_cpu_cores", "qdb_rss_gib", "producer_cpu_cores",
                         "free_disk_gib"],
        "samples": samples,
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{args.run_id}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[producer] DONE sent={sent} originals={originals} dups={dups} "
          f"in {ended - started:.1f}s avg={sent / (ended - started):.1f}/s -> {path}", flush=True)


if __name__ == "__main__":
    main()
