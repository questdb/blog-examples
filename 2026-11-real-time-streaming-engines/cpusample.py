#!/usr/bin/env python3
"""
Sample a process's CPU and RSS once a second (QuestDB by default), for runs where the
producer is on another host. Optionally also samples a second process on the same host
(for example the Kafka Connect worker next to QuestDB).

Usage: cpusample.py <run_id> <seconds> [--pattern io.questdb.ServerMain] [--extra PATTERN]
  --pattern  substring of the main process's command line (kafka.Kafka for the broker)
  --extra    substring of a second process's command line (ConnectStandalone for Connect)
Writes runs/<run_id>_cpu.json: per second elapsed_s, main CPU cores, main RSS GiB, extra
CPU cores, free GiB on DATA_MOUNT (warned about below 20 GiB), extra RSS GiB.
"""
import argparse
import json
import os
import shutil
import sys
import time

import psutil

from common import RUNS_DIR

DATA = os.environ.get("DATA_MOUNT", "/data")


def find(pattern):
    me = os.getpid()
    for p in psutil.process_iter(["cmdline"]):
        if p.pid != me and pattern in " ".join(p.info["cmdline"] or []):
            return p
    sys.exit(f"no process matching {pattern!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("seconds", type=int)
    ap.add_argument("--pattern", default="io.questdb.ServerMain")
    ap.add_argument("--extra")
    args = ap.parse_args()
    p = find(args.pattern)
    x = find(args.extra) if args.extra else None
    for proc in (p, x):
        if proc:
            proc.cpu_percent(None)
    data = DATA if os.path.exists(DATA) else "/"
    samples = []
    started = time.time()
    while time.time() - started < args.seconds:
        time.sleep(1.0)
        free = shutil.disk_usage(data).free / 1024 ** 3
        if free < 20:
            print(f"[cpusample] WARNING: {data} free {free:.1f} GiB", flush=True)
        samples.append((round(time.time() - started, 2), round(p.cpu_percent(None) / 100.0, 3),
                        round(p.memory_info().rss / 1024 ** 3, 3),
                        round(x.cpu_percent(None) / 100.0, 3) if x else None, round(free, 2),
                        round(x.memory_info().rss / 1024 ** 3, 3) if x else None))
    out = os.path.join(RUNS_DIR, f"{args.run_id}_cpu.json")
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"run_id": args.run_id, "pattern": args.pattern, "extra": args.extra,
                   "pid": p.pid, "started": started,
                   "samples_cols": ["elapsed_s", "cpu_cores", "rss_gib", "extra_cpu_cores",
                                    "free_disk_gib", "extra_rss_gib"],
                   "samples": samples}, f, indent=2)
    main_cpu = [s[1] for s in samples[2:]]
    msg = f"[cpusample] {args.run_id} {args.pattern} pid={p.pid} mean={sum(main_cpu) / len(main_cpu):.2f} " \
          f"max={max(main_cpu):.2f} cores"
    if x:
        extra_cpu = [s[3] for s in samples[2:]]
        msg += f"; {args.extra} pid={x.pid} mean={sum(extra_cpu) / len(extra_cpu):.2f} " \
               f"max={max(extra_cpu):.2f} cores"
    print(f"{msg} -> {out}")


if __name__ == "__main__":
    main()
