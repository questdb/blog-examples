#!/usr/bin/env python3
"""
Slice a long server-side CPU sample (runs/<sample>_cpu.json from cpusample.py) by the
start/end epochs of producer runs (runs/<run_id>.json), for runs where the producer and
QuestDB are on different hosts with synced clocks. Skips the first 2 s of each run.

Usage: cpuslice.py <sample_id> <run_id> [<run_id> ...]
"""
import json
import os
import statistics
import sys

from common import RUNS_DIR


def load(name):
    with open(os.path.join(RUNS_DIR, f"{name}.json")) as f:
        return json.load(f)


def main():
    cpu = load(f"{sys.argv[1]}_cpu")
    t0 = cpu["started"]
    for run_id in sys.argv[2:]:
        m = load(run_id)
        s = [x for x in cpu["samples"] if m["started"] + 2 <= t0 + x[0] <= m["ended"]]
        qdb = sorted(x[1] for x in s)
        print(f"{run_id:>22}  n={len(qdb):>3}  qdb_cpu mean={statistics.mean(qdb):.2f} "
              f"p95={qdb[int(len(qdb) * 0.95)]:.2f} max={qdb[-1]:.2f} cores  "
              f"rss_max={max(x[2] for x in s):.2f} GiB")


if __name__ == "__main__":
    main()
