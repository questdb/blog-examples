#!/usr/bin/env python3
"""Print CPU and memory summaries for one or more runs/<run_id>.json files."""
import json
import os
import statistics
import sys

from common import RUNS_DIR

for run_id in sys.argv[1:]:
    with open(os.path.join(RUNS_DIR, f"{run_id}.json")) as f:
        m = json.load(f)
    s = m["samples"][2:]   # skip the first two seconds of warm-up
    qdb = [x[1] for x in s if x[1] is not None]
    prod = [x[3] for x in s]
    rss = [x[2] for x in m["samples"] if x[2] is not None]
    print(f"{run_id:>16}  table={m['table']:<16} originals={m['originals']:>9} dups={m['dups']:>8}  "
          f"qdb_cpu mean={statistics.mean(qdb):.2f} p95={sorted(qdb)[int(len(qdb) * 0.95)]:.2f} "
          f"max={max(qdb):.2f} cores  qdb_rss_max={max(rss):.2f} GiB  "
          f"producer_cpu={statistics.mean(prod):.2f}  free_disk_end={m['samples'][-1][4]} GiB")
