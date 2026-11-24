#!/usr/bin/env python3
"""
Archive what the latency and resource analysis needs before a server goes away: table
metadata, materialized view status, partitions, and the WAL telemetry (sys.telemetry_wal)
of every table whose name starts with <prefix>, as CSV files under runs/archive/.

latency.py, view_latency.py and cover_latency.py read the telemetry plus the rows' ts;
the rows themselves are not archived (tens of millions per run), only their per-second
counts, which is enough to re-check totals and rates.

Usage: archive.py <prefix> [out_dir under runs/, default archive]
"""
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from common import QDB_HTTP, RUNS_DIR, sql


def export(query, path):
    """Write a query's CSV export to path; returns bytes written, 0 if the server rejects it
    (for example table_partitions() on a plain view, which has no partitions)."""
    url = f"{QDB_HTTP}/exp?{urllib.parse.urlencode({'query': query})}"
    try:
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        print(f"  skipped {os.path.basename(path)}: {e.read().decode(errors='replace')[:200]}")
        return 0
    return os.path.getsize(path)


def main():
    prefix = sys.argv[1]
    out = os.path.join(RUNS_DIR, sys.argv[2] if len(sys.argv) > 2 else "archive")
    os.makedirs(out, exist_ok=True)
    like = f"table_name LIKE '{prefix}%'"
    export(f"SELECT * FROM tables() WHERE {like}", os.path.join(out, f"{prefix}tables.csv"))
    export(f"SELECT * FROM materialized_views() WHERE view_name LIKE '{prefix}%'",
           os.path.join(out, f"{prefix}materialized_views.csv"))
    names = [r[0] for r in sql(f"SELECT table_name FROM tables() WHERE {like} ORDER BY id")["dataset"]]
    total = 0
    for t in names:
        total += export(f"SELECT * FROM 'sys.telemetry_wal' WHERE tableId = "
                        f"(SELECT id FROM tables() WHERE table_name = '{t}')",
                        os.path.join(out, f"{t}_telemetry_wal.csv"))
        total += export(f"SELECT * FROM table_partitions('{t}')",
                        os.path.join(out, f"{t}_partitions.csv"))
        cols = [r[0] for r in sql(f"SELECT \"column\" FROM table_columns('{t}')")["dataset"]]
        if "ts" in cols:
            total += export(f"SELECT ts, count() rows FROM {t} SAMPLE BY 1s",
                            os.path.join(out, f"{t}_rows_per_s.csv"))
        print(f"archived {t}", flush=True)
    print(f"{len(names)} tables, {total / 1024 ** 2:.1f} MiB -> {out}")


if __name__ == "__main__":
    main()
