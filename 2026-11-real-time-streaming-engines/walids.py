#!/usr/bin/env python3
"""
List the WAL writers (walId) that committed to a table, with their first and last commit
and row counts, to map each concurrent producer to its writer for latency.py --wal-id.

Usage: walids.py <table>
"""
import sys

from common import sql, table_id


def main():
    t = sys.argv[1]
    res = sql(f"""
SELECT walId, count() txns, sum(rowCount) rows, min(created) first_commit, max(created) last_commit,
       min(minTimestamp) min_ts
FROM 'sys.telemetry_wal' WHERE tableId = {table_id(t)} AND event = 109
GROUP BY walId ORDER BY first_commit""")
    print(f"{t}: WAL writers")
    for wal, txns, rows, first, last, min_ts in res["dataset"]:
        print(f"  walId={wal:<4} txns={txns:>6} rows={rows:>10} first={first} last={last} min_ts={min_ts}")


if __name__ == "__main__":
    main()
