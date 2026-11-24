#!/usr/bin/env python3
"""
Create a fresh benchmark base table, optionally with the benchmark views on it.

Usage: mktable.py <table> [--clean-mv] [--clean-view] [--s1-mv] [--dedup UPSERT|SKIP]
  --clean-mv    passthrough materialized view <table>_clean_mv (stateless transform, stored)
  --clean-view  plain view <table>_clean_v (same transform, computed at query time)
  --s1-mv       SAMPLE BY 10s materialized view <table>_s1_10s (S1 windowed aggregation)
Refuses to touch a table that already exists.
"""
import argparse
import sys

from common import sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--clean-mv", action="store_true")
    ap.add_argument("--clean-view", action="store_true")
    ap.add_argument("--s1-mv", action="store_true")
    ap.add_argument("--dedup", choices=["UPSERT", "SKIP"], default="UPSERT")
    args = ap.parse_args()
    t = args.table

    exists = sql(f"SELECT count() FROM tables() WHERE table_name = '{t}'")["dataset"][0][0]
    if exists:
        sys.exit(f"{t} already exists; pick a new name")

    sql(f"CREATE TABLE {t} (ts TIMESTAMP, event_id LONG, user_id LONG, event_type SYMBOL, "
        f"country SYMBOL, amount DOUBLE) TIMESTAMP(ts) PARTITION BY HOUR "
        f"DEDUP {args.dedup} KEYS(ts, event_id)")
    print(f"created {t} (DEDUP {args.dedup})")
    make_views(t, args.clean_mv, args.clean_view, args.s1_mv)


def make_views(t, clean_mv=False, clean_view=False, s1_mv=False):
    clean = (f"SELECT ts, event_id, user_id, event_type, amount * 1.21 amount_with_tax, "
             f"upper(country) country FROM {t} "
             f"WHERE event_type IN ('purchase', 'add_to_cart') AND amount > 0")
    if clean_mv:
        sql(f"CREATE MATERIALIZED VIEW {t}_clean_mv AS ({clean})")
        print(f"created {t}_clean_mv")
    if clean_view:
        sql(f"CREATE VIEW {t}_clean_v AS ({clean})")
        print(f"created {t}_clean_v")
    if s1_mv:
        sql(f"CREATE MATERIALIZED VIEW {t}_s1_10s AS (SELECT ts, user_id, count() cnt, "
            f"sum(amount) sum_amount, max(amount) max_amount, max(ts) max_ts FROM {t} "
            f"SAMPLE BY 10s) PARTITION BY HOUR")
        print(f"created {t}_s1_10s")


if __name__ == "__main__":
    main()
