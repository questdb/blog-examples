#!/usr/bin/env python3
"""Run the named statements in a .sql file against QuestDB and print the results.

Statements are separated by `-- name: <name>` comment lines. Uses the QuestDB
REST API and the Python standard library only.

    QDB_HTTP   QuestDB HTTP endpoint (default: http://127.0.0.1:9000)
    QDB_TOKEN  QuestDB REST token, sent as a bearer token

Usage:
    python3 run_sql.py sql/queries.sql                 # run every statement
    python3 run_sql.py sql/queries.sql latest_quotes   # run one by name
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

QDB_HTTP = os.environ.get("QDB_HTTP", "http://127.0.0.1:9000").rstrip("/")
QDB_TOKEN = os.environ.get("QDB_TOKEN")
MAX_ROWS = 20


def parse_statements(path):
    """Return [(name, sql)] from `-- name:` delimited blocks."""
    statements, name, lines = [], None, []
    with open(path) as f:
        for line in f:
            if line.startswith("-- name:"):
                if name and "".join(lines).strip():
                    statements.append((name, "".join(lines).strip().rstrip(";")))
                name, lines = line.split(":", 1)[1].strip(), []
            elif name:
                lines.append(line)
    if name and "".join(lines).strip():
        statements.append((name, "".join(lines).strip().rstrip(";")))
    return statements


def execute(sql):
    url = f"{QDB_HTTP}/exec?" + urllib.parse.urlencode({"query": sql, "limit": MAX_ROWS})
    headers = {"Authorization": f"Bearer {QDB_TOKEN}"} if QDB_TOKEN else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        return json.load(e)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return json.dumps(value)
    return "" if value is None else str(value)


def print_result(result):
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    if "columns" not in result:
        print(f"  OK ({result.get('ddl') or result.get('dml') or 'done'})")
        return
    headers = [c["name"] for c in result["columns"]]
    rows = [[fmt(v) for v in row] for row in result["dataset"]]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))
    total = result.get("count", len(rows))
    if total > len(rows):
        print(f"  ... {total - len(rows)} more row(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("names", nargs="*", help="statement names to run (default: all)")
    args = parser.parse_args()

    statements = parse_statements(args.file)
    if args.names:
        unknown = set(args.names) - {n for n, _ in statements}
        if unknown:
            sys.exit(f"Unknown statement(s): {', '.join(sorted(unknown))}")
        statements = [(n, s) for n, s in statements if n in args.names]

    failed = False
    for name, sql in statements:
        print(f"\n== {name}")
        result = execute(sql)
        failed |= "error" in result
        print_result(result)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
