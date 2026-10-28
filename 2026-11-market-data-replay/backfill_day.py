#!/usr/bin/env python3
"""Ingest one full trading day of EQUS.MINI history into QuestDB.

Picks the latest complete US trading day, skipping weekends, market holidays
and early-close days, writes a Historical connector configuration for it, and
runs the connector. The day covers the whole EQUS.MINI session, 04:00 to 20:00
New York time, for top of book (mbp-1) and trades.

Databento streams explicit symbol lists far faster than ALL_SYMBOLS, and
accepts up to 2,000 symbols per request. For larger lists, including
--all-symbols, the script downloads batches of symbols in parallel, each into
its own staging table, so every table is written in time order. It then merges
the staging tables into the final tables window by window in time order, and
drops them.

    DATABENTO_API_KEY      Databento API key with Historical access
    QDB_CLIENT_CONF        QuestDB client configuration string
    DATABENTO_QUESTDB_BIN  Connector binary (default: databento-questdb on PATH)

Usage:
    python3 backfill_day.py                     # NVDA and AAPL, latest full trading day
    python3 backfill_day.py --all-symbols       # every symbol, about 30 GB
    python3 backfill_day.py --date 2026-10-23 --symbols AAPL,MSFT
    python3 backfill_day.py --dry-run           # pick the day and estimate only
"""

import argparse
import base64
import csv
import datetime as dt
import io
import json
import os
import shutil
import signal
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

DATASET = "EQUS.MINI"
SCHEMAS = {"mbp-1": "orderbook", "trades": "trades"}
NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN, SESSION_CLOSE = dt.time(4, 0), dt.time(20, 0)
# A regular day prints thousands of SPY trades between 15:00 and 16:00 New York
# time. Holidays print none and early-close days a handful.
CLOSE_CHECK_SYMBOL, CLOSE_CHECK_MIN_TRADES = "SPY", 1000
MAX_LOOKBACK_DAYS = 14
MAX_SYMBOLS_PER_REQUEST = 2000
HERE = Path(__file__).resolve().parent

# Column layouts from the connector's table contract.
COMMON_COLUMNS = [
    ("dataset", "SYMBOL"), ("symbol", "SYMBOL CAPACITY 16384"), ("publisher_id", "INT"),
    ("instrument_id", "LONG"), ("ts_event", "TIMESTAMP_NS"), ("price", "DOUBLE"), ("size", "LONG"),
    ("action", "SYMBOL"), ("side", "SYMBOL"), ("flags", "SHORT"), ("depth", "SHORT"),
    ("ts_recv", "TIMESTAMP_NS"), ("ts_in_delta", "LONG"), ("sequence", "LONG"),
]
TABLE_COLUMNS = {
    "orderbook": COMMON_COLUMNS + [("bids", "DOUBLE[][]"), ("asks", "DOUBLE[][]"),
                                   ("bid_ct", "DOUBLE[]"), ("ask_ct", "DOUBLE[]")],
    "trades": COMMON_COLUMNS,
}


class Databento:
    """Databento Historical HTTP API. Metadata calls are not billed."""

    def __init__(self, api_key):
        token = base64.b64encode(f"{api_key}:".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}"}

    def call(self, endpoint, data=None, timeout=60, **params):
        url = f"https://hist.databento.com/v0/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body = urllib.parse.urlencode(data).encode() if data else None
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=self.headers),
                                        timeout=timeout) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as e:
            sys.exit(f"Databento {endpoint} failed ({e.code}): {e.read().decode()[:500]}")

    def meta(self, endpoint, **params):
        return json.loads(self.call(f"metadata.{endpoint}", **params))

    def symbols(self, day, end):
        """Every raw symbol defined for the day. Definitions are published at 00:00 UTC."""
        start = dt.datetime.combine(day, dt.time(0), dt.timezone.utc)
        text = self.call("timeseries.get_range", timeout=900, data=dict(
            dataset=DATASET, schema="definition", symbols="ALL_SYMBOLS", stype_in="raw_symbol",
            start=rfc3339(start), end=rfc3339(end), encoding="csv", compression="none"))
        return sorted({row["raw_symbol"] for row in csv.DictReader(io.StringIO(text))})


class QuestDb:
    """QuestDB REST endpoint derived from the QDB_CLIENT_CONF connection string."""

    def __init__(self, conf, tls_skip_verify):
        protocol, _, rest = conf.partition("::")
        params = dict(p.split("=", 1) for p in rest.split(";") if "=" in p)
        scheme = {"ws": "http", "http": "http", "wss": "https", "https": "https"}.get(protocol)
        if not scheme:
            sys.exit(f"Cannot derive the QuestDB HTTP endpoint from protocol {protocol!r}")
        self.url = f"{scheme}://{params['addr'].split(',')[0]}/exec"
        # Merging a window can run for minutes; raise the server's statement timeout.
        self.headers = {"Statement-Timeout": "1800000"}
        if "token" in params:
            self.headers["Authorization"] = f"Bearer {params['token']}"
        elif "username" in params:
            basic = base64.b64encode(f"{params['username']}:{params.get('password', '')}".encode()).decode()
            self.headers["Authorization"] = f"Basic {basic}"
        self.context = None
        if scheme == "https":
            self.context = ssl.create_default_context(cafile=params.get("tls_roots"))
            if tls_skip_verify:
                self.context.check_hostname = False
                self.context.verify_mode = ssl.CERT_NONE

    def exec(self, sql):
        url = f"{self.url}?" + urllib.parse.urlencode({"query": sql})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers),
                                        context=self.context, timeout=1800) as resp:
                result = json.load(resp)
        except urllib.error.HTTPError as e:
            sys.exit(f"QuestDB query failed ({e.code}): {e.read().decode()[:500]}\n{sql[:300]}")
        if "error" in result:
            sys.exit(f"QuestDB query failed: {result['error']}\n{sql[:300]}")
        return result

    def value(self, sql):
        """First column of the first row, or None. Aggregates over empty tables return no rows."""
        rows = self.exec(sql)["dataset"]
        return rows[0][0] if rows else None

    def table_exists(self, table):
        return bool(self.exec(f"SELECT table_name FROM tables() WHERE table_name = '{table}'")["dataset"])

    def create_table(self, table, kind):
        if not self.table_exists(table):
            columns = ",\n    ".join(f"{name} {kind_}" for name, kind_ in TABLE_COLUMNS[kind])
            self.exec(f"CREATE TABLE {table} (\n    {columns}\n) TIMESTAMP(ts_recv) PARTITION BY DAY WAL")

    def wait_for_wal(self, tables):
        """Wait until QuestDB has applied every WAL transaction for these tables."""
        names = ", ".join(f"'{t}'" for t in tables)
        while True:
            rows = self.exec(f"SELECT name, suspended, writerTxn, sequencerTxn FROM wal_tables() "
                             f"WHERE name IN ({names})")["dataset"]
            if any(row[1] for row in rows):
                sys.exit(f"WAL apply is suspended for {[row[0] for row in rows if row[1]]}")
            if all(row[2] == row[3] for row in rows):
                return
            time.sleep(2)


def utc(day, time_of_day):
    return dt.datetime.combine(day, time_of_day, NEW_YORK).astimezone(dt.timezone.utc)


def rfc3339(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def is_full_trading_day(databento, day):
    count = databento.meta(
        "get_record_count", dataset=DATASET, schema="trades", stype_in="raw_symbol",
        symbols=CLOSE_CHECK_SYMBOL, start=rfc3339(utc(day, dt.time(15))), end=rfc3339(utc(day, dt.time(16))),
    )
    return count >= CLOSE_CHECK_MIN_TRADES


def latest_full_trading_day(databento):
    available_end = dt.datetime.fromisoformat(databento.meta("get_dataset_range", dataset=DATASET)["end"][:19])
    available_end = available_end.replace(tzinfo=dt.timezone.utc)
    day = available_end.astimezone(NEW_YORK).date()
    for _ in range(MAX_LOOKBACK_DAYS):
        if day.weekday() < 5 and utc(day, SESSION_CLOSE) <= available_end:
            if is_full_trading_day(databento, day):
                return day
            print(f"Skipping {day}: market holiday or early close")
        day -= dt.timedelta(days=1)
    sys.exit(f"No full trading day found in the last {MAX_LOOKBACK_DAYS} days")


def estimate(databento, symbols, start, end):
    """Print rows, size and cost per schema; return the total cost in USD."""
    total = 0.0
    print(f"{'schema':8} {'rows':>14} {'size':>10} {'cost':>9}")
    for schema in SCHEMAS:
        params = dict(dataset=DATASET, schema=schema, stype_in="raw_symbol",
                      symbols=",".join(symbols), start=rfc3339(start), end=rfc3339(end))
        rows = databento.meta("get_record_count", **params)
        size = databento.meta("get_billable_size", **params)
        cost = databento.meta("get_cost", **params)
        total += cost
        print(f"{schema:8} {rows:>14,} {size / 1e9:>8.2f}GB {cost:>8.2f}$")
    return total


def toml_string(value):
    return json.dumps(value)


def write_config(path, task, batches, start, end, tables):
    """Write a Historical config with one task per symbol batch.

    The connector runs the tasks concurrently. tables[i] maps each schema
    suffix to the table batch i writes.
    """
    lines = [
        "# Generated by backfill_day.py. Credentials come from the environment.",
        "version = 1",
        'on_task_failure = "exit"',
        "progress_log_interval_s = 30",
        "",
        "[pools.main]",
        'conf_env = "QDB_CLIENT_CONF"',
    ]
    for i, symbols in enumerate(batches):
        lines += [
            "",
            "[[tasks]]",
            f"name = {toml_string(f'{task}-{i + 1:02d}' if len(batches) > 1 else task)}",
            'pool = "main"',
            f"dataset = {toml_string(DATASET)}",
            'mode = "historical"',
            f"end = {toml_string(rfc3339(end))}",
            "# Completed queries are skipped when the same command runs again.",
            f"state_dir = {toml_string(f'./checkpoint/{i + 1:02d}')}",
            "",
            "[tasks.source]",
            'api_key_env = "DATABENTO_API_KEY"',
            "",
            "# No request_timeout_ms: large downloads take a long time.",
            "[tasks.source.http]",
            "connect_timeout_ms = 10000",
        ]
        for schema, suffix in SCHEMAS.items():
            lines += [
                "",
                "[[tasks.subscriptions]]",
                f"schema = {toml_string(schema)}",
                f"symbols = {toml_string(symbols)}",
                'stype_in = "raw_symbol"',
                f"start = {toml_string(rfc3339(start))}",
                f"table = {toml_string(tables[i][suffix])}",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_connector(binary, config, env):
    subprocess.run([binary, "--config", str(config), "--check-config"], check=True, stdout=subprocess.DEVNULL, env=env)
    print("\nIngesting. Ctrl+C stops cleanly; run the same command again to resume.\n", flush=True)
    connector = subprocess.Popen([binary, "--config", str(config)], env=env)
    signal.signal(signal.SIGTERM, lambda *_: connector.send_signal(signal.SIGTERM))
    while True:
        try:
            return connector.wait()
        except KeyboardInterrupt:
            # The connector got the same SIGINT; let it drain and checkpoint.
            print("\nStopping: waiting for the connector to drain and checkpoint", flush=True)


def merge(qdb, kind, stages, final, window):
    """Append the staging tables to the final table in time order, one window at a time.

    Each INSERT sorts one window and lands after everything already in the
    final table, so QuestDB only appends. A rerun continues after the last
    merged window.
    """
    columns = ", ".join(name for name, _ in TABLE_COLUMNS[kind])
    qdb.wait_for_wal(stages + [final])
    first = min(qdb.value(f"SELECT min(ts_recv) FROM {t}") or "9999" for t in stages)
    last = max(qdb.value(f"SELECT max(ts_recv) FROM {t}") or "" for t in stages)
    if first == "9999":
        return
    merged_until = qdb.value(f"SELECT max(ts_recv) FROM {final}")
    lo = dt.datetime.fromisoformat(first[:16]).replace(tzinfo=dt.timezone.utc)
    hi = dt.datetime.fromisoformat(last[:19]).replace(tzinfo=dt.timezone.utc)
    lo -= dt.timedelta(minutes=lo.minute % (window.seconds // 60))
    while lo <= hi:
        upper = lo + window
        if not merged_until or merged_until[:19] < rfc3339(lo)[:19]:
            where = f"ts_recv >= '{rfc3339(lo)}' AND ts_recv < '{rfc3339(upper)}'"
            union = "\n    UNION ALL\n    ".join(f"SELECT {columns} FROM {t} WHERE {where}" for t in stages)
            started = time.monotonic()
            qdb.exec(f"INSERT INTO {final} ({columns})\nSELECT * FROM (\n    {union}\n) ORDER BY ts_recv")
            print(f"Merged {final} {rfc3339(lo)} to {rfc3339(upper)} in {time.monotonic() - started:.1f}s",
                  flush=True)
        lo = upper


def find_connector(explicit):
    binary = explicit or os.environ.get("DATABENTO_QUESTDB_BIN") or shutil.which("databento-questdb")
    if not binary or not Path(binary).is_file():
        sys.exit("Connector binary not found: put databento-questdb on PATH or set DATABENTO_QUESTDB_BIN")
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--symbols", default="NVDA,AAPL", help="comma-separated symbols (default: NVDA,AAPL)")
    scope.add_argument("--all-symbols", action="store_true", help="every symbol in EQUS.MINI, about 30 GB a day")
    parser.add_argument("--date", type=dt.date.fromisoformat,
                        help="trading day as YYYY-MM-DD (default: latest full trading day)")
    parser.add_argument("--table-prefix", default="equs_hist",
                        help="tables are <prefix>_orderbook and <prefix>_trades (default: equs_hist)")
    parser.add_argument("--batch-size", type=int, default=MAX_SYMBOLS_PER_REQUEST,
                        help=f"symbols per Databento request (default and maximum: {MAX_SYMBOLS_PER_REQUEST})")
    parser.add_argument("--merge-minutes", type=int, default=10, choices=[1, 2, 5, 10, 15, 30, 60],
                        help="time window per merge statement (default: 10)")
    parser.add_argument("--bin", help="connector binary (default: DATABENTO_QUESTDB_BIN or PATH)")
    parser.add_argument("--dry-run", action="store_true", help="pick the day, estimate and write the config only")
    parser.add_argument("--yes", action="store_true", help="do not ask before a download Databento would bill")
    parser.add_argument("--tls-skip-verify", action="store_true",
                        help="accept QuestDB's TLS certificate without verifying it, for self-signed test servers")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= MAX_SYMBOLS_PER_REQUEST:
        sys.exit(f"--batch-size must be between 1 and {MAX_SYMBOLS_PER_REQUEST}")

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        sys.exit("Set DATABENTO_API_KEY")
    if not args.dry_run and not os.environ.get("QDB_CLIENT_CONF"):
        sys.exit("Set QDB_CLIENT_CONF, for example ws::addr=localhost:9000;")
    binary = None if args.dry_run else find_connector(args.bin)

    databento = Databento(api_key)
    day = args.date or latest_full_trading_day(databento)
    if day.weekday() >= 5:
        sys.exit(f"{day} is a weekend")
    if args.date and not is_full_trading_day(databento, day):
        print(f"Warning: {day} looks like a market holiday or early close")

    start, end = utc(day, SESSION_OPEN), utc(day, SESSION_CLOSE)
    if args.all_symbols:
        print(f"Trading day {day}: {rfc3339(start)} to {rfc3339(end)} (all symbols)\n")
        cost = estimate(databento, ["ALL_SYMBOLS"], start, end)
        print("\nFetching the symbol list from Databento definitions...", flush=True)
        symbols = databento.symbols(day, end)
        label = "all"
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"Trading day {day}: {rfc3339(start)} to {rfc3339(end)} ({', '.join(symbols)})\n")
        cost = estimate(databento, symbols, start, end)
        label = "-".join(symbols).lower() if len(symbols) <= 5 else f"{len(symbols)}-symbols"

    batches = [symbols[i:i + args.batch_size] for i in range(0, len(symbols), args.batch_size)]
    finals = {suffix: f"{args.table_prefix}_{suffix}" for suffix in SCHEMAS.values()}
    if len(batches) == 1:
        tables = [finals]
    else:
        tables = [{suffix: f"{args.table_prefix}_stage{i + 1:02d}_{suffix}" for suffix in SCHEMAS.values()}
                  for i in range(len(batches))]

    # The connector needs a new state directory when the tables or symbols change.
    task = f"{args.table_prefix}-{day}-{label}".replace("_", "-")
    if len(batches) > 1:
        task += f"-b{args.batch_size}"
    config = HERE / "state" / task / "connector.toml"
    write_config(config, task, batches, start, end, tables)
    print(f"\nTables: {finals['orderbook']}, {finals['trades']}")
    if len(batches) > 1:
        print(f"Requests: {len(symbols):,} symbols in {len(batches)} parallel batches, "
              f"staged in {args.table_prefix}_stageNN_* and merged in time order")
    print(f"Config: {config.relative_to(HERE)}")
    if args.dry_run:
        return 0

    if cost > 0 and not args.yes:
        if not sys.stdin.isatty() or input(f"\nDatabento will bill about ${cost:.2f}. Continue? [y/N] ").lower() != "y":
            sys.exit("Aborted")

    env = dict(os.environ)
    if args.tls_skip_verify:
        # The connector reads QDB_CLIENT_CONF; tls_verify=unsafe_off disables
        # certificate checks for this run only.
        env["QDB_CLIENT_CONF"] = env["QDB_CLIENT_CONF"].rstrip(";") + ";tls_verify=unsafe_off;"
        print("\nWarning: QuestDB TLS certificate verification is off", flush=True)

    qdb = QuestDb(os.environ["QDB_CLIENT_CONF"], args.tls_skip_verify)
    merged_marker = {kind: config.parent / f"merged-{kind}" for kind in finals}
    for batch_tables in tables:
        for kind, table in batch_tables.items():
            if not merged_marker[kind].exists():
                qdb.create_table(table, kind)

    exit_code = run_connector(binary, config, env)
    if exit_code != 0 or len(batches) == 1:
        return exit_code

    print("\nDownload complete. Merging staging tables in time order.\n", flush=True)
    window = dt.timedelta(minutes=args.merge_minutes)
    for kind, final in finals.items():
        if merged_marker[kind].exists():
            continue
        qdb.create_table(final, kind)
        stages = [t[kind] for t in tables]
        merge(qdb, kind, stages, final, window)
        qdb.wait_for_wal([final])
        staged = sum(qdb.value(f"SELECT count() FROM {t}") for t in stages)
        merged = qdb.value(f"SELECT count() FROM {final}")
        if staged != merged:
            sys.exit(f"{final} has {merged:,} rows but the staging tables hold {staged:,}; "
                     f"staging tables kept for inspection")
        merged_marker[kind].touch()
        for table in stages:
            qdb.exec(f"DROP TABLE {table}")
        print(f"{final}: {merged:,} rows; dropped {len(stages)} staging tables", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
