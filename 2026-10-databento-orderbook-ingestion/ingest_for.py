#!/usr/bin/env python3
"""Run the Databento connector for a fixed number of minutes, then stop it cleanly.

Live market data never ends, so for a demo we bound the run: start the connector,
let it stream for --minutes, then send SIGINT. The connector drains in-flight
batches, waits for QuestDB to acknowledge them, and saves its checkpoint before
exiting, so a later run with the same state directory resumes where this one
stopped.

Settings are read from the environment, or from a .env file next to this script:

    DATABENTO_API_KEY      Databento API key with Live access to EQUS.MINI
    QDB_CLIENT_CONF        QuestDB client configuration, e.g. ws::addr=127.0.0.1:9000;
    DATABENTO_QUESTDB_BIN  Optional path to the connector binary (default: on PATH)

Usage:
    python3 ingest_for.py --minutes 3
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Connector drain timeout (30 s default) plus time to persist the final checkpoint.
STOP_GRACE_S = 90


def load_dotenv(path):
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def find_binary(explicit):
    candidate = explicit or os.environ.get("DATABENTO_QUESTDB_BIN") or "databento-questdb"
    resolved = shutil.which(candidate)
    if resolved is None:
        sys.exit(
            f"Connector binary not found: {candidate}\n"
            "Put databento-questdb on PATH or set DATABENTO_QUESTDB_BIN."
        )
    return resolved


def stop(proc):
    """Ask the connector to drain and checkpoint, then wait for it to exit."""
    if proc.poll() is not None:
        return proc.returncode
    print(f"\nStopping connector (SIGINT), waiting up to {STOP_GRACE_S}s to drain...", flush=True)
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        print("Connector did not stop in time; terminating.", file=sys.stderr)
        proc.terminate()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=3.0, help="how long to stream (default: 3)")
    parser.add_argument("--config", default=str(HERE / "connector.toml"), help="connector TOML file")
    parser.add_argument("--bin", help="path to the databento-questdb binary")
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    for name in ("DATABENTO_API_KEY", "QDB_CLIENT_CONF"):
        if not os.environ.get(name):
            sys.exit(f"{name} is not set. Copy .env.example to .env and fill it in.")

    binary = find_binary(args.bin)
    # Validate the configuration without connecting to Databento or QuestDB.
    subprocess.run([binary, "--config", args.config, "--check-config"], check=True)

    print(f"Streaming for {args.minutes:g} minute(s). Press Ctrl+C to stop early.", flush=True)
    # A new session keeps a terminal Ctrl+C from reaching the connector twice:
    # this script forwards a single SIGINT instead.
    proc = subprocess.Popen([binary, "--config", args.config], start_new_session=True)
    deadline = time.monotonic() + args.minutes * 60
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"Connector exited early with code {proc.returncode}.", file=sys.stderr)
                return proc.returncode
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    code = stop(proc)
    print(f"Connector exited with code {code}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
