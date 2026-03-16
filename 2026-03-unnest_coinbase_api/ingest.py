"""
Coinbase Exchange → QuestDB continuous ingestion via REST polling + ILP.

Polls order book snapshots and recent trades for BTC-USD.
Stores raw JSON payloads with no client-side parsing.
All transformation happens at query time in SQL:
  - Order book: json_extract + DOUBLE[][] cast
  - Trades: JSON UNNEST with COLUMNS()

Requirements:
    pip install -r requirements.txt

Usage:
    python ingest.py

QuestDB must be running with HTTP on port 9000 and PG wire on port 8812.
"""

import time

import psycopg2
import requests
from questdb.ingress import Sender, ServerTimestamp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRODUCTS = ["BTC-USD"]

BASE_URL = "https://api.exchange.coinbase.com"

# Both orderbook and trades polled every cycle.
# 1 product × 2 endpoints = 2 req/cycle at 0.75s = 9,600/hr (under 10K limit)
POLL_INTERVAL = 0.75

# QuestDB ILP over HTTP (autoflush defaults: every 1s or 75K rows)
QDB_CONF = "http::addr=localhost:9000;"

# QuestDB PG wire (for DDL only)
QDB_PG = {
    "host": "localhost",
    "port": 8812,
    "user": "admin",
    "password": "quest",
    "dbname": "qdb",
}

# ---------------------------------------------------------------------------
# DDL (via PG wire)
# ---------------------------------------------------------------------------

DDL = [
    """
    CREATE TABLE IF NOT EXISTS cb_orderbook_raw (
        timestamp TIMESTAMP_NS,
        symbol SYMBOL,
        payload VARCHAR
    ) TIMESTAMP(timestamp) PARTITION BY HOUR
    """,
    """
    CREATE TABLE IF NOT EXISTS cb_trades_raw (
        timestamp TIMESTAMP_NS,
        symbol SYMBOL,
        payload VARCHAR
    ) TIMESTAMP(timestamp) PARTITION BY HOUR
    """,
]


def ensure_tables():
    conn = psycopg2.connect(**QDB_PG)
    try:
        with conn.cursor() as cur:
            for stmt in DDL:
                cur.execute(stmt)
        print("Tables ready.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Coinbase Exchange API
# ---------------------------------------------------------------------------

def fetch_orderbook(product_id):
    """Fetch level 2 order book. Returns raw response text."""
    url = f"{BASE_URL}/products/{product_id}/book?level=2"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        return resp.text
    print(f"  WARN orderbook {product_id}: HTTP {resp.status_code}")
    return None


def fetch_trades(product_id, limit=250):
    """Fetch latest trades. Returns raw response text."""
    url = f"{BASE_URL}/products/{product_id}/trades?limit={limit}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        return resp.text
    print(f"  WARN trades {product_id}: HTTP {resp.status_code}")
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def ingest():
    ensure_tables()

    print(f"Polling: {', '.join(PRODUCTS)}")
    print(f"Interval: {POLL_INTERVAL}s\n")

    book_count = 0
    trade_count = 0

    with Sender.from_conf(QDB_CONF) as sender:
        print("QuestDB ILP sender ready\n")

        while True:
            cycle_start = time.monotonic()

            for product in PRODUCTS:
                # --- Order book ---
                payload = fetch_orderbook(product)
                if payload:
                    sender.row(
                        "cb_orderbook_raw",
                        symbols={"symbol": product},
                        columns={"payload": payload},
                        at=ServerTimestamp,
                    )
                    book_count += 1

                # --- Trades ---
                payload = fetch_trades(product)
                if payload:
                    sender.row(
                        "cb_trades_raw",
                        symbols={"symbol": product},
                        columns={"payload": payload},
                        at=ServerTimestamp,
                    )
                    trade_count += 1

            print(
                f"  books: {book_count}  |  trade batches: {trade_count}"
            )

            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, POLL_INTERVAL - elapsed)
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ingest()
    except KeyboardInterrupt:
        print("\nStopped.")
