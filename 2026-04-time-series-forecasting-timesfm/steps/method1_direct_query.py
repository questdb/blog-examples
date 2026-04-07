#!/usr/bin/env python3
"""
Method 1: Direct SQL Queries

Demonstrates querying QuestDB directly via REST API and ConnectorX.
Best for exploration and smaller datasets.
"""
import sys
sys.path.insert(0, ".")

from src.data_loader import query_rest_api, query_connectorx, get_ohlcv_query
from config.settings import SYMBOL, QUESTDB_HTTP, QUESTDB_PG


def main():
    query = get_ohlcv_query(symbol=SYMBOL, days=1)
    
    print("=" * 60)
    print("METHOD 1: Direct SQL Queries")
    print("=" * 60)
    print("\nConnection settings (from config/settings.py):")
    print(f"  REST API:   {QUESTDB_HTTP}")
    print(f"  ConnectorX: {QUESTDB_PG}")
    
    # Method 1a: REST API
    print("\n" + "-" * 60)
    print("1a. REST API")
    print("-" * 60)
    print(f"Connecting to: {QUESTDB_HTTP}")
    print(f"Query: {SYMBOL} OHLCV, 1-minute bars, last 24 hours")
    
    df = query_rest_api(query)
    print(f"\n✓ Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nLast 5 rows:")
    print(df.tail().to_string(index=False))
    
    # Method 1b: ConnectorX
    print("\n" + "-" * 60)
    print("1b. ConnectorX (PostgreSQL wire protocol)")
    print("-" * 60)
    print(f"Connecting to: {QUESTDB_PG}")
    print(f"Query: {SYMBOL} OHLCV, 1-minute bars, last 24 hours")
    print()
    
    try:
        df_cx = query_connectorx(query)
        print(f"✓ Loaded {len(df_cx):,} rows")
        print(f"\nFirst 5 rows:")
        print(df_cx.head().to_string(index=False))
    except ImportError:
        print("⚠ ConnectorX not installed. Run: pip install connectorx")
    except Exception as e:
        print(f"⚠ ConnectorX failed: {e}")
        print("\n  This is expected if you don't have local QuestDB running.")
        print("  ConnectorX requires PostgreSQL wire protocol access (port 8812),")
        print("  which the public demo instance doesn't expose.")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
REST API (demo.questdb.io)
  - Works out of the box, no local setup needed
  - Good for exploration and moderate datasets
  
ConnectorX (localhost:8812)
  - Requires local QuestDB with PostgreSQL wire protocol
  - ~5x faster for large result sets (100K+ rows)
  - Edit config/settings.py to change connection URLs

Both methods return pandas DataFrames ready for analysis or ML.
""")
    
    return df


if __name__ == "__main__":
    main()
