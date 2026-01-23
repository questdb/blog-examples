#!/usr/bin/env python3
"""
Method 1: Direct SQL Queries

Demonstrates querying QuestDB directly via REST API and ConnectorX.
Best for exploration and smaller datasets.
"""
import sys
sys.path.insert(0, ".")

from src.data_loader import query_rest_api, query_connectorx, get_training_query
from config.settings import SYMBOL


def main():
    query = get_training_query(symbol=SYMBOL, days=1)  # Just 1 day for demo
    
    print("=" * 60)
    print("METHOD 1: Direct SQL Queries")
    print("=" * 60)
    
    # Method 1a: REST API
    print("\n1a. Querying via REST API...")
    print(f"    Query: SELECT ... FROM trades WHERE symbol='{SYMBOL}' SAMPLE BY 1m")
    
    df_rest = query_rest_api(query)
    print(f"    ✓ Loaded {len(df_rest)} rows")
    print(f"    Columns: {list(df_rest.columns)}")
    print(f"\n    First 5 rows:")
    print(df_rest.head().to_string(index=False))
    
    # Method 1b: ConnectorX (faster for large results)
    print("\n" + "-" * 60)
    print("\n1b. Querying via ConnectorX (PostgreSQL wire protocol)...")
    
    try:
        df_cx = query_connectorx(query)
        print(f"    ✓ Loaded {len(df_cx)} rows")
    except ImportError:
        print("    ⚠ ConnectorX not installed. Run: pip install connectorx")
    except Exception as e:
        print(f"    ⚠ ConnectorX error: {e}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - REST API works everywhere, simple to use")
    print("  - ConnectorX is faster for larger result sets")
    print("  - Both return pandas DataFrames ready for analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
