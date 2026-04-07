#!/usr/bin/env python3
"""
Method 2: Parquet Export via REST API

Demonstrates exporting query results to Parquet files.
Best for reproducible datasets and larger exports.
"""
import sys
sys.path.insert(0, ".")

import pyarrow.parquet as pq

from src.data_loader import export_to_parquet, get_ohlcv_query
from config.settings import SYMBOL, DATASET_DIR


def main():
    print("=" * 60)
    print("METHOD 2: Parquet Export via REST API")
    print("=" * 60)
    
    # Export OHLCV data
    query = get_ohlcv_query(symbol=SYMBOL, days=7)
    output_path = DATASET_DIR / "btc_ohlcv_7d.parquet"
    
    print(f"\nExporting 7 days of {SYMBOL} OHLCV data...")
    print(f"Output: {output_path}")
    
    export_to_parquet(query, output_path)
    
    # Verify the export
    print("\n" + "-" * 60)
    print("Verifying exported file...")
    
    table = pq.read_table(output_path)
    print(f"  Rows:    {table.num_rows:,}")
    print(f"  Size:    {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Schema:")
    for field in table.schema:
        print(f"    - {field.name}: {field.type}")
    
    # Show sample
    df = table.to_pandas()
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Parquet preserves types and compresses well")
    print("  - File can be versioned and shared")
    print(f"  - Ready for forecasting: {output_path}")
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    main()
