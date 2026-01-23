#!/usr/bin/env python3
"""
Method 2: Parquet Export via REST API

Demonstrates exporting query results to Parquet files.
Best for reproducible training datasets and larger exports.
"""
import sys
sys.path.insert(0, ".")

import pyarrow.parquet as pq

from src.data_loader import export_to_parquet, get_training_query
from config.settings import SYMBOL, DATASET_DIR


def main():
    print("=" * 60)
    print("METHOD 2: Parquet Export via REST API")
    print("=" * 60)
    
    # Export training data
    query = get_training_query(symbol=SYMBOL, days=7)
    output_path = DATASET_DIR / "training_data.parquet"
    
    print(f"\nExporting 7 days of {SYMBOL} data to Parquet...")
    print(f"Output: {output_path}")
    
    export_to_parquet(query, output_path)
    
    # Verify the export
    print("\n" + "-" * 60)
    print("Verifying exported file...")
    
    table = pq.read_table(output_path)
    print(f"  Rows: {table.num_rows:,}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Schema:")
    for field in table.schema:
        print(f"    - {field.name}: {field.type}")
    
    # Show sample
    df = table.to_pandas()
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Parquet files are compressed and efficient")
    print("  - Schema and types are preserved")
    print("  - File can be versioned and shared")
    print(f"  - Ready for training: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
