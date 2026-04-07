#!/usr/bin/env python3
"""
Method 3: Direct Parquet Partition Access

Demonstrates reading QuestDB's Parquet partitions directly.
Best for terabyte-scale data and streaming processing.

NOTE: This requires:
  1. Local or mounted access to QuestDB's data directory
  2. Parquet partition format enabled in QuestDB config
  
If you're using the remote demo instance, this won't work—use Method 1 or 2.
"""
import sys
sys.path.insert(0, ".")

from config.settings import QUESTDB_DATA_DIR, TABLE_NAME


def main():
    print("=" * 60)
    print("METHOD 3: Direct Parquet Partition Access")
    print("=" * 60)
    
    print(f"\nLooking for partitions in: {QUESTDB_DATA_DIR / TABLE_NAME}")
    
    table_dir = QUESTDB_DATA_DIR / TABLE_NAME
    
    if not table_dir.exists():
        print(f"""
    ⚠ Directory not found: {table_dir}
    
    This method requires local QuestDB access with Parquet partitions.
    
    To enable Parquet partitions in QuestDB:
    
    1. In server.conf, set:
       cairo.o3.partition.format=PARQUET
    
    2. Or convert existing table partitions:
       ALTER TABLE {TABLE_NAME} SET PARAM o3PartitionFormat = 'PARQUET';
    
    For the remote demo instance, use Method 1 or Method 2 instead.
        """)
        return
    
    try:
        import pyarrow.dataset as ds
        
        parquet_files = list(table_dir.glob("**/*.parquet"))
        
        if not parquet_files:
            print(f"    ⚠ No .parquet files found in {table_dir}")
            return
        
        print(f"    ✓ Found {len(parquet_files)} partition files")
        
        # Load as dataset
        dataset = ds.dataset(table_dir, format="parquet")
        print(f"    ✓ Schema:")
        for field in dataset.schema:
            print(f"      - {field.name}: {field.type}")
        
        # Demo streaming
        print("\n    Streaming first 3 batches...")
        scanner = dataset.scanner(batch_size=10000)
        
        for i, batch in enumerate(scanner.to_batches()):
            if i >= 3:
                break
            print(f"      Batch {i}: {batch.num_rows} rows")
        
        print("\n" + "=" * 60)
        print("Summary:")
        print("  - Direct partition access bypasses query engine")
        print("  - PyArrow dataset API enables efficient streaming")
        print("  - Best for large-scale batch processing")
        print("=" * 60)
        
    except Exception as e:
        print(f"    ⚠ Error: {e}")


if __name__ == "__main__":
    main()
