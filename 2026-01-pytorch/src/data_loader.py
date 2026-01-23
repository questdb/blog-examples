"""
Three methods to load data from QuestDB into Python/PyTorch.
"""
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

from config.settings import QUESTDB_HTTP, QUESTDB_PG


# =============================================================================
# Method 1: Direct Queries
# =============================================================================

def query_rest_api(query: str) -> pd.DataFrame:
    """
    Query QuestDB via REST API, return DataFrame.
    
    Simple and works everywhere. Best for exploration and smaller datasets.
    """
    response = requests.get(
        f"{QUESTDB_HTTP}/exp",
        params={"query": query, "fmt": "csv"}
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def query_connectorx(query: str) -> pd.DataFrame:
    """
    Query QuestDB via PostgreSQL wire protocol using ConnectorX.
    
    Faster than REST for larger results. Requires: pip install connectorx
    """
    import connectorx as cx
    return cx.read_sql(QUESTDB_PG, query)


# =============================================================================
# Method 2: Parquet Export via REST API
# =============================================================================

def export_to_parquet(query: str, output_path: Path) -> Path:
    """
    Export query results to Parquet file via QuestDB REST API.
    
    Falls back to CSV if Parquet export is disabled on the server.
    Best for reproducible training datasets.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try native Parquet export
    response = requests.get(
        f"{QUESTDB_HTTP}/exp",
        params={"query": query, "fmt": "parquet"},
        stream=True
    )
    
    content_type = response.headers.get("content-type", "")
    
    if response.ok and "parquet" in content_type:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Exported native Parquet: {output_path}")
    else:
        # Fallback: CSV export + pandas conversion
        print("  Parquet export not available, using CSV fallback...")
        csv_response = requests.get(
            f"{QUESTDB_HTTP}/exp",
            params={"query": query, "fmt": "csv"}
        )
        csv_response.raise_for_status()
        df = pd.read_csv(StringIO(csv_response.text))
        df.to_parquet(output_path, index=False)
        print(f"✓ Exported via CSV→Parquet: {output_path}")
    
    return output_path


# =============================================================================
# Method 3: Direct Parquet Partition Access
# =============================================================================

def load_parquet_partitions(table_name: str, data_dir: Path):
    """
    Load QuestDB Parquet partitions directly via PyArrow.
    
    Requires:
    - Local/mounted access to QuestDB data directory
    - Parquet partition format enabled (o3PartitionFormat = 'PARQUET')
    
    Best for terabyte-scale data and streaming processing.
    """
    import pyarrow.dataset as ds
    
    table_dir = data_dir / table_name
    parquet_files = list(table_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {table_dir}. "
            "Ensure Parquet partition format is enabled in QuestDB."
        )
    
    print(f"✓ Found {len(parquet_files)} partition files")
    return ds.dataset(table_dir, format="parquet")


def stream_from_partitions(dataset, columns: list[str], batch_size: int = 100_000):
    """
    Stream data from Parquet partitions in batches.
    
    Memory-efficient for large datasets.
    """
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


# =============================================================================
# Query builders
# =============================================================================

def get_training_query(symbol: str = "BTC-USDT", days: int = 7) -> str:
    """Build query for training data: 1-minute aggregated bars."""
    return f"""
    SELECT 
        timestamp,
        symbol,
        sum(amount) as volume,
        count() as trade_count,
        avg(price) as avg_price,
        max(price) - min(price) as price_range
    FROM trades
    WHERE symbol = '{symbol}'
        AND timestamp > dateadd('d', -{days}, now())
    SAMPLE BY 1m
    ALIGN TO CALENDAR
    """
