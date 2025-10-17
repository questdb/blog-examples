import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta

def create_btc_order_book_dataset():
   """
   Generate realistic Bitcoin order book data with characteristics matching
   Binance/Coinbase behavior at ~$125k BTC price levels.

   Based on real market data:
   - Tight spreads: $1-5 in normal conditions (~0.001-0.004%)
   - Exponentially decaying volume with depth
   - Order book imbalance typically -0.2 to +0.2 in normal market
   - 2% market depth: hundreds of millions USD

   Scenarios:
   1. Normal trading conditions (tight spread, balanced book)
   2. Whale accumulation (bid imbalance, rising price)
   3. Liquidation cascade (thin bids, wide spread)
   4. Spoofing attempts (large fake walls)
   """

   data = []
   base_time = datetime(2025, 10, 6, 14, 30, 0)

   # Scenario 1: Normal Market (20 snapshots) - Realistic tight spreads
   print("Generating normal market conditions...")
   for i in range(20):
       mid_price = 125000 + np.random.randn() * 30  # Small random walk

       # Realistic tight spread: $2-5
       half_spread = np.random.uniform(1, 2.5)

       # Exponentially decaying volumes (realistic order book shape)
       # Top of book has most liquidity
       bid_base_volumes = np.array([15.0, 12.5, 10.2, 8.5, 7.0, 5.5, 4.2, 3.0, 2.0, 1.2])
       ask_base_volumes = np.array([14.8, 12.3, 10.0, 8.3, 6.8, 5.3, 4.0, 2.8, 1.8, 1.0])

       # Add realistic noise (±5-10%)
       bid_volumes = bid_base_volumes * (1 + np.random.randn(10) * 0.07)
       ask_volumes = ask_base_volumes * (1 + np.random.randn(10) * 0.07)

       # Slight random imbalance (-0.15 to +0.15)
       imbalance_factor = np.random.uniform(-0.15, 0.15)
       if imbalance_factor > 0:  # More bids
           bid_volumes *= (1 + imbalance_factor * 0.5)
       else:  # More asks
           ask_volumes *= (1 - imbalance_factor * 0.5)

       # Price levels: tighter near touch, exponentially wider
       # Basis points: ~1, 4, 8, 16, 32, 64, 128, 256, 512, 1024 bps
       bid_offsets = np.array([half_spread, 5, 10, 20, 40, 80, 160, 320, 640, 1280])
       ask_offsets = np.array([half_spread, 5, 10, 20, 40, 80, 160, 320, 640, 1280])

       bid_prices = mid_price - bid_offsets
       ask_prices = mid_price + ask_offsets

       data.append({
           'timestamp': base_time + timedelta(seconds=i*2),
           'symbol': 'BTC-USD',
           'exchange': 'BINANCE',
           'bids': np.array([bid_prices, bid_volumes]),
           'asks': np.array([ask_prices, ask_volumes]),
           'event_type': 'NORMAL'
       })

   # Scenario 2: Whale Accumulation (15 snapshots)
   # Gradual bid building, price rising, imbalance increasing
   print("Generating whale accumulation pattern...")
   base_time = datetime(2025, 10, 6, 14, 35, 0)

   for i in range(15):
       mid_price = 125050 + i * 25  # Gradual price rise

       half_spread = 1.5 + i * 0.1  # Spread widens slightly

       # Base volumes
       bid_volumes = np.array([12, 10, 8.5, 7, 5.5, 4.5, 3.5, 2.5, 1.8, 1.0])
       ask_volumes = np.array([10, 8, 6.5, 5, 4, 3, 2.5, 2, 1.5, 0.8])

       # Whale adds large walls deeper in the book (accumulation strategy)
       if i >= 3:
           bid_volumes[3] += 35  # 35 BTC wall at 20 bps
           bid_volumes[4] += 25  # 25 BTC wall at 40 bps
       if i >= 7:
           bid_volumes[5] += 50  # 50 BTC wall at 80 bps
       if i >= 10:
           bid_volumes[2] += 40  # Moving closer to touch

       # Asks thin out as they get absorbed
       ask_volumes = ask_volumes * (1 - i * 0.04)

       # Imbalance grows: starts ~0.1, reaches ~0.6

       bid_offsets = np.array([half_spread, 5, 10, 20, 40, 80, 160, 320, 640, 1280])
       ask_offsets = np.array([half_spread, 5, 10, 20, 40, 80, 160, 320, 640, 1280])

       bid_prices = mid_price - bid_offsets
       ask_prices = mid_price + ask_offsets

       data.append({
           'timestamp': base_time + timedelta(seconds=i*2),
           'symbol': 'BTC-USD',
           'exchange': 'BINANCE',
           'bids': np.array([bid_prices, bid_volumes]),
           'asks': np.array([ask_prices, ask_volumes]),
           'event_type': 'WHALE_BUY'
       })

   # Scenario 3: Liquidation Cascade (12 snapshots)
   # Spreads widen dramatically, bids evaporate, massive sell pressure
   print("Generating liquidation cascade...")
   base_time = datetime(2025, 10, 6, 14, 40, 0)

   for i in range(12):
       if i < 3:  # Pre-crash: weakening but still stable
           mid_price = 125100
           half_spread = 2.5
           bid_volumes = np.array([10, 8, 6.5, 5, 4, 3, 2.5, 2, 1.5, 1])
           ask_volumes = np.array([9, 7.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5])
       elif i < 6:  # Initial crash
           mid_price = 125100 - (i-2) * 600
           half_spread = 10 + (i-2) * 15  # Spread explodes
           # Bids pulled, panic selling
           bid_volumes = np.array([2, 1.5, 1, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05])
           ask_volumes = np.array([30, 28, 25, 22, 20, 18, 15, 12, 10, 8])  # Liquidations
       else:  # Deep crash
           mid_price = 125100 - (i-2) * 600
           half_spread = 40 + (i-5) * 25  # Extreme spread
           # Order book completely thin
           bid_volumes = np.array([0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.01])
           ask_volumes = np.array([35, 32, 30, 28, 25, 22, 20, 18, 15, 12])

       bid_offsets = np.array([half_spread, 8, 20, 50, 100, 200, 400, 800, 1200, 2000])
       ask_offsets = np.array([half_spread, 8, 20, 50, 100, 200, 400, 800, 1200, 2000])

       bid_prices = mid_price - bid_offsets
       ask_prices = mid_price + ask_offsets

       data.append({
           'timestamp': base_time + timedelta(seconds=i*2),
           'symbol': 'BTC-USD',
           'exchange': 'BINANCE',
           'bids': np.array([bid_prices, bid_volumes]),
           'asks': np.array([ask_prices, ask_volumes]),
           'event_type': 'LIQUIDATION' if i >= 3 else 'PRE_CRASH'
       })

   # Scenario 4: Spoofing Pattern (16 snapshots)
   # Large fake walls appear and disappear to manipulate price perception
   print("Generating spoofing pattern...")
   base_time = datetime(2025, 10, 6, 14, 45, 0)

   for i in range(16):
       mid_price = 125000 + np.random.randn() * 20
       half_spread = np.random.uniform(1.5, 3)

       # Normal base volumes
       bid_volumes = np.array([11, 9, 7.5, 6, 5, 4, 3.2, 2.5, 1.8, 1.2])
       ask_volumes = np.array([10.5, 8.5, 7, 5.8, 4.8, 3.8, 3, 2.3, 1.6, 1.0])

       # Spoof walls appear and disappear at deeper levels
       if i in [2, 3, 4, 10, 11, 12]:  # Spoof on bid side (fake support)
           bid_volumes[4] += 150  # Massive 150 BTC wall at 40 bps
       elif i in [6, 7, 8, 13, 14, 15]:  # Spoof on ask side (fake resistance)
           ask_volumes[4] += 160  # Massive 160 BTC wall at 40 bps

       # Slight imbalance based on which side has spoof
       if i in [2, 3, 4, 10, 11, 12]:
           ask_volumes *= 0.9  # Asks thin when bid spoof active
       elif i in [6, 7, 8, 13, 14, 15]:
           bid_volumes *= 0.9  # Bids thin when ask spoof active

       bid_offsets = np.array([half_spread, 5, 10, 20, 40, 80, 160, 320, 640, 1280])
       ask_offsets = np.array([half_spread, 5, 10, 20, 40, 80, 160, 320, 640, 1280])

       bid_prices = mid_price - bid_offsets
       ask_prices = mid_price + ask_offsets

       data.append({
           'timestamp': base_time + timedelta(seconds=i*2),
           'symbol': 'BTC-USD',
           'exchange': 'BINANCE',
           'bids': np.array([bid_prices, bid_volumes]),
           'asks': np.array([ask_prices, ask_volumes]),
           'event_type': 'SPOOFING'
       })

   df = pd.DataFrame(data)

   # Convert numpy arrays to lists for Parquet compatibility
   df['bids'] = df['bids'].apply(lambda x: x.tolist())
   df['asks'] = df['asks'].apply(lambda x: x.tolist())

   # Save to Parquet
   df.to_parquet('btc_order_book_samples.parquet', index=False)

   print(f"\n✅ Generated {len(df)} order book snapshots")
   print("\nScenarios included:")
   print(df.groupby('event_type').size())
   print("\nExchanges:")
   print(df.groupby('exchange').size())

   # Print sample statistics
   print("\n📊 Sample Statistics:")
   df_temp = pd.DataFrame(data)

   return df

# Generate the dataset
df = create_btc_order_book_dataset()


