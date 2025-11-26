"""
Automated Training Data Collection Script

Collects 5-year historical data for 195 quality stocks.
Runs via GitHub Actions monthly to refresh training data.
"""

import os
import sys
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collector_enhanced import collect_data_for_universe
from nifty500_tickers import get_quality_stocks


def collect_training_data():
    """
    Collect training data for model

    Uses:
    - 195 quality stocks (Nifty 50/100/200)
    - 5 years historical data
    - Price + fundamental data
    """

    print("="*80)
    print("TRAINING DATA COLLECTION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Get quality stocks
    print("\n[1/2] Loading quality stocks list...")
    tickers = get_quality_stocks(n=195)  # Get all 195 quality stocks
    print(f"✅ Selected {len(tickers)} quality stocks (Nifty 50/100/200)")

    # Step 2: Collect data
    print("\n[2/2] Collecting 5-year historical data...")
    print("⏱️  This will take 15-20 minutes...")

    data = collect_data_for_universe(tickers, lookback_years=5)

    if data is None or len(data) == 0:
        print("❌ ERROR: Failed to collect training data!")
        sys.exit(1)

    # Save training data
    cache_file = 'stock_data_cache.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    print(f"Total stocks: {data['Ticker'].nunique()}")
    print(f"Total rows: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Saved to: {cache_file}")
    print("="*80)

    return True


if __name__ == "__main__":
    success = collect_training_data()
    sys.exit(0 if success else 1)
