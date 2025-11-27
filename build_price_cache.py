"""
Build Price Cache

Fetches price data for all stocks in screener database and caches it.
This cache is used by model training to avoid fetching prices every time.

Run monthly to update cache.
"""

import pandas as pd
import pickle
import sqlite3
from price_data_helper import PriceDataHelper
from typing import Dict
import time
from datetime import datetime


def build_price_cache(db_path: str = 'screener_data.db',
                      cache_path: str = 'price_cache.pkl') -> Dict:
    """
    Build cache of price data for all stocks in database

    Args:
        db_path: Path to screener database
        cache_path: Path to save cache file

    Returns:
        Dictionary mapping ticker -> price DataFrame
    """
    print("=" * 80)
    print("BUILD PRICE DATA CACHE")
    print("=" * 80)

    # Get all tickers from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM quarterly_results ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"\nTotal stocks in database: {len(tickers)}")
    print(f"Fetching price data for all stocks...")
    print(f"This may take {len(tickers) * 0.15 / 60:.1f} minutes (with 0.15s delay per stock)\n")

    # Fetch prices for all stocks
    price_helper = PriceDataHelper()
    price_cache = {}

    success_count = 0
    fail_count = 0
    verbose_limit = 10  # Show details for first 10 failures

    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(tickers) - i) / rate
            print(f"  [{i}/{len(tickers)}] Success: {success_count}, Failed: {fail_count} "
                  f"(ETA: {remaining/60:.1f} min)")

        # Fetch price data (with verbose for first few failures)
        verbose = (fail_count < verbose_limit)
        prices = price_helper._fetch_price_data(ticker, verbose=verbose)

        if prices is not None and not prices.empty:
            price_cache[ticker] = prices
            success_count += 1
        else:
            fail_count += 1
            if verbose:
                print(f"    [FAIL] {ticker}: No price data available")

        # Delay to avoid rate limiting
        time.sleep(0.15)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("PRICE CACHE BUILD COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nTotal stocks: {len(tickers)}")
    print(f"Successful: {success_count} ({success_count/len(tickers)*100:.1f}%)")
    print(f"Failed: {fail_count} ({fail_count/len(tickers)*100:.1f}%)")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Average rate: {len(tickers)/elapsed:.1f} stocks/sec")

    # Calculate cache statistics
    total_data_points = sum(len(df) for df in price_cache.values())
    print(f"\nCache statistics:")
    print(f"  Stocks with data: {len(price_cache)}")
    print(f"  Total price data points: {total_data_points:,}")
    print(f"  Average days per stock: {total_data_points/len(price_cache):.0f}")

    # Save cache
    cache_data = {
        'prices': price_cache,
        'build_date': datetime.now().isoformat(),
        'success_rate': success_count / len(tickers),
        'total_stocks': len(tickers),
        'successful_stocks': success_count,
        'failed_stocks': fail_count
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    # Get file size
    import os
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)

    print(f"\n[OK] Cache saved to: {cache_path}")
    print(f"      File size: {file_size_mb:.2f} MB")
    print(f"      Build date: {cache_data['build_date']}")

    return cache_data


def load_price_cache(cache_path: str = 'price_cache.pkl') -> Dict:
    """
    Load price cache from file

    Args:
        cache_path: Path to cache file

    Returns:
        Cache dictionary with 'prices', 'build_date', etc.
    """
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        print(f"[OK] Loaded price cache from: {cache_path}")
        print(f"     Build date: {cache_data['build_date']}")
        print(f"     Stocks cached: {cache_data['successful_stocks']}/{cache_data['total_stocks']}")
        print(f"     Success rate: {cache_data['success_rate']*100:.1f}%")

        return cache_data

    except FileNotFoundError:
        print(f"[ERROR] Cache file not found: {cache_path}")
        return None


if __name__ == "__main__":
    cache = build_price_cache(
        db_path='screener_data.db',
        cache_path='price_cache.pkl'
    )

    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print(f"{'=' * 80}")
    print("\n1. Commit the cache file:")
    print("   git add price_cache.pkl")
    print("   git commit -m 'Add: Price data cache for model training'")
    print("   git push")
    print("\n2. Update model_trainer_screener.py to use cached prices")
    print("\n3. Training will now use cached data instead of fetching from yfinance")
