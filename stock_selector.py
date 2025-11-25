"""
Stock Selection Module - Select high-quality liquid stocks for training
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StockSelector:
    """Select stocks based on quality criteria"""

    def __init__(self):
        pass

    def load_universe(self) -> pd.DataFrame:
        """Load stock universe from CSV"""
        try:
            # Try NSE_Universe first (has more metadata)
            df = pd.read_csv('NSE_Universe.csv')
            df['Ticker'] = df['Ticker'].apply(lambda x: x if str(x).endswith('.NS') else f"{x}.NS")
            return df
        except Exception as e:
            print(f"Error loading NSE_Universe.csv: {e}")

            # Try backup
            try:
                df = pd.read_csv('indian_stocks_tickers.csv', encoding='utf-8-sig')
                return df
            except Exception as e2:
                print(f"Error loading backup: {e2}")
                return pd.DataFrame()

    def get_stock_quality_metrics(self, ticker: str) -> dict:
        """
        Get basic quality metrics for a stock

        Returns dict with:
        - market_cap
        - avg_volume (3 month)
        - has_data (boolean)
        - data_months (number of months available)
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get historical data to check availability and volume
            hist = stock.history(period="3mo")

            if hist.empty:
                return {
                    'ticker': ticker,
                    'market_cap': 0,
                    'avg_volume': 0,
                    'has_data': False,
                    'data_months': 0
                }

            market_cap = info.get('marketCap', 0) or 0
            avg_volume = hist['Volume'].mean()

            # Check how many months of data available
            hist_5y = stock.history(period="5y")
            data_months = len(hist_5y) / 21 if not hist_5y.empty else 0  # Approx 21 trading days/month

            return {
                'ticker': ticker,
                'market_cap': market_cap,
                'avg_volume': avg_volume,
                'has_data': not hist.empty,
                'data_months': int(data_months)
            }

        except Exception as e:
            return {
                'ticker': ticker,
                'market_cap': 0,
                'avg_volume': 0,
                'has_data': False,
                'data_months': 0,
                'error': str(e)
            }

    def select_stocks(self,
                     n_stocks: int = 500,
                     min_market_cap: float = 1e9,  # 1 billion
                     min_avg_volume: float = 100000,  # 100k daily
                     min_data_months: int = 48,  # 4 years
                     use_cached: bool = True) -> List[str]:
        """
        Select stocks based on quality criteria

        Args:
            n_stocks: Target number of stocks
            min_market_cap: Minimum market cap (default 1 billion)
            min_avg_volume: Minimum average daily volume
            min_data_months: Minimum months of historical data
            use_cached: Use cached selection if available

        Returns:
            List of selected tickers
        """
        cache_file = 'selected_stocks_cache.csv'

        # Try to load cached selection
        if use_cached:
            try:
                cached = pd.read_csv(cache_file)
                print(f"Loaded {len(cached)} stocks from cache")
                return cached['ticker'].tolist()[:n_stocks]
            except:
                print("No cache found, will select stocks...")

        # Load universe
        universe = self.load_universe()

        if universe.empty:
            print("ERROR: Could not load stock universe!")
            return []

        print(f"\nTotal stocks in universe: {len(universe)}")

        # Get all tickers
        if 'Ticker' in universe.columns:
            all_tickers = universe['Ticker'].dropna().unique().tolist()
        else:
            print("ERROR: No Ticker column found!")
            return []

        # Add .NS suffix if not present
        all_tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in all_tickers]

        print(f"Processing {len(all_tickers)} tickers...")
        print("This will take several minutes...\n")

        # Get quality metrics for all stocks
        quality_data = []

        for i, ticker in enumerate(all_tickers):
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(all_tickers)} stocks...")

            metrics = self.get_stock_quality_metrics(ticker)
            quality_data.append(metrics)

        # Create DataFrame
        quality_df = pd.DataFrame(quality_data)

        # Save to cache
        quality_df.to_csv(cache_file, index=False)
        print(f"\nQuality data saved to {cache_file}")

        # Filter stocks
        print(f"\nApplying filters:")
        print(f"  - Market cap >= {min_market_cap/1e9:.1f}B")
        print(f"  - Avg volume >= {min_avg_volume:,.0f}")
        print(f"  - Data months >= {min_data_months}")

        filtered = quality_df[
            (quality_df['has_data'] == True) &
            (quality_df['market_cap'] >= min_market_cap) &
            (quality_df['avg_volume'] >= min_avg_volume) &
            (quality_df['data_months'] >= min_data_months)
        ].copy()

        print(f"\nStocks passing filters: {len(filtered)}")

        if len(filtered) == 0:
            print("WARNING: No stocks passed filters! Relaxing criteria...")
            # Relax filters
            filtered = quality_df[
                (quality_df['has_data'] == True) &
                (quality_df['data_months'] >= min_data_months)
            ].copy()
            print(f"With relaxed filters: {len(filtered)}")

        # Sort by market cap (largest first)
        filtered = filtered.sort_values('market_cap', ascending=False)

        # Select top N
        selected = filtered.head(n_stocks)

        print(f"\nSelected {len(selected)} stocks")
        print(f"Market cap range: {selected['market_cap'].min()/1e9:.1f}B to {selected['market_cap'].max()/1e9:.1f}B")
        print(f"Avg volume range: {selected['avg_volume'].min():,.0f} to {selected['avg_volume'].max():,.0f}")

        return selected['ticker'].tolist()

    def get_stratified_sample(self,
                             n_stocks: int = 500,
                             by_market_cap: bool = True,
                             by_sector: bool = False) -> List[str]:
        """
        Get stratified sample of stocks

        Args:
            n_stocks: Target number
            by_market_cap: Stratify by market cap (large/mid/small)
            by_sector: Stratify by sector

        Returns:
            List of tickers
        """
        # First get quality-filtered stocks
        all_stocks = self.select_stocks(
            n_stocks=2000,  # Get large pool first
            use_cached=True
        )

        if not all_stocks:
            return []

        # Load universe with metadata
        universe = self.load_universe()

        if by_market_cap:
            # Get market caps
            print("\nGetting market caps for stratification...")
            caps = []
            for ticker in all_stocks[:500]:  # Limit for speed
                try:
                    stock = yf.Ticker(ticker)
                    cap = stock.info.get('marketCap', 0)
                    caps.append({'ticker': ticker, 'market_cap': cap})
                except:
                    pass

            caps_df = pd.DataFrame(caps)

            # Divide into terciles (large/mid/small cap)
            caps_df['cap_category'] = pd.qcut(
                caps_df['market_cap'],
                q=3,
                labels=['Small', 'Mid', 'Large']
            )

            # Sample proportionally from each
            n_per_category = n_stocks // 3

            selected = []
            for category in ['Large', 'Mid', 'Small']:
                category_stocks = caps_df[caps_df['cap_category'] == category]['ticker'].tolist()
                selected.extend(category_stocks[:n_per_category])

            print(f"\nStratified sample: {len(selected)} stocks")
            print(f"  - Large cap: {n_per_category}")
            print(f"  - Mid cap: {n_per_category}")
            print(f"  - Small cap: {n_per_category}")

            return selected

        else:
            # Just return top N by market cap
            return all_stocks[:n_stocks]


def quick_select_top_stocks(n: int = 500) -> List[str]:
    """
    Quick selection: Top N stocks by market cap with quality filters

    This is the recommended approach for most users.
    """
    selector = StockSelector()
    return selector.select_stocks(
        n_stocks=n,
        min_market_cap=1e9,      # 1 billion minimum
        min_avg_volume=100000,   # 100k daily volume
        min_data_months=48,      # 4 years of data
        use_cached=True
    )


if __name__ == "__main__":
    # Test the selector
    print("="*80)
    print("STOCK SELECTION TEST")
    print("="*80)

    selector = StockSelector()

    # Test with different sample sizes
    for n in [50, 200, 500]:
        print(f"\n{'='*80}")
        print(f"Selecting {n} stocks")
        print(f"{'='*80}")

        stocks = selector.select_stocks(
            n_stocks=n,
            min_market_cap=1e9,
            min_avg_volume=100000,
            min_data_months=48,
            use_cached=True
        )

        if stocks:
            print(f"\nSelected {len(stocks)} stocks")
            print(f"First 10: {stocks[:10]}")
            print(f"Last 10: {stocks[-10:]}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
For training:
  - 200 stocks: MINIMUM (quick test)
  - 500 stocks: RECOMMENDED (good balance)
  - 1000 stocks: EXCELLENT (very robust)

Always use quality filters:
  - Market cap >= 1B (liquid, established companies)
  - Volume >= 100k (tradeable)
  - Data >= 4 years (sufficient history)
    """)
