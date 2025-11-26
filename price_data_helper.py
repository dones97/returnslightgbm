"""
Price Data Helper

Fetches price data from yfinance to calculate returns.
Screener.in provides fundamentals, yfinance provides prices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class PriceDataHelper:
    """Helper to fetch price data and calculate returns"""

    def __init__(self):
        self.cache = {}  # Cache price data to avoid repeated API calls

    def get_quarterly_returns(self, ticker: str, quarter_dates: pd.Series,
                             forward_periods: int = 1) -> pd.Series:
        """
        Calculate returns from each quarter date to next quarter

        Args:
            ticker: Stock ticker (without .NS suffix)
            quarter_dates: Series of quarter end dates
            forward_periods: Number of quarters to look forward (default: 1)

        Returns:
            Series of returns (%) for each quarter
        """
        if ticker in self.cache:
            prices = self.cache[ticker]
        else:
            prices = self._fetch_price_data(ticker)
            if prices is None or prices.empty:
                return pd.Series([np.nan] * len(quarter_dates), index=quarter_dates.index)
            self.cache[ticker] = prices

        returns = []

        for quarter_date in quarter_dates:
            try:
                # Get price at quarter end
                start_price = self._get_price_near_date(prices, quarter_date)

                if start_price is None:
                    returns.append(np.nan)
                    continue

                # Get price ~3 months later (next quarter)
                future_date = quarter_date + timedelta(days=90 * forward_periods)
                end_price = self._get_price_near_date(prices, future_date)

                if end_price is None:
                    returns.append(np.nan)
                    continue

                # Calculate return
                ret = ((end_price - start_price) / start_price) * 100
                returns.append(ret)

            except Exception as e:
                returns.append(np.nan)

        return pd.Series(returns, index=quarter_dates.index)

    def _fetch_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data from yfinance

        Args:
            ticker: Stock ticker (without .NS suffix)

        Returns:
            DataFrame with Date index and Close prices
        """
        try:
            import time

            # Add .NS suffix for NSE stocks
            yf_ticker = f"{ticker}.NS"

            # Fetch 3 years of data (enough for quarters + forward returns)
            # Disable progress bar and logs to avoid spam
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)

            stock = yf.Ticker(yf_ticker)

            # Add small delay to avoid rate limiting
            time.sleep(0.1)

            hist = stock.history(period="3y", timeout=15, raise_errors=False)

            if hist is None or hist.empty:
                return None

            # Keep only Close prices
            prices = hist[['Close']].copy()

            # Remove timezone if present
            if hasattr(prices.index, 'tz') and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            else:
                prices.index = pd.to_datetime(prices.index)

            return prices

        except Exception as e:
            # Silently fail and return None - this is expected for some stocks
            return None

    def _get_price_near_date(self, prices: pd.DataFrame, target_date: pd.Timestamp,
                            tolerance_days: int = 10) -> Optional[float]:
        """
        Get price closest to target date (within tolerance)

        Args:
            prices: DataFrame with DatetimeIndex and Close column
            target_date: Target date
            tolerance_days: Maximum days to search before/after

        Returns:
            Price (float) or None
        """
        try:
            # Ensure target_date is timezone-naive
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.tz_localize(None)

            # Find nearest date within tolerance
            start = target_date - timedelta(days=tolerance_days)
            end = target_date + timedelta(days=tolerance_days)

            nearby = prices[(prices.index >= start) & (prices.index <= end)]

            if nearby.empty:
                return None

            # Get price closest to target
            idx = (nearby.index - target_date).abs().argmin()
            return nearby.iloc[idx]['Close']

        except Exception as e:
            return None

    def clear_cache(self):
        """Clear price data cache"""
        self.cache = {}


def test_price_helper():
    """Test price data helper"""
    helper = PriceDataHelper()

    # Test with a known stock
    test_ticker = "RELIANCE"

    print(f"\nTesting price data for {test_ticker}...")

    # Create sample quarter dates
    quarter_dates = pd.Series([
        pd.Timestamp('2023-03-31'),
        pd.Timestamp('2023-06-30'),
        pd.Timestamp('2023-09-30'),
        pd.Timestamp('2023-12-31'),
        pd.Timestamp('2024-03-31')
    ])

    # Get returns
    returns = helper.get_quarterly_returns(test_ticker, quarter_dates)

    print(f"\nQuarter-over-quarter returns:")
    for date, ret in zip(quarter_dates, returns):
        if pd.notna(ret):
            print(f"  {date.strftime('%Y-%m-%d')}: {ret:+.2f}%")
        else:
            print(f"  {date.strftime('%Y-%m-%d')}: N/A")


if __name__ == "__main__":
    test_price_helper()
