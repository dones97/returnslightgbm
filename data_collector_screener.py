"""
Data Collector for Screener.in Database

Extracts quarterly fundamental data from screener.in SQLite database
and prepares features for predicting next quarter returns.

Key improvements over yfinance:
- 10+ quarters of clean fundamental data
- Pre-calculated ratios (ROCE, cash conversion cycle, etc.)
- Consistent data quality across all stocks
- Rich balance sheet and cash flow metrics
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ScreenerDataCollector:
    """Collect and prepare data from screener.in database"""

    def __init__(self, db_path: str = 'screener_data.db'):
        """
        Initialize data collector

        Args:
            db_path: Path to screener.in SQLite database
        """
        self.db_path = db_path

    def get_available_stocks(self) -> List[str]:
        """Get list of all stocks with data in database"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT ticker FROM companies WHERE data_available = 1"
        stocks = pd.read_sql_query(query, conn)
        conn.close()
        return stocks['ticker'].tolist()

    def get_quarterly_data(self, ticker: str) -> pd.DataFrame:
        """
        Get all quarterly data for a stock

        Returns DataFrame with quarters as rows and all metrics as columns
        """
        conn = sqlite3.connect(self.db_path)

        # Quarterly P&L
        quarterly = pd.read_sql_query(
            "SELECT * FROM quarterly_results WHERE ticker = ? ORDER BY quarter_date DESC",
            conn, params=(ticker,)
        )

        conn.close()

        if quarterly.empty:
            return pd.DataFrame()

        # Convert quarter_date to datetime
        quarterly['quarter_date'] = pd.to_datetime(quarterly['quarter_date'], errors='coerce')
        quarterly = quarterly.sort_values('quarter_date').reset_index(drop=True)

        return quarterly

    def get_annual_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get all annual data for a stock"""
        conn = sqlite3.connect(self.db_path)

        data = {
            'profit_loss': pd.read_sql_query(
                "SELECT * FROM annual_profit_loss WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
            'balance_sheet': pd.read_sql_query(
                "SELECT * FROM balance_sheet WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
            'cash_flow': pd.read_sql_query(
                "SELECT * FROM cash_flow WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
            'ratios': pd.read_sql_query(
                "SELECT * FROM annual_ratios WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            )
        }

        conn.close()
        return data

    def create_features_for_stock(self, ticker: str) -> pd.DataFrame:
        """
        Create feature set for one stock

        Features include:
        - Quarterly metrics (sales, profit, margins)
        - Growth rates (QoQ, YoY)
        - Quality metrics (ROE, ROCE, ratios)
        - Trend indicators

        Returns DataFrame with one row per quarter
        """
        quarterly = self.get_quarterly_data(ticker)

        if quarterly.empty or len(quarterly) < 3:
            return pd.DataFrame()

        # Start with quarterly data
        df = quarterly.copy()
        df['ticker'] = ticker

        # Calculate growth rates (Quarter-over-Quarter)
        df['sales_growth_qoq'] = df['sales'].pct_change() * 100
        df['profit_growth_qoq'] = df['net_profit'].pct_change() * 100
        df['eps_growth_qoq'] = df['eps'].pct_change() * 100

        # Calculate growth rates (Year-over-Year - compare to same quarter last year)
        df['sales_growth_yoy'] = df['sales'].pct_change(periods=4) * 100
        df['profit_growth_yoy'] = df['net_profit'].pct_change(periods=4) * 100
        df['eps_growth_yoy'] = df['eps'].pct_change(periods=4) * 100

        # Profitability ratios (already have OPM%, calculate more)
        df['profit_margin'] = (df['net_profit'] / df['sales']) * 100
        df['ebit_margin'] = ((df['operating_profit'] - df['depreciation']) / df['sales']) * 100

        # Efficiency metrics
        df['expense_ratio'] = (df['expenses'] / df['sales']) * 100
        df['interest_coverage'] = df['operating_profit'] / (df['interest'] + 1e-6)
        df['tax_rate'] = df['tax_percent']

        # Momentum indicators (moving averages of growth)
        df['sales_growth_ma3'] = df['sales_growth_qoq'].rolling(3).mean()
        df['profit_growth_ma3'] = df['profit_growth_qoq'].rolling(3).mean()

        # Technical/Momentum features derived from fundamentals
        # Acceleration (rate of change of growth)
        df['sales_acceleration'] = df['sales_growth_qoq'].diff()
        df['profit_acceleration'] = df['profit_growth_qoq'].diff()

        # Momentum strength (consecutive positive/negative growth)
        df['sales_momentum_streak'] = (df['sales_growth_qoq'] > 0).astype(int)
        df['profit_momentum_streak'] = (df['profit_growth_qoq'] > 0).astype(int)

        # Relative strength (current vs historical average)
        df['sales_relative_strength'] = df['sales'] / df['sales'].rolling(4).mean()
        df['profit_relative_strength'] = df['net_profit'] / (df['net_profit'].rolling(4).mean() + 1e-6)

        # Volatility metrics
        df['sales_volatility'] = df['sales'].rolling(4).std() / (df['sales'].rolling(4).mean() + 1e-6)
        df['profit_volatility'] = df['net_profit'].rolling(4).std() / (df['net_profit'].rolling(4).mean() + 1e-6)

        # Quality score (higher is better)
        # Positive profit, positive sales growth, improving margins
        df['quality_score'] = 0
        df.loc[df['net_profit'] > 0, 'quality_score'] += 1
        df.loc[df['sales_growth_yoy'] > 0, 'quality_score'] += 1
        df.loc[df['opm_percent'] > df['opm_percent'].shift(1), 'quality_score'] += 1
        df.loc[df['profit_margin'] > df['profit_margin'].shift(1), 'quality_score'] += 1

        # Trend direction (1 = improving, -1 = deteriorating, 0 = stable)
        df['sales_trend'] = np.sign(df['sales_growth_ma3'])
        df['profit_trend'] = np.sign(df['profit_growth_ma3'])

        return df

    def get_price_returns(self, ticker: str, quarterly_dates: pd.Series) -> pd.Series:
        """
        Calculate quarterly returns for a stock

        For now, we'll use a placeholder since we don't have price data in screener.in
        You'll need to fetch this from yfinance or another source

        Returns: Series with next quarter returns (our target variable)
        """
        # TODO: Integrate with yfinance for price data
        # For now, return empty series - you'll need to add price data source
        return pd.Series(index=quarterly_dates.index, dtype=float)

    def prepare_training_data(self, tickers: List[str],
                            lookback_quarters: int = 10) -> pd.DataFrame:
        """
        Prepare training dataset from multiple stocks

        Args:
            tickers: List of stock tickers
            lookback_quarters: Number of quarters to use (default: 10 = 2.5 years)

        Returns:
            DataFrame with features for all stocks
        """
        all_data = []

        print(f"\n{'='*80}")
        print(f"PREPARING TRAINING DATA")
        print(f"{'='*80}")
        print(f"Stocks to process: {len(tickers)}")
        print(f"Lookback: {lookback_quarters} quarters")

        for i, ticker in enumerate(tickers, 1):
            if i % 50 == 0:
                print(f"Processing {i}/{len(tickers)}...")

            try:
                # Get features
                features = self.create_features_for_stock(ticker)

                if features.empty:
                    continue

                # Keep only recent quarters
                features = features.tail(lookback_quarters)

                if len(features) < 4:  # Need at least 4 quarters
                    continue

                all_data.append(features)

            except Exception as e:
                print(f"  [WARNING] Error processing {ticker}: {str(e)}")
                continue

        if not all_data:
            print("[ERROR] No data collected!")
            return pd.DataFrame()

        # Combine all stocks
        combined = pd.concat(all_data, ignore_index=True)

        print(f"\n[OK] Data collection complete")
        print(f"    Total quarters: {len(combined)}")
        print(f"    Unique stocks: {combined['ticker'].nunique()}")
        print(f"    Features: {len(combined.columns)}")
        print(f"    DataFrame shape: {combined.shape}")
        print(f"    DataFrame empty: {combined.empty}")

        return combined

    def get_current_data_for_screening(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get most recent quarter data for all stocks (for screening)

        Args:
            tickers: List of tickers to screen

        Returns:
            DataFrame with latest quarter features for each stock
        """
        screening_data = []

        print(f"\n{'='*80}")
        print(f"PREPARING SCREENING DATA")
        print(f"{'='*80}")
        print(f"Stocks to screen: {len(tickers)}")

        for i, ticker in enumerate(tickers, 1):
            if i % 100 == 0:
                print(f"Processing {i}/{len(tickers)}...")

            try:
                features = self.create_features_for_stock(ticker)

                if features.empty:
                    continue

                # Get most recent quarter
                latest = features.iloc[-1:].copy()
                screening_data.append(latest)

            except Exception as e:
                continue

        if not screening_data:
            return pd.DataFrame()

        result = pd.concat(screening_data, ignore_index=True)

        print(f"\n[OK] Screening data ready")
        print(f"    Stocks with data: {len(result)}")
        print(f"    Features: {len(result.columns)}")

        return result


def test_data_collector():
    """Test the data collector"""
    collector = ScreenerDataCollector()

    # Get available stocks
    stocks = collector.get_available_stocks()
    print(f"\nAvailable stocks in database: {len(stocks)}")

    if not stocks:
        print("[ERROR] No stocks found in database!")
        return

    # Test with first stock
    test_ticker = stocks[0]
    print(f"\nTesting with: {test_ticker}")

    # Get quarterly data
    quarterly = collector.get_quarterly_data(test_ticker)
    print(f"\nQuarterly data: {len(quarterly)} quarters")
    if not quarterly.empty:
        print(f"Date range: {quarterly['quarter_date'].min()} to {quarterly['quarter_date'].max()}")
        print(f"Columns: {list(quarterly.columns)}")

    # Create features
    features = collector.create_features_for_stock(test_ticker)
    print(f"\nFeatures created: {len(features)} rows x {len(features.columns)} columns")
    if not features.empty:
        print(f"Feature columns: {list(features.columns)}")
        print(f"\nSample data:")
        print(features[['quarter_date', 'sales', 'net_profit', 'sales_growth_yoy',
                       'profit_margin', 'quality_score']].tail())


if __name__ == "__main__":
    test_data_collector()
