"""
Simple Monthly Data Collector

BACK TO BASICS:
- Previous month fundamentals (RATIOS ONLY - no absolute values)
- Previous month technicals
- Current month return

Feature philosophy:
✓ Growth rates (sales, EPS, profit) - SIZE AGNOSTIC
✓ Margins (operating, net, ROE, ROCE) - SIZE AGNOSTIC
✓ Efficiency ratios (asset turnover, working capital) - SIZE AGNOSTIC
✓ Risk ratios (debt/equity, interest coverage) - SIZE AGNOSTIC
✓ Quality scores (Piotroski-like) - SIZE AGNOSTIC
✗ Absolute values (sales, profit, interest) - SIZE DEPENDENT = BAD!
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import List
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from fundamental_ratios import FundamentalRatios


class SimpleMonthlyCollector:
    """Collect monthly data with SIZE-AGNOSTIC fundamental ratios"""

    def __init__(self, db_path: str = 'screener_data.db'):
        self.db_path = db_path
        self.ratio_calculator = FundamentalRatios(db_path)

    def get_available_stocks(self) -> List[str]:
        """Get list of stocks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM quarterly_results")
        stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
        return stocks

    def get_quarterly_for_month(self, ticker: str, month_end: pd.Timestamp) -> int:
        """
        Find which quarterly index to use for this month

        Rule: Use most recent quarter that was REPORTED by month_end
        (Quarter + 45 days for reporting)
        """
        quarterly = self.ratio_calculator.get_quarterly_data(ticker)

        if quarterly.empty:
            return -1

        # Find quarters reported before month_end
        reported_dates = quarterly['quarter_date'] + timedelta(days=45)
        available = quarterly[reported_dates <= month_end]

        if available.empty:
            return -1

        # Return index of most recent quarter
        return len(quarterly[quarterly['quarter_date'] <= available.iloc[-1]['quarter_date']]) - 1

    def prepare_monthly_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Prepare monthly data with SIZE-AGNOSTIC ratios

        For each month:
        - Find latest REPORTED quarter
        - Calculate fundamental RATIOS (no absolute values!)
        - Month-end date for technicals
        """
        quarterly = self.ratio_calculator.get_quarterly_data(ticker)

        if quarterly.empty:
            return pd.DataFrame()

        # Generate month-end dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        month_ends = pd.date_range(start=start, end=end, freq='ME')  # Use 'ME' instead of deprecated 'M'

        monthly_data = []

        for month_end in month_ends:
            # Get latest reported quarter index
            quarter_idx = self.get_quarterly_for_month(ticker, month_end)

            if quarter_idx == -1:
                continue

            # Calculate ALL ratios for this quarter (SIZE-AGNOSTIC!)
            ratios = self.ratio_calculator.calculate_all_ratios(ticker, quarter_idx)

            if not ratios:
                continue

            quarter_date = quarterly.iloc[quarter_idx]['quarter_date']
            staleness = (month_end - quarter_date).days

            # Combine into row
            row = {
                'ticker': ticker,
                'month_end_date': month_end,
                'quarter_date': quarter_date,
                'staleness_days': staleness,
                **ratios  # All SIZE-AGNOSTIC ratios
            }

            monthly_data.append(row)

        if not monthly_data:
            return pd.DataFrame()

        return pd.DataFrame(monthly_data)

    def prepare_training_data(self, tickers: List[str],
                             start_date: str = '2014-12-31',
                             end_date: str = '2024-11-30') -> pd.DataFrame:
        """
        Prepare monthly training data for multiple stocks

        Returns DataFrame with SIZE-AGNOSTIC ratios only!
        """
        print(f"\n{'='*80}")
        print("PREPARING MONTHLY DATA - SIZE-AGNOSTIC RATIOS ONLY")
        print(f"{'='*80}")
        print(f"\nDate range: {start_date} to {end_date}")
        print(f"Stocks to process: {len(tickers)}")

        all_data = []

        for i, ticker in enumerate(tickers, 1):
            if i % 50 == 0:
                print(f"Processing {i}/{len(tickers)}...")

            try:
                monthly = self.prepare_monthly_data(ticker, start_date, end_date)

                if not monthly.empty:
                    all_data.append(monthly)

            except Exception as e:
                print(f"[WARNING] {ticker}: {str(e)}")
                continue

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        print(f"\n[OK] Monthly data collection complete")
        print(f"    Total months: {len(combined)}")
        print(f"    Unique stocks: {combined['ticker'].nunique()}")
        print(f"    Date range: {combined['month_end_date'].min().date()} to {combined['month_end_date'].max().date()}")

        # Show feature breakdown
        feature_cols = [col for col in combined.columns
                       if col not in ['ticker', 'month_end_date', 'quarter_date', 'staleness_days']]

        print(f"\nFundamental ratios (SIZE-AGNOSTIC):")
        print(f"    Total ratios: {len(feature_cols)}")

        # Categorize features
        growth_features = [f for f in feature_cols if 'growth' in f or 'acceleration' in f]
        profit_features = [f for f in feature_cols if 'margin' in f or 'roe' in f or 'roce' in f or 'opm' in f]
        risk_features = [f for f in feature_cols if 'debt' in f or 'interest' in f or 'coverage' in f]
        quality_features = [f for f in feature_cols if 'quality' in f or 'is_' in f or 'improving' in f or 'positive' in f or 'low_' in f]

        print(f"    Growth metrics: {len(growth_features)} - {growth_features[:3]}...")
        print(f"    Profitability: {len(profit_features)} - {profit_features[:3]}...")
        print(f"    Risk metrics: {len(risk_features)} - {risk_features}")
        print(f"    Quality scores: {len(quality_features)} - {quality_features[:3]}...")

        return combined
