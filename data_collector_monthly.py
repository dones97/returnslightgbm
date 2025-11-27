"""
Monthly Data Collector with Proper Fundamental Alignment

Key principle: Only use fundamentals that were PUBLICLY AVAILABLE at prediction time.
No look-ahead bias!

For each month-end:
1. Find most recently REPORTED quarterly fundamentals
2. Calculate technical indicators as of month-end
3. Calculate 1-month forward return

Assumption: Companies report earnings within 45 days of quarter end
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import List, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class MonthlyDataCollector:
    """Collect monthly training data with proper fundamental lag"""

    def __init__(self, db_path: str = 'screener_data.db'):
        self.db_path = db_path

    def get_available_stocks(self) -> List[str]:
        """Get list of all stocks in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM quarterly_results ORDER BY ticker")
        stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
        return stocks

    def get_quarterly_data(self, ticker: str) -> pd.DataFrame:
        """
        Get quarterly fundamental data for a stock

        Returns DataFrame with quarter_date and all fundamentals
        """
        conn = sqlite3.connect(self.db_path)

        quarterly = pd.read_sql_query(
            "SELECT * FROM quarterly_results WHERE ticker = ? ORDER BY quarter_date",
            conn, params=(ticker,)
        )

        conn.close()

        if quarterly.empty:
            return pd.DataFrame()

        # Convert dates and numeric columns
        quarterly['quarter_date'] = pd.to_datetime(quarterly['quarter_date'], errors='coerce')

        # Convert numeric columns
        numeric_columns = [
            'sales', 'expenses', 'operating_profit', 'opm_percent', 'other_income',
            'interest', 'depreciation', 'profit_before_tax', 'tax_percent',
            'net_profit', 'eps', 'equity'
        ]

        for col in numeric_columns:
            if col in quarterly.columns:
                quarterly[col] = pd.to_numeric(quarterly[col], errors='coerce')

        # Sort by date
        quarterly = quarterly.sort_values('quarter_date').reset_index(drop=True)

        return quarterly

    def get_reported_date(self, quarter_end_date: pd.Timestamp) -> pd.Timestamp:
        """
        Estimate when quarterly results were reported

        Rule: Companies report within 45 days of quarter end
        Conservative estimate: Assume reported 45 days after quarter end
        """
        return quarter_end_date + timedelta(days=45)

    def get_latest_reported_quarter(self, current_date: pd.Timestamp,
                                   quarterly_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get the most recent quarterly data that was REPORTED by current_date

        Args:
            current_date: The date at which we're making prediction
            quarterly_data: DataFrame of all quarterly results

        Returns:
            Single row DataFrame of most recent reported quarter, or empty if none available
        """
        if quarterly_data.empty:
            return pd.DataFrame()

        # For each quarter, calculate its reporting date
        reported_dates = quarterly_data['quarter_date'].apply(self.get_reported_date)

        # Find quarters that were reported BEFORE current_date
        available_quarters = quarterly_data[reported_dates <= current_date]

        if available_quarters.empty:
            return pd.DataFrame()

        # Return the most recent one
        return available_quarters.iloc[[-1]]  # Return as DataFrame, not Series

    def create_fundamental_features(self, quarter_data: pd.DataFrame) -> dict:
        """
        Create features from a single quarter's fundamental data

        Args:
            quarter_data: Single row DataFrame with quarterly fundamentals

        Returns:
            Dictionary of fundamental features
        """
        if quarter_data.empty:
            return {}

        row = quarter_data.iloc[0]

        features = {
            # Raw metrics
            'sales': row.get('sales', np.nan),
            'expenses': row.get('expenses', np.nan),
            'operating_profit': row.get('operating_profit', np.nan),
            'net_profit': row.get('net_profit', np.nan),
            'eps': row.get('eps', np.nan),
            'opm_percent': row.get('opm_percent', np.nan),
            'other_income': row.get('other_income', np.nan),
            'interest': row.get('interest', np.nan),
            'depreciation': row.get('depreciation', np.nan),
            'profit_before_tax': row.get('profit_before_tax', np.nan),
            'tax_percent': row.get('tax_percent', np.nan),
            'equity': row.get('equity', np.nan),
        }

        return features

    def create_growth_features(self, ticker: str, current_quarter: pd.DataFrame,
                              quarterly_data: pd.DataFrame) -> dict:
        """
        Create growth features by comparing to historical quarters

        Args:
            ticker: Stock ticker
            current_quarter: Single row of current quarter
            quarterly_data: Full quarterly history

        Returns:
            Dictionary of growth features
        """
        if current_quarter.empty or quarterly_data.empty:
            return {}

        current_date = current_quarter['quarter_date'].iloc[0]
        current_idx = quarterly_data[quarterly_data['quarter_date'] == current_date].index

        if len(current_idx) == 0:
            return {}

        idx = current_idx[0]

        features = {}

        # QoQ growth (vs previous quarter)
        if idx > 0:
            prev_quarter = quarterly_data.iloc[idx - 1]

            for metric in ['sales', 'net_profit', 'eps']:
                current_val = current_quarter[metric].iloc[0]
                prev_val = prev_quarter.get(metric, np.nan)

                if pd.notna(current_val) and pd.notna(prev_val) and prev_val != 0:
                    features[f'{metric}_growth_qoq'] = ((current_val - prev_val) / abs(prev_val)) * 100
                else:
                    features[f'{metric}_growth_qoq'] = np.nan

        # YoY growth (vs same quarter last year)
        if idx >= 4:
            yoy_quarter = quarterly_data.iloc[idx - 4]

            for metric in ['sales', 'net_profit', 'eps']:
                current_val = current_quarter[metric].iloc[0]
                yoy_val = yoy_quarter.get(metric, np.nan)

                if pd.notna(current_val) and pd.notna(yoy_val) and yoy_val != 0:
                    features[f'{metric}_growth_yoy'] = ((current_val - yoy_val) / abs(yoy_val)) * 100
                else:
                    features[f'{metric}_growth_yoy'] = np.nan

        # Margins
        sales = current_quarter['sales'].iloc[0]
        net_profit = current_quarter['net_profit'].iloc[0]
        expenses = current_quarter['expenses'].iloc[0]
        operating_profit = current_quarter['operating_profit'].iloc[0]
        depreciation = current_quarter.get('depreciation', pd.Series([np.nan])).iloc[0]

        if pd.notna(sales) and sales != 0:
            if pd.notna(net_profit):
                features['profit_margin'] = (net_profit / sales) * 100
            if pd.notna(expenses):
                features['expense_ratio'] = (expenses / sales) * 100
            if pd.notna(operating_profit) and pd.notna(depreciation):
                features['ebit_margin'] = ((operating_profit - depreciation) / sales) * 100

        return features

    def prepare_monthly_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Prepare monthly data for a single stock

        Args:
            ticker: Stock ticker
            start_date: Start date (e.g., '2014-12-31')
            end_date: End date (e.g., '2024-11-30')

        Returns:
            DataFrame with one row per month, containing:
            - month_end_date
            - fundamental features (from most recent reported quarter)
            - staleness (days since quarter ended)
            - quarter_date (which quarter's data we're using)
        """
        # Get all quarterly data for this stock
        quarterly_data = self.get_quarterly_data(ticker)

        if quarterly_data.empty:
            return pd.DataFrame()

        # Generate list of month-end dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        month_ends = pd.date_range(start=start, end=end, freq='M')

        monthly_data = []

        for month_end in month_ends:
            # Get most recent REPORTED quarter as of this month
            latest_quarter = self.get_latest_reported_quarter(month_end, quarterly_data)

            if latest_quarter.empty:
                # No fundamental data available yet
                continue

            quarter_date = latest_quarter['quarter_date'].iloc[0]

            # Calculate staleness (how old is this fundamental data?)
            staleness_days = (month_end - quarter_date).days

            # Create features
            fundamental_features = self.create_fundamental_features(latest_quarter)
            growth_features = self.create_growth_features(ticker, latest_quarter, quarterly_data)

            # Combine all features
            row = {
                'ticker': ticker,
                'month_end_date': month_end,
                'quarter_date': quarter_date,
                'staleness_days': staleness_days,
                **fundamental_features,
                **growth_features
            }

            monthly_data.append(row)

        if not monthly_data:
            return pd.DataFrame()

        df = pd.DataFrame(monthly_data)
        return df

    def prepare_training_data(self, tickers: List[str], start_date: str = '2014-12-31',
                             end_date: str = '2024-11-30') -> pd.DataFrame:
        """
        Prepare monthly training data for multiple stocks

        Args:
            tickers: List of stock tickers
            start_date: Start date (10 years of data)
            end_date: End date (latest month)

        Returns:
            Combined DataFrame with all stocks' monthly data
        """
        print(f"\n{'='*80}")
        print("PREPARING MONTHLY TRAINING DATA")
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
                print(f"[WARNING] Error processing {ticker}: {str(e)}")
                continue

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        print(f"\n[OK] Monthly data collection complete")
        print(f"    Total months: {len(combined)}")
        print(f"    Unique stocks: {combined['ticker'].nunique()}")
        print(f"    Features: {len(combined.columns)}")
        print(f"    Date range: {combined['month_end_date'].min().date()} to {combined['month_end_date'].max().date()}")

        # Check for missing values
        fundamental_cols = [col for col in combined.columns
                          if col not in ['ticker', 'month_end_date', 'quarter_date', 'staleness_days']]

        if fundamental_cols:
            missing_pct = combined[fundamental_cols].isna().sum().sum() / (len(combined) * len(fundamental_cols)) * 100
            print(f"    Missing values: {missing_pct:.1f}%")

        return combined
