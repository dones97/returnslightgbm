"""
Enhanced Data Collection Module - Optimized for Fundamental Analysis
Extracts quarterly fundamental data and calculates time-varying ratios
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EnhancedStockDataCollector:
    """Enhanced data collector focused on fundamental analysis"""

    def __init__(self, lookback_years: int = 5):
        """
        Initialize enhanced data collector

        Args:
            lookback_years: Number of years of historical data to fetch
        """
        self.lookback_years = lookback_years
        self.start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')

    def fetch_stock_data(self, ticker: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch comprehensive data for a stock

        Returns:
            Tuple of (price_data, info, quarterly_financials, quarterly_balance, quarterly_cashflow)
        """
        try:
            stock = yf.Ticker(ticker)

            # Fetch historical price data
            hist_data = stock.history(start=self.start_date, end=self.end_date)
            if hist_data.empty:
                return None, None, None, None, None

            # Fetch fundamental data
            info = stock.info

            # Fetch quarterly financial statements
            try:
                quarterly_financials = stock.quarterly_financials
            except:
                quarterly_financials = pd.DataFrame()

            try:
                quarterly_balance = stock.quarterly_balance_sheet
            except:
                quarterly_balance = pd.DataFrame()

            try:
                quarterly_cashflow = stock.quarterly_cashflow
            except:
                quarterly_cashflow = pd.DataFrame()

            return hist_data, info, quarterly_financials, quarterly_balance, quarterly_cashflow

        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None, None, None, None, None

    def extract_quarterly_fundamentals(self,
                                      quarterly_financials: pd.DataFrame,
                                      quarterly_balance: pd.DataFrame,
                                      quarterly_cashflow: pd.DataFrame,
                                      price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and calculate quarterly fundamental metrics with time-varying ratios

        Returns:
            DataFrame with quarterly fundamentals indexed by date
        """
        quarterly_data = pd.DataFrame()

        if quarterly_financials.empty and quarterly_balance.empty:
            return quarterly_data

        # Get dates (columns are dates in quarterly data)
        all_dates = []
        if not quarterly_financials.empty:
            all_dates.extend(quarterly_financials.columns.tolist())
        if not quarterly_balance.empty:
            all_dates.extend(quarterly_balance.columns.tolist())

        dates = sorted(set(all_dates))

        if not dates:
            return quarterly_data

        # Extract metrics for each quarter
        metrics_by_quarter = []

        for date in dates:
            quarter_metrics = {'Date': date}

            # Income Statement metrics
            if not quarterly_financials.empty and date in quarterly_financials.columns:
                fin = quarterly_financials[date]

                quarter_metrics['Total_Revenue'] = fin.get('Total Revenue', np.nan)
                quarter_metrics['Gross_Profit'] = fin.get('Gross Profit', np.nan)
                quarter_metrics['Operating_Income'] = fin.get('Operating Income', np.nan)
                quarter_metrics['EBITDA'] = fin.get('EBITDA', np.nan)
                quarter_metrics['EBIT'] = fin.get('EBIT', np.nan)
                quarter_metrics['Net_Income'] = fin.get('Net Income', np.nan)
                quarter_metrics['Tax_Provision'] = fin.get('Tax Provision', np.nan)
                quarter_metrics['Interest_Expense'] = fin.get('Interest Expense', np.nan)

                # Calculate margins
                revenue = quarter_metrics['Total_Revenue']
                if revenue and revenue > 0:
                    quarter_metrics['Gross_Margin'] = quarter_metrics['Gross_Profit'] / revenue if quarter_metrics.get('Gross_Profit') else np.nan
                    quarter_metrics['Operating_Margin'] = quarter_metrics['Operating_Income'] / revenue if quarter_metrics.get('Operating_Income') else np.nan
                    quarter_metrics['Net_Margin'] = quarter_metrics['Net_Income'] / revenue if quarter_metrics.get('Net_Income') else np.nan
                    quarter_metrics['EBITDA_Margin'] = quarter_metrics['EBITDA'] / revenue if quarter_metrics.get('EBITDA') else np.nan
                else:
                    quarter_metrics['Gross_Margin'] = np.nan
                    quarter_metrics['Operating_Margin'] = np.nan
                    quarter_metrics['Net_Margin'] = np.nan
                    quarter_metrics['EBITDA_Margin'] = np.nan

            # Balance Sheet metrics
            if not quarterly_balance.empty and date in quarterly_balance.columns:
                bal = quarterly_balance[date]

                quarter_metrics['Total_Assets'] = bal.get('Total Assets', np.nan)
                quarter_metrics['Total_Debt'] = bal.get('Total Debt', np.nan)
                quarter_metrics['Total_Liabilities'] = bal.get('Total Liabilities Net Minority Interest', np.nan)
                quarter_metrics['Stockholders_Equity'] = bal.get('Stockholders Equity', np.nan)
                quarter_metrics['Current_Assets'] = bal.get('Current Assets', np.nan)
                quarter_metrics['Current_Liabilities'] = bal.get('Current Liabilities', np.nan)
                quarter_metrics['Cash'] = bal.get('Cash And Cash Equivalents', np.nan)
                quarter_metrics['Inventory'] = bal.get('Inventory', np.nan)

                # Calculate ratios
                equity = quarter_metrics['Stockholders_Equity']
                assets = quarter_metrics['Total_Assets']
                debt = quarter_metrics['Total_Debt']

                if equity and equity != 0:
                    quarter_metrics['Debt_to_Equity'] = debt / equity if debt else 0
                    # ROE will be calculated with trailing 12-month earnings
                else:
                    quarter_metrics['Debt_to_Equity'] = np.nan

                if assets and assets != 0:
                    quarter_metrics['Debt_to_Assets'] = debt / assets if debt else 0
                    # ROA will be calculated with trailing 12-month earnings
                else:
                    quarter_metrics['Debt_to_Assets'] = np.nan

                # Current ratio
                curr_assets = quarter_metrics['Current_Assets']
                curr_liab = quarter_metrics['Current_Liabilities']
                if curr_liab and curr_liab != 0:
                    quarter_metrics['Current_Ratio'] = curr_assets / curr_liab if curr_assets else 0
                else:
                    quarter_metrics['Current_Ratio'] = np.nan

                # Quick ratio (Current Assets - Inventory) / Current Liabilities
                inventory = quarter_metrics.get('Inventory', 0) or 0
                if curr_liab and curr_liab != 0 and curr_assets:
                    quarter_metrics['Quick_Ratio'] = (curr_assets - inventory) / curr_liab
                else:
                    quarter_metrics['Quick_Ratio'] = np.nan

            # Cash Flow metrics
            if not quarterly_cashflow.empty and date in quarterly_cashflow.columns:
                cf = quarterly_cashflow[date]

                quarter_metrics['Operating_Cash_Flow'] = cf.get('Operating Cash Flow', np.nan)
                quarter_metrics['Free_Cash_Flow'] = cf.get('Free Cash Flow', np.nan)
                quarter_metrics['Capital_Expenditure'] = cf.get('Capital Expenditure', np.nan)

            metrics_by_quarter.append(quarter_metrics)

        # Create DataFrame
        quarterly_data = pd.DataFrame(metrics_by_quarter)

        # Convert Date column and strip timezone
        quarterly_data['Date'] = pd.to_datetime(quarterly_data['Date'])
        if hasattr(quarterly_data['Date'].dtype, 'tz') and quarterly_data['Date'].dtype.tz is not None:
            quarterly_data['Date'] = quarterly_data['Date'].dt.tz_localize(None)

        # CRITICAL FIX: Ensure ALL columns are proper numeric types
        # Some stocks may have accidentally stored datetime objects in numeric columns
        for col in quarterly_data.columns:
            if col != 'Date':
                # Force convert to numeric, coercing any non-numeric (like datetime) to NaN
                quarterly_data[col] = pd.to_numeric(quarterly_data[col], errors='coerce')

        quarterly_data = quarterly_data.sort_values('Date')
        quarterly_data = quarterly_data.set_index('Date')

        # Calculate growth rates (YoY and QoQ)
        if len(quarterly_data) >= 2:
            if 'Total_Revenue' in quarterly_data.columns:
                quarterly_data['Revenue_Growth_QoQ'] = quarterly_data['Total_Revenue'].pct_change()
            if 'Net_Income' in quarterly_data.columns:
                quarterly_data['Net_Income_Growth_QoQ'] = quarterly_data['Net_Income'].pct_change()

        if len(quarterly_data) >= 4:  # YoY needs 4 quarters
            if 'Total_Revenue' in quarterly_data.columns:
                quarterly_data['Revenue_Growth_YoY'] = quarterly_data['Total_Revenue'].pct_change(periods=4)
            if 'Net_Income' in quarterly_data.columns:
                quarterly_data['Net_Income_Growth_YoY'] = quarterly_data['Net_Income'].pct_change(periods=4)
            if 'EBITDA' in quarterly_data.columns:
                quarterly_data['EBITDA_Growth_YoY'] = quarterly_data['EBITDA'].pct_change(periods=4)

        # Calculate trailing 12-month metrics
        if len(quarterly_data) >= 4:
            # Only calculate TTM for columns that exist
            if 'Total_Revenue' in quarterly_data.columns:
                quarterly_data['TTM_Revenue'] = quarterly_data['Total_Revenue'].rolling(window=4, min_periods=4).sum()

            if 'Net_Income' in quarterly_data.columns:
                quarterly_data['TTM_Net_Income'] = quarterly_data['Net_Income'].rolling(window=4, min_periods=4).sum()

            if 'Operating_Income' in quarterly_data.columns:
                quarterly_data['TTM_Operating_Income'] = quarterly_data['Operating_Income'].rolling(window=4, min_periods=4).sum()

            if 'EBITDA' in quarterly_data.columns:
                quarterly_data['TTM_EBITDA'] = quarterly_data['EBITDA'].rolling(window=4, min_periods=4).sum()

            if 'Operating_Cash_Flow' in quarterly_data.columns:
                quarterly_data['TTM_Operating_Cash_Flow'] = quarterly_data['Operating_Cash_Flow'].rolling(window=4, min_periods=4).sum()

            # Calculate ROE and ROA using TTM earnings
            if 'TTM_Net_Income' in quarterly_data.columns and 'Stockholders_Equity' in quarterly_data.columns:
                quarterly_data['ROE'] = quarterly_data['TTM_Net_Income'] / quarterly_data['Stockholders_Equity']

            if 'TTM_Net_Income' in quarterly_data.columns and 'Total_Assets' in quarterly_data.columns:
                quarterly_data['ROA'] = quarterly_data['TTM_Net_Income'] / quarterly_data['Total_Assets']

            # Calculate ROIC (Return on Invested Capital)
            if ('TTM_Operating_Income' in quarterly_data.columns and
                'Stockholders_Equity' in quarterly_data.columns and
                'Total_Debt' in quarterly_data.columns):
                invested_capital = quarterly_data['Stockholders_Equity'] + quarterly_data['Total_Debt']
                quarterly_data['ROIC'] = quarterly_data['TTM_Operating_Income'] / invested_capital

        # Calculate price-based ratios if price data available
        if not price_data.empty and len(quarterly_data) > 0:
            for idx in quarterly_data.index:
                # Find closest price date to quarter end
                try:
                    closest_price_date = price_data.index[price_data.index.get_indexer([idx], method='nearest')[0]]
                    price = price_data.loc[closest_price_date, 'Close']

                    # Get shares outstanding from info (approximate, but better than nothing)
                    # This would ideally come from quarterly data

                    # P/E ratio (using TTM earnings if available)
                    ttm_net_income = quarterly_data.loc[idx, 'TTM_Net_Income']
                    stockholders_equity = quarterly_data.loc[idx, 'Stockholders_Equity']

                    # P/B ratio
                    if stockholders_equity and stockholders_equity > 0:
                        # Store price for ratio calculation (will need market cap)
                        quarterly_data.loc[idx, 'Price'] = price

                except:
                    pass

        return quarterly_data

    def merge_quarterly_to_monthly(self, monthly_price_data: pd.DataFrame,
                                  quarterly_fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Merge quarterly fundamental data to monthly price data using forward-fill

        Args:
            monthly_price_data: Monthly price/technical data
            quarterly_fundamentals: Quarterly fundamental data

        Returns:
            Merged DataFrame with fundamentals forward-filled monthly
        """
        if quarterly_fundamentals.empty:
            return monthly_price_data

        # Reset index for both
        monthly_price_data = monthly_price_data.reset_index()
        quarterly_fundamentals = quarterly_fundamentals.reset_index()

        # Remove timezone info to avoid merge issues
        if pd.api.types.is_datetime64_any_dtype(monthly_price_data['Date']):
            monthly_price_data['Date'] = pd.to_datetime(monthly_price_data['Date']).dt.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(quarterly_fundamentals['Date']):
            quarterly_fundamentals['Date'] = pd.to_datetime(quarterly_fundamentals['Date']).dt.tz_localize(None)

        # Merge with forward fill
        merged = pd.merge_asof(
            monthly_price_data.sort_values('Date'),
            quarterly_fundamentals.sort_values('Date'),
            on='Date',
            direction='backward'  # Use most recent quarterly data
        )

        merged = merged.set_index('Date')
        return merged

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators (simplified for fundamental focus)"""
        df = df.copy()

        # Key momentum indicators
        df['ROC_1M'] = df['Close'].pct_change(periods=21)
        df['ROC_3M'] = df['Close'].pct_change(periods=63)
        df['ROC_6M'] = df['Close'].pct_change(periods=126)
        df['ROC_12M'] = df['Close'].pct_change(periods=252)

        # Volatility
        df['Volatility_60'] = df['Close'].pct_change().rolling(window=60).std() * np.sqrt(252)

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volume trend
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        return df

    def compute_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to monthly and compute returns"""
        monthly = df.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'ROC_1M': 'last',
            'ROC_3M': 'last',
            'ROC_6M': 'last',
            'ROC_12M': 'last',
            'Volatility_60': 'mean',
            'RSI': 'last',
            'Volume_Ratio': 'mean'
        })

        # Compute forward returns
        monthly['Return_Next_Month'] = monthly['Close'].pct_change().shift(-1)
        monthly['Return_Direction'] = (monthly['Return_Next_Month'] > 0).astype(int)

        return monthly

    def calculate_advanced_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced fundamental scores (Piotroski F-Score, Altman Z-Score)
        """
        df = df.copy()

        # Piotroski F-Score (simplified version with available data)
        df['F_Score'] = 0

        # Profitability (4 points)
        df['F_Score'] += (df['ROA'] > 0).astype(int) if 'ROA' in df.columns else 0
        df['F_Score'] += (df['Operating_Cash_Flow'] > 0).astype(int) if 'Operating_Cash_Flow' in df.columns else 0
        df['F_Score'] += (df['ROA'].diff() > 0).astype(int) if 'ROA' in df.columns else 0
        df['F_Score'] += ((df['Operating_Cash_Flow'] > df['Net_Income']).astype(int)
                         if 'Operating_Cash_Flow' in df.columns and 'Net_Income' in df.columns else 0)

        # Leverage, Liquidity (3 points)
        df['F_Score'] += (df['Debt_to_Assets'].diff() < 0).astype(int) if 'Debt_to_Assets' in df.columns else 0
        df['F_Score'] += (df['Current_Ratio'].diff() > 0).astype(int) if 'Current_Ratio' in df.columns else 0

        # Operating Efficiency (2 points)
        df['F_Score'] += (df['Gross_Margin'].diff() > 0).astype(int) if 'Gross_Margin' in df.columns else 0
        df['F_Score'] += (df['TTM_Revenue'].pct_change() > 0).astype(int) if 'TTM_Revenue' in df.columns else 0

        # Altman Z-Score (simplified for available data)
        # Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
        # X1 = Working Capital / Total Assets
        # X2 = Retained Earnings / Total Assets (approximated)
        # X3 = EBIT / Total Assets
        # X4 = Market Cap / Total Liabilities (approximated)
        # X5 = Sales / Total Assets

        if all(col in df.columns for col in ['Current_Assets', 'Current_Liabilities', 'Total_Assets']):
            working_capital = df['Current_Assets'] - df['Current_Liabilities']
            x1 = working_capital / df['Total_Assets']

            x2 = df['Stockholders_Equity'] / df['Total_Assets'] if 'Stockholders_Equity' in df.columns else 0
            x3 = df['EBIT'] / df['Total_Assets'] if 'EBIT' in df.columns else 0
            x5 = df['TTM_Revenue'] / df['Total_Assets'] if 'TTM_Revenue' in df.columns else 0

            df['Z_Score'] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 1.0*x5
            # Note: X4 omitted due to lack of real-time market cap in historical data

        return df

    def create_feature_dataframe(self, ticker: str) -> pd.DataFrame:
        """
        Create complete feature dataframe with time-varying fundamentals
        """
        # Fetch all data
        price_data, info, quarterly_financials, quarterly_balance, quarterly_cashflow = self.fetch_stock_data(ticker)

        if price_data is None or price_data.empty:
            return None

        # Compute technical indicators
        price_data_with_indicators = self.compute_technical_indicators(price_data)

        # Extract quarterly fundamentals with time-varying ratios
        quarterly_fundamentals = self.extract_quarterly_fundamentals(
            quarterly_financials, quarterly_balance, quarterly_cashflow, price_data
        )

        # Convert to monthly
        monthly_price_data = self.compute_monthly_returns(price_data_with_indicators)

        # Merge quarterly fundamentals to monthly (forward-fill)
        if not quarterly_fundamentals.empty:
            monthly_data = self.merge_quarterly_to_monthly(monthly_price_data, quarterly_fundamentals)
        else:
            monthly_data = monthly_price_data

        # Calculate advanced scores
        if not quarterly_fundamentals.empty:
            monthly_data = self.calculate_advanced_scores(monthly_data)

        # Add current fundamental snapshot (for any missing historical data)
        if info:
            monthly_data['market_cap_current'] = info.get('marketCap', np.nan)
            monthly_data['beta'] = info.get('beta', np.nan)
            monthly_data['dividend_yield'] = info.get('dividendYield', np.nan)

        # Add ticker
        monthly_data['Ticker'] = ticker

        # FINAL SAFETY CHECK: Ensure no datetime objects in numeric columns
        # This catches any edge cases where datetime might have slipped through
        # Force convert ALL columns to numeric (except Date and Ticker)
        for col in monthly_data.columns:
            if col not in ['Date', 'Ticker']:
                # Force convert to numeric, regardless of current dtype
                # This handles: object, datetime64, or any mixed types
                monthly_data[col] = pd.to_numeric(monthly_data[col], errors='coerce')

        return monthly_data


def collect_data_for_universe(tickers: List[str], lookback_years: int = 5) -> pd.DataFrame:
    """
    Collect enhanced data for a universe of stocks
    """
    collector = EnhancedStockDataCollector(lookback_years=lookback_years)

    all_data = []

    for i, ticker in enumerate(tickers):
        print(f"Processing {ticker} ({i+1}/{len(tickers)})...")

        df = collector.create_feature_dataframe(ticker)

        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, axis=0)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)

    # Ensure Date column is properly formatted and timezone-naive
    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        if hasattr(combined_df['Date'].dtype, 'tz') and combined_df['Date'].dtype.tz is not None:
            combined_df['Date'] = combined_df['Date'].dt.tz_localize(None)

    # CRITICAL: One final aggressive type conversion on the combined dataset
    # This ensures absolutely no datetime objects remain in numeric columns
    print("\n🔧 Final data type cleanup...")
    for col in combined_df.columns:
        if col not in ['Date', 'Ticker']:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    print("✅ Data types verified")

    return combined_df
