"""
Fundamental Ratios Calculator

Calculate size-agnostic financial ratios that actually matter for stock returns:
- Growth rates (sales, EPS, profit)
- Profitability (ROE, ROCE, margins)
- Efficiency (asset turnover, working capital)
- Quality (Piotroski score, Altman Z-score)
- Risk (debt ratios, volatility)

NO absolute numbers - only ratios and percentages!
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict


class FundamentalRatios:
    """Calculate fundamental ratios from screener.in database"""

    def __init__(self, db_path: str = 'screener_data.db'):
        self.db_path = db_path

    def get_quarterly_data(self, ticker: str) -> pd.DataFrame:
        """Get quarterly data with numeric conversion"""
        conn = sqlite3.connect(self.db_path)

        quarterly = pd.read_sql_query(
            "SELECT * FROM quarterly_results WHERE ticker = ? ORDER BY quarter_date",
            conn, params=(ticker,)
        )

        conn.close()

        if quarterly.empty:
            return pd.DataFrame()

        # Convert dates (format is "Dec 2022", "Jun 2023", etc.)
        quarterly['quarter_date'] = pd.to_datetime(quarterly['quarter_date'], format='%b %Y', errors='coerce')

        # Convert numeric columns (CRITICAL: SQLite stores as TEXT!)
        numeric_cols = [
            'sales', 'expenses', 'operating_profit', 'opm_percent', 'other_income',
            'interest', 'depreciation', 'profit_before_tax', 'tax_percent',
            'net_profit', 'eps'
        ]

        for col in numeric_cols:
            if col in quarterly.columns:
                quarterly[col] = pd.to_numeric(quarterly[col], errors='coerce')

        return quarterly.sort_values('quarter_date').reset_index(drop=True)

    def get_annual_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get annual data (balance sheet, cash flow, ratios, profit/loss)"""
        conn = sqlite3.connect(self.db_path)

        data = {
            'balance_sheet': pd.read_sql_query(
                "SELECT * FROM balance_sheet WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
            'annual_ratios': pd.read_sql_query(
                "SELECT * FROM annual_ratios WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
            'cash_flow': pd.read_sql_query(
                "SELECT * FROM cash_flow WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
            'annual_profit_loss': pd.read_sql_query(
                "SELECT * FROM annual_profit_loss WHERE ticker = ? ORDER BY year",
                conn, params=(ticker,)
            ),
        }

        conn.close()

        # Convert numeric columns in balance sheet
        if not data['balance_sheet'].empty:
            numeric_cols = ['total_assets', 'total_liabilities', 'equity_capital',
                          'reserves', 'borrowings', 'fixed_assets', 'investments']
            for col in numeric_cols:
                if col in data['balance_sheet'].columns:
                    data['balance_sheet'][col] = pd.to_numeric(
                        data['balance_sheet'][col], errors='coerce'
                    )

        # Convert numeric columns in annual ratios
        if not data['annual_ratios'].empty:
            numeric_cols = ['roce_percent', 'debtor_days', 'inventory_days',
                          'days_payable', 'cash_conversion_cycle', 'working_capital_days']
            for col in numeric_cols:
                if col in data['annual_ratios'].columns:
                    data['annual_ratios'][col] = pd.to_numeric(
                        data['annual_ratios'][col], errors='coerce'
                    )

        # Convert numeric columns in cash flow
        if not data['cash_flow'].empty:
            numeric_cols = ['cash_from_operating_activity', 'cash_from_investing_activity',
                          'cash_from_financing_activity', 'net_cash_flow']
            for col in numeric_cols:
                if col in data['cash_flow'].columns:
                    data['cash_flow'][col] = pd.to_numeric(
                        data['cash_flow'][col], errors='coerce'
                    )

        # Convert numeric columns in annual profit/loss
        if not data['annual_profit_loss'].empty:
            numeric_cols = ['sales', 'expenses', 'operating_profit', 'net_profit',
                          'eps', 'opm_percent', 'dividend_payout_percent']
            for col in numeric_cols:
                if col in data['annual_profit_loss'].columns:
                    data['annual_profit_loss'][col] = pd.to_numeric(
                        data['annual_profit_loss'][col], errors='coerce'
                    )

        return data

    def calculate_growth_rates(self, quarterly: pd.DataFrame, idx: int) -> Dict:
        """
        Calculate growth rates (QoQ and YoY)

        These are THE MOST IMPORTANT predictors of stock returns!

        Note: We have operating_profit and EPS, but NOT sales or net_profit in database
        """
        features = {}

        if idx > 0:  # QoQ growth
            prev = quarterly.iloc[idx - 1]
            curr = quarterly.iloc[idx]

            # Use operating_profit and EPS (what we have in database)
            for metric in ['operating_profit', 'eps']:
                prev_val = prev.get(metric, np.nan)
                curr_val = curr.get(metric, np.nan)

                if pd.notna(prev_val) and pd.notna(curr_val) and prev_val != 0:
                    features[f'{metric}_growth_qoq'] = ((curr_val - prev_val) / abs(prev_val)) * 100
                else:
                    features[f'{metric}_growth_qoq'] = np.nan

        if idx >= 4:  # YoY growth
            yoy = quarterly.iloc[idx - 4]
            curr = quarterly.iloc[idx]

            # Use operating_profit and EPS (what we have in database)
            for metric in ['operating_profit', 'eps']:
                yoy_val = yoy.get(metric, np.nan)
                curr_val = curr.get(metric, np.nan)

                if pd.notna(yoy_val) and pd.notna(curr_val) and yoy_val != 0:
                    features[f'{metric}_growth_yoy'] = ((curr_val - yoy_val) / abs(yoy_val)) * 100
                else:
                    features[f'{metric}_growth_yoy'] = np.nan

        return features

    def calculate_profitability_ratios(self, quarter: pd.Series) -> Dict:
        """
        Calculate profitability ratios

        Note: We only have opm_percent (operating margin) in database
        We don't have sales, net_profit, or expenses
        """
        features = {}

        # Operating Profit Margin (already calculated in database)
        features['opm'] = quarter.get('opm_percent', np.nan)

        # We have operating_profit and interest, can calculate interest coverage
        # (moved to calculate_risk_ratios)

        return features

    def calculate_efficiency_ratios(self, quarterly: pd.DataFrame, idx: int) -> Dict:
        """
        Calculate efficiency metrics

        Note: We have operating_profit and EPS, but not sales or net_profit
        """
        features = {}

        # Operating profit growth acceleration (is growth accelerating?)
        if idx >= 2:
            if pd.notna(quarterly.iloc[idx].get('operating_profit', np.nan)) and \
               pd.notna(quarterly.iloc[idx-1].get('operating_profit', np.nan)) and \
               pd.notna(quarterly.iloc[idx-2].get('operating_profit', np.nan)):

                curr_growth = (quarterly.iloc[idx]['operating_profit'] - quarterly.iloc[idx-1]['operating_profit']) / \
                             abs(quarterly.iloc[idx-1]['operating_profit'])
                prev_growth = (quarterly.iloc[idx-1]['operating_profit'] - quarterly.iloc[idx-2]['operating_profit']) / \
                             abs(quarterly.iloc[idx-2]['operating_profit'])

                features['operating_profit_acceleration'] = (curr_growth - prev_growth) * 100

        # EPS growth acceleration
        if idx >= 2:
            if pd.notna(quarterly.iloc[idx].get('eps', np.nan)) and \
               pd.notna(quarterly.iloc[idx-1].get('eps', np.nan)) and \
               pd.notna(quarterly.iloc[idx-2].get('eps', np.nan)):

                curr_growth = (quarterly.iloc[idx]['eps'] - quarterly.iloc[idx-1]['eps']) / \
                             abs(quarterly.iloc[idx-1]['eps'] + 0.01)  # Avoid division by zero
                prev_growth = (quarterly.iloc[idx-1]['eps'] - quarterly.iloc[idx-2]['eps']) / \
                             abs(quarterly.iloc[idx-2]['eps'] + 0.01)

                features['eps_acceleration'] = (curr_growth - prev_growth) * 100

        return features

    def calculate_risk_ratios(self, quarter: pd.Series, annual_data: Dict) -> Dict:
        """
        Calculate risk metrics

        Debt ratios, interest coverage - bankruptcy risk indicators
        """
        features = {}

        operating_profit = quarter.get('operating_profit', np.nan)
        interest = quarter.get('interest', np.nan)

        # Interest coverage (critical risk metric!)
        if pd.notna(operating_profit) and pd.notna(interest) and interest != 0:
            features['interest_coverage'] = operating_profit / interest
        else:
            features['interest_coverage'] = np.nan

        # Calculate debt-to-equity from balance sheet (size-agnostic!)
        # Note: borrowings column is empty in database, use total_liabilities - equity as debt
        if not annual_data['balance_sheet'].empty:
            latest_bs = annual_data['balance_sheet'].iloc[-1]
            total_liabilities = latest_bs.get('total_liabilities', np.nan)
            equity_capital = latest_bs.get('equity_capital', np.nan)
            reserves = latest_bs.get('reserves', np.nan)

            # Total equity = equity_capital + reserves
            if pd.notna(equity_capital) and pd.notna(reserves):
                total_equity = equity_capital + reserves

                # Debt = Total Liabilities - Equity (since Assets = Liabilities + Equity)
                if pd.notna(total_liabilities) and total_equity != 0:
                    debt = total_liabilities - total_equity
                    features['debt_to_equity'] = debt / total_equity if debt > 0 else 0
                else:
                    features['debt_to_equity'] = np.nan
            else:
                features['debt_to_equity'] = np.nan
        else:
            features['debt_to_equity'] = np.nan

        return features

    def calculate_annual_ratios(self, annual_data: Dict) -> Dict:
        """
        Calculate size-agnostic ratios from annual data

        These are CRITICAL - they stay constant for the whole year but are highly predictive!
        """
        features = {}

        # 1. ROCE from annual_ratios table
        if not annual_data['annual_ratios'].empty:
            latest = annual_data['annual_ratios'].iloc[-1]
            features['roce'] = latest.get('roce_percent', np.nan)

            # Working capital efficiency
            features['cash_conversion_cycle'] = latest.get('cash_conversion_cycle', np.nan)
            features['working_capital_days'] = latest.get('working_capital_days', np.nan)
            features['debtor_days'] = latest.get('debtor_days', np.nan)
            features['inventory_days'] = latest.get('inventory_days', np.nan)
            features['days_payable'] = latest.get('days_payable', np.nan)

        # 2. Balance sheet ratios (size-agnostic!)
        if not annual_data['balance_sheet'].empty:
            latest_bs = annual_data['balance_sheet'].iloc[-1]

            total_assets = latest_bs.get('total_assets', np.nan)
            total_liabilities = latest_bs.get('total_liabilities', np.nan)
            equity_capital = latest_bs.get('equity_capital', np.nan)
            reserves = latest_bs.get('reserves', np.nan)
            fixed_assets = latest_bs.get('fixed_assets', np.nan)
            borrowings = latest_bs.get('borrowings', np.nan)

            # Asset efficiency ratios
            if pd.notna(total_assets) and total_assets != 0:
                # Leverage ratio
                if pd.notna(total_liabilities):
                    features['leverage_ratio'] = total_liabilities / total_assets

                # Fixed asset ratio
                if pd.notna(fixed_assets):
                    features['fixed_asset_ratio'] = fixed_assets / total_assets

            # Equity composition
            if pd.notna(equity_capital) and pd.notna(reserves):
                total_equity = equity_capital + reserves
                if total_equity != 0:
                    # Reserve ratio (higher = more retained earnings)
                    features['reserve_ratio'] = reserves / total_equity

        # 3. Cash flow ratios (size-agnostic!)
        if not annual_data['cash_flow'].empty:
            latest_cf = annual_data['cash_flow'].iloc[-1]
            operating_cf = latest_cf.get('cash_from_operating_activity', np.nan)
            net_cf = latest_cf.get('net_cash_flow', np.nan)

            # Get annual profit for comparison
            if not annual_data['annual_profit_loss'].empty:
                latest_pl = annual_data['annual_profit_loss'].iloc[-1]
                net_profit = latest_pl.get('net_profit', np.nan)
                operating_profit = latest_pl.get('operating_profit', np.nan)

                # Cash flow quality (operating CF / net profit)
                # High ratio = good quality earnings (cash-backed)
                if pd.notna(operating_cf) and pd.notna(net_profit) and net_profit != 0:
                    features['cf_to_profit_ratio'] = operating_cf / net_profit

                # Free cash flow indicator
                # Positive net CF = good sign
                if pd.notna(net_cf):
                    features['positive_free_cf'] = 1 if net_cf > 0 else 0

        # 4. Annual profit metrics (size-agnostic!)
        if not annual_data['annual_profit_loss'].empty:
            latest_pl = annual_data['annual_profit_loss'].iloc[-1]

            # ROE calculation if we have annual data
            net_profit = latest_pl.get('net_profit', np.nan)
            if pd.notna(net_profit) and not annual_data['balance_sheet'].empty:
                latest_bs = annual_data['balance_sheet'].iloc[-1]
                equity_capital = latest_bs.get('equity_capital', np.nan)
                reserves = latest_bs.get('reserves', np.nan)

                if pd.notna(equity_capital) and pd.notna(reserves):
                    total_equity = equity_capital + reserves
                    if total_equity != 0:
                        features['roe'] = (net_profit / total_equity) * 100

            # Dividend payout ratio
            features['dividend_payout'] = latest_pl.get('dividend_payout_percent', np.nan)

        return features

    def calculate_quality_scores(self, quarterly: pd.DataFrame, idx: int,
                                annual_data: Dict) -> Dict:
        """
        Calculate quality scores (Piotroski-like)

        Binary signals of financial health
        Note: We have operating_profit, EPS, OPM but not net_profit
        """
        features = {}

        curr = quarterly.iloc[idx]

        # 1. Profitability (is company profitable?)
        # Use operating_profit since we don't have net_profit
        operating_profit = curr.get('operating_profit', np.nan)
        features['is_profitable'] = 1 if pd.notna(operating_profit) and operating_profit > 0 else 0

        # 2. Improving margins (is efficiency improving?)
        if idx >= 4:  # Compare to year ago
            curr_margin = curr.get('opm_percent', np.nan)
            prev_margin = quarterly.iloc[idx-4].get('opm_percent', np.nan)

            if pd.notna(curr_margin) and pd.notna(prev_margin):
                features['improving_margins'] = 1 if curr_margin > prev_margin else 0
            else:
                features['improving_margins'] = 0
        else:
            features['improving_margins'] = 0

        # 3. Positive ROE (calculate from annual data)
        positive_roe = False
        if not annual_data['annual_profit_loss'].empty and not annual_data['balance_sheet'].empty:
            latest_pl = annual_data['annual_profit_loss'].iloc[-1]
            latest_bs = annual_data['balance_sheet'].iloc[-1]

            net_profit = latest_pl.get('net_profit', np.nan)
            equity_capital = latest_bs.get('equity_capital', np.nan)
            reserves = latest_bs.get('reserves', np.nan)

            if pd.notna(net_profit) and pd.notna(equity_capital) and pd.notna(reserves):
                total_equity = equity_capital + reserves
                if total_equity != 0:
                    roe = (net_profit / total_equity) * 100
                    positive_roe = roe > 0

        features['positive_roe'] = 1 if positive_roe else 0

        # 4. Improving ROE
        improving_roe = False
        if len(annual_data['annual_profit_loss']) >= 2 and len(annual_data['balance_sheet']) >= 2:
            curr_pl = annual_data['annual_profit_loss'].iloc[-1]
            prev_pl = annual_data['annual_profit_loss'].iloc[-2]
            curr_bs = annual_data['balance_sheet'].iloc[-1]
            prev_bs = annual_data['balance_sheet'].iloc[-2]

            curr_profit = curr_pl.get('net_profit', np.nan)
            prev_profit = prev_pl.get('net_profit', np.nan)
            curr_equity = curr_bs.get('equity_capital', np.nan) + curr_bs.get('reserves', np.nan) if \
                         pd.notna(curr_bs.get('equity_capital', np.nan)) and pd.notna(curr_bs.get('reserves', np.nan)) else np.nan
            prev_equity = prev_bs.get('equity_capital', np.nan) + prev_bs.get('reserves', np.nan) if \
                         pd.notna(prev_bs.get('equity_capital', np.nan)) and pd.notna(prev_bs.get('reserves', np.nan)) else np.nan

            if pd.notna(curr_profit) and pd.notna(prev_profit) and \
               pd.notna(curr_equity) and pd.notna(prev_equity) and \
               curr_equity != 0 and prev_equity != 0:
                curr_roe = (curr_profit / curr_equity) * 100
                prev_roe = (prev_profit / prev_equity) * 100
                improving_roe = curr_roe > prev_roe

        features['improving_roe'] = 1 if improving_roe else 0

        # 5. Low debt (debt/equity < 1)
        # Calculate from balance sheet (debt = total_liabilities - equity)
        low_debt = False
        if not annual_data['balance_sheet'].empty:
            latest_bs = annual_data['balance_sheet'].iloc[-1]
            total_liabilities = latest_bs.get('total_liabilities', np.nan)
            equity_capital = latest_bs.get('equity_capital', np.nan)
            reserves = latest_bs.get('reserves', np.nan)

            if pd.notna(total_liabilities) and pd.notna(equity_capital) and pd.notna(reserves):
                total_equity = equity_capital + reserves
                if total_equity != 0:
                    debt = total_liabilities - total_equity
                    debt_to_equity = debt / total_equity if debt > 0 else 0
                    low_debt = debt_to_equity < 1

        features['low_debt'] = 1 if low_debt else 0

        # Composite quality score (0-5)
        quality_components = [
            features['is_profitable'],
            features['improving_margins'],
            features['positive_roe'],
            features['improving_roe'],
            features['low_debt']
        ]
        features['quality_score'] = sum(quality_components)

        return features

    def calculate_all_ratios(self, ticker: str, quarter_idx: int) -> Dict:
        """
        Calculate all fundamental ratios for a specific quarter

        Args:
            ticker: Stock ticker
            quarter_idx: Index of quarter in sorted quarterly data

        Returns:
            Dictionary of all ratios
        """
        quarterly = self.get_quarterly_data(ticker)

        if quarterly.empty or quarter_idx >= len(quarterly):
            return {}

        annual_data = self.get_annual_data(ticker)
        curr_quarter = quarterly.iloc[quarter_idx]

        # Combine all ratio categories
        ratios = {}

        # 1. Growth rates (MOST IMPORTANT - quarterly)
        ratios.update(self.calculate_growth_rates(quarterly, quarter_idx))

        # 2. Profitability (quarterly)
        ratios.update(self.calculate_profitability_ratios(curr_quarter))

        # 3. Efficiency (quarterly)
        ratios.update(self.calculate_efficiency_ratios(quarterly, quarter_idx))

        # 4. Risk (quarterly + annual)
        ratios.update(self.calculate_risk_ratios(curr_quarter, annual_data))

        # 5. Quality (quarterly + annual)
        ratios.update(self.calculate_quality_scores(quarterly, quarter_idx, annual_data))

        # 6. Annual ratios (CRITICAL - balance sheet, cash flow, working capital)
        ratios.update(self.calculate_annual_ratios(annual_data))

        return ratios
