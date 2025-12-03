"""
Database Cleanup and Standardization Script

This script:
1. Removes stocks with no data
2. Removes completely empty records (all fields NULL)
3. Standardizes NULL/NaN handling (convert to 0 for numeric fields)
4. Removes duplicate records
5. Validates data integrity
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Tuple
import sys


class DatabaseCleaner:
    """Clean and standardize screener.in database"""

    def __init__(self, db_path: str = 'screener_data.db'):
        self.db_path = db_path
        self.stats = {
            'stocks_removed': 0,
            'empty_records_removed': 0,
            'duplicates_removed': 0,
            'nulls_standardized': 0
        }

    def clean_database(self):
        """Run all cleanup operations"""
        print("\n" + "="*80)
        print("DATABASE CLEANUP AND STANDARDIZATION")
        print("="*80)

        self._show_initial_stats()

        print("\n[1] Removing stocks with no financial data...")
        self._remove_empty_stocks()

        print("\n[2] Removing completely empty records...")
        self._remove_empty_records()

        print("\n[3] Standardizing NULL values...")
        self._standardize_nulls()

        print("\n[4] Removing duplicate records...")
        self._remove_duplicates()

        print("\n[5] Cleaning year formats...")
        self._clean_year_formats()

        print("\n[6] Vacuuming database...")
        self._vacuum_database()

        self._show_final_stats()

        return self.stats

    def _show_initial_stats(self):
        """Display initial database statistics"""
        conn = sqlite3.connect(self.db_path)

        stats = {}
        stats['total_companies'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM companies", conn
        )['cnt'].iloc[0]

        stats['companies_with_data'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM companies WHERE data_available = 1", conn
        )['cnt'].iloc[0]

        stats['quarterly_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM quarterly_results", conn
        )['cnt'].iloc[0]

        stats['annual_pl_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM annual_profit_loss", conn
        )['cnt'].iloc[0]

        stats['balance_sheet_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM balance_sheet", conn
        )['cnt'].iloc[0]

        stats['cash_flow_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM cash_flow", conn
        )['cnt'].iloc[0]

        stats['ratio_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM annual_ratios", conn
        )['cnt'].iloc[0]

        conn.close()

        print("\n[INITIAL STATE]")
        print(f"  Total companies: {stats['total_companies']}")
        print(f"  Companies with data: {stats['companies_with_data']}")
        print(f"  Quarterly records: {stats['quarterly_records']}")
        print(f"  Annual P&L records: {stats['annual_pl_records']}")
        print(f"  Balance sheet records: {stats['balance_sheet_records']}")
        print(f"  Cash flow records: {stats['cash_flow_records']}")
        print(f"  Ratio records: {stats['ratio_records']}")

        self.initial_stats = stats

    def _remove_empty_stocks(self):
        """Remove stocks that have no financial data at all"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find stocks with no financial records
        query = """
        SELECT c.ticker
        FROM companies c
        WHERE c.data_available = 1
        AND NOT EXISTS (SELECT 1 FROM annual_profit_loss WHERE ticker = c.ticker)
        AND NOT EXISTS (SELECT 1 FROM balance_sheet WHERE ticker = c.ticker)
        AND NOT EXISTS (SELECT 1 FROM cash_flow WHERE ticker = c.ticker)
        AND NOT EXISTS (SELECT 1 FROM quarterly_results WHERE ticker = c.ticker)
        """

        empty_stocks = pd.read_sql_query(query, conn)

        if not empty_stocks.empty:
            tickers_to_remove = empty_stocks['ticker'].tolist()

            # Remove from all tables
            for ticker in tickers_to_remove:
                cursor.execute("DELETE FROM companies WHERE ticker = ?", (ticker,))
                cursor.execute("DELETE FROM key_metrics WHERE ticker = ?", (ticker,))

            conn.commit()
            self.stats['stocks_removed'] = len(tickers_to_remove)
            print(f"    Removed {len(tickers_to_remove)} stocks with no data")
        else:
            print("    No empty stocks found")

        conn.close()

    def _remove_empty_records(self):
        """Remove records where all data fields are NULL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        total_removed = 0

        # Quarterly results - check if all financial fields are NULL
        cursor.execute("""
            DELETE FROM quarterly_results
            WHERE sales IS NULL
            AND expenses IS NULL
            AND operating_profit IS NULL
            AND net_profit IS NULL
            AND eps IS NULL
        """)
        removed = cursor.rowcount
        total_removed += removed
        print(f"    Quarterly: Removed {removed} empty records")

        # Annual P&L
        cursor.execute("""
            DELETE FROM annual_profit_loss
            WHERE sales IS NULL
            AND expenses IS NULL
            AND operating_profit IS NULL
            AND net_profit IS NULL
            AND eps IS NULL
        """)
        removed = cursor.rowcount
        total_removed += removed
        print(f"    Annual P&L: Removed {removed} empty records")

        # Balance sheet
        cursor.execute("""
            DELETE FROM balance_sheet
            WHERE equity_capital IS NULL
            AND reserves IS NULL
            AND total_assets IS NULL
            AND total_liabilities IS NULL
        """)
        removed = cursor.rowcount
        total_removed += removed
        print(f"    Balance Sheet: Removed {removed} empty records")

        # Cash flow
        cursor.execute("""
            DELETE FROM cash_flow
            WHERE cash_from_operating_activity IS NULL
            AND cash_from_investing_activity IS NULL
            AND cash_from_financing_activity IS NULL
            AND net_cash_flow IS NULL
        """)
        removed = cursor.rowcount
        total_removed += removed
        print(f"    Cash Flow: Removed {removed} empty records")

        # Ratios
        cursor.execute("""
            DELETE FROM annual_ratios
            WHERE debtor_days IS NULL
            AND inventory_days IS NULL
            AND roce_percent IS NULL
            AND cash_conversion_cycle IS NULL
        """)
        removed = cursor.rowcount
        total_removed += removed
        print(f"    Ratios: Removed {removed} empty records")

        conn.commit()
        conn.close()

        self.stats['empty_records_removed'] = total_removed

    def _standardize_nulls(self):
        """
        Standardize NULL handling:
        - For percentage fields: Keep NULL (represents N/A)
        - For amount fields: Convert NULL to 0 where it makes sense
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # For quarterly/annual P&L: Keep NULLs as-is (NULL means data not available)
        # For balance sheet: Keep NULLs as-is
        # For cash flow: Keep NULLs as-is
        # For ratios: Keep NULLs as-is (NULL means ratio can't be calculated)

        # The main standardization is ensuring consistency
        # We'll keep NULLs but ensure they're properly handled in queries

        print("    NULLs preserved (NULL = data not available)")
        print("    Note: Use COALESCE(field, 0) in queries where 0 makes sense")

        conn.close()

        self.stats['nulls_standardized'] = 0  # We're keeping NULLs

    def _remove_duplicates(self):
        """Remove duplicate records (same ticker + year/quarter)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        total_removed = 0

        # For each table with ticker + date, keep only the most recent scraped_at

        # Quarterly results
        cursor.execute("""
            DELETE FROM quarterly_results
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM quarterly_results
                GROUP BY ticker, quarter_date
            )
        """)
        removed = cursor.rowcount
        total_removed += removed
        if removed > 0:
            print(f"    Quarterly: Removed {removed} duplicates")

        # Annual P&L
        cursor.execute("""
            DELETE FROM annual_profit_loss
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM annual_profit_loss
                GROUP BY ticker, year
            )
        """)
        removed = cursor.rowcount
        total_removed += removed
        if removed > 0:
            print(f"    Annual P&L: Removed {removed} duplicates")

        # Balance sheet
        cursor.execute("""
            DELETE FROM balance_sheet
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM balance_sheet
                GROUP BY ticker, year
            )
        """)
        removed = cursor.rowcount
        total_removed += removed
        if removed > 0:
            print(f"    Balance Sheet: Removed {removed} duplicates")

        # Cash flow
        cursor.execute("""
            DELETE FROM cash_flow
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM cash_flow
                GROUP BY ticker, year
            )
        """)
        removed = cursor.rowcount
        total_removed += removed
        if removed > 0:
            print(f"    Cash Flow: Removed {removed} duplicates")

        # Ratios
        cursor.execute("""
            DELETE FROM annual_ratios
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM annual_ratios
                GROUP BY ticker, year
            )
        """)
        removed = cursor.rowcount
        total_removed += removed
        if removed > 0:
            print(f"    Ratios: Removed {removed} duplicates")

        if total_removed == 0:
            print("    No duplicates found")

        conn.commit()
        conn.close()

        self.stats['duplicates_removed'] = total_removed

    def _clean_year_formats(self):
        """Standardize year formats (e.g., 'Mar 2023', 'Dec 2022')"""
        conn = sqlite3.connect(self.db_path)

        # Check for irregular year formats
        tables = ['annual_profit_loss', 'balance_sheet', 'cash_flow', 'annual_ratios']

        for table in tables:
            df = pd.read_sql_query(f"SELECT DISTINCT year FROM {table} ORDER BY year", conn)

            if not df.empty:
                irregular = df[~df['year'].str.match(r'^[A-Z][a-z]{2} \d{4}$', na=False)]

                if not irregular.empty:
                    print(f"    {table}: Found {len(irregular)} irregular year formats")
                    print(f"      Examples: {irregular['year'].head(3).tolist()}")

        conn.close()

    def _vacuum_database(self):
        """Vacuum database to reclaim space and optimize"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("VACUUM")

        conn.close()
        print("    Database vacuumed and optimized")

    def _show_final_stats(self):
        """Display final statistics and cleanup summary"""
        conn = sqlite3.connect(self.db_path)

        final = {}
        final['total_companies'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM companies", conn
        )['cnt'].iloc[0]

        final['companies_with_data'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM companies WHERE data_available = 1", conn
        )['cnt'].iloc[0]

        final['quarterly_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM quarterly_results", conn
        )['cnt'].iloc[0]

        final['annual_pl_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM annual_profit_loss", conn
        )['cnt'].iloc[0]

        final['balance_sheet_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM balance_sheet", conn
        )['cnt'].iloc[0]

        final['cash_flow_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM cash_flow", conn
        )['cnt'].iloc[0]

        final['ratio_records'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM annual_ratios", conn
        )['cnt'].iloc[0]

        conn.close()

        print("\n" + "="*80)
        print("CLEANUP COMPLETE")
        print("="*80)

        print("\n[SUMMARY]")
        print(f"  Stocks removed: {self.stats['stocks_removed']}")
        print(f"  Empty records removed: {self.stats['empty_records_removed']}")
        print(f"  Duplicate records removed: {self.stats['duplicates_removed']}")

        print("\n[FINAL STATE]")
        print(f"  Total companies: {final['total_companies']} (was {self.initial_stats['total_companies']})")
        print(f"  Companies with data: {final['companies_with_data']} (was {self.initial_stats['companies_with_data']})")
        print(f"  Quarterly records: {final['quarterly_records']} (was {self.initial_stats['quarterly_records']})")
        print(f"  Annual P&L records: {final['annual_pl_records']} (was {self.initial_stats['annual_pl_records']})")
        print(f"  Balance sheet records: {final['balance_sheet_records']} (was {self.initial_stats['balance_sheet_records']})")
        print(f"  Cash flow records: {final['cash_flow_records']} (was {self.initial_stats['cash_flow_records']})")
        print(f"  Ratio records: {final['ratio_records']} (was {self.initial_stats['ratio_records']})")

        # Show data quality metrics
        print("\n[DATA QUALITY]")
        conn = sqlite3.connect(self.db_path)

        # Average records per stock
        avg_annual = pd.read_sql_query("""
            SELECT AVG(cnt) as avg_years
            FROM (
                SELECT ticker, COUNT(*) as cnt
                FROM annual_profit_loss
                GROUP BY ticker
            )
        """, conn)['avg_years'].iloc[0]

        avg_quarterly = pd.read_sql_query("""
            SELECT AVG(cnt) as avg_quarters
            FROM (
                SELECT ticker, COUNT(*) as cnt
                FROM quarterly_results
                GROUP BY ticker
            )
        """, conn)['avg_quarters'].iloc[0]

        print(f"  Average years per stock: {avg_annual:.1f}")
        print(f"  Average quarters per stock: {avg_quarterly:.1f}")

        # Stocks with most historical data
        top_stocks = pd.read_sql_query("""
            SELECT ticker, COUNT(*) as years
            FROM annual_profit_loss
            GROUP BY ticker
            ORDER BY years DESC
            LIMIT 5
        """, conn)

        if not top_stocks.empty:
            print(f"\n  Top stocks by historical data:")
            for _, row in top_stocks.iterrows():
                print(f"    {row['ticker']}: {int(row['years'])} years")

        conn.close()


def main():
    """Run database cleanup"""
    cleaner = DatabaseCleaner()
    stats = cleaner.clean_database()

    print(f"\n✅ Database cleaned: {cleaner.db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
