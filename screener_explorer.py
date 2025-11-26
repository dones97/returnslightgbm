"""
Screener.in Data Explorer

This script explores screener.in structure to understand:
1. What data is available
2. How to access it
3. Best scraping approach
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Dict, List, Optional
import json

class ScreenerExplorer:
    """Explore screener.in structure and data availability"""

    def __init__(self):
        self.base_url = "https://www.screener.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def explore_stock_page(self, ticker: str) -> Dict:
        """
        Explore a single stock page to see what data is available

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE', 'TCS')

        Returns:
            Dictionary with available data categories
        """
        url = f"{self.base_url}/company/{ticker}/"

        print(f"\n{'='*80}")
        print(f"Exploring: {ticker}")
        print(f"URL: {url}")
        print(f"{'='*80}\n")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            data = {
                'ticker': ticker,
                'url': url,
                'status': 'success',
                'available_data': {}
            }

            # 1. Check for company info
            print("[1] COMPANY INFO:")
            company_name = soup.find('h1', class_='company-name')
            if company_name:
                print(f"   Company Name: {company_name.text.strip()}")
                data['company_name'] = company_name.text.strip()

            # 2. Check for key metrics (ratios section)
            print("\n[2] KEY METRICS:")
            ratios = soup.find('div', id='top-ratios')
            if ratios:
                ratio_items = ratios.find_all('li', class_='flex-row')
                print(f"   Found {len(ratio_items)} key metrics")
                data['available_data']['key_metrics'] = len(ratio_items)

                # Sample first few
                for item in ratio_items[:5]:
                    name = item.find('span', class_='name')
                    value = item.find('span', class_='number')
                    if name and value:
                        print(f"   - {name.text.strip()}: {value.text.strip()}")

            # 3. Check for quarterly results
            print("\n[3] QUARTERLY RESULTS:")
            quarterly_table = soup.find('section', id='quarters')
            if quarterly_table:
                table = quarterly_table.find('table')
                if table:
                    headers = [th.text.strip() for th in table.find_all('th')]
                    rows = table.find_all('tr')[1:]  # Skip header
                    print(f"   Found quarterly table with {len(headers)} columns, {len(rows)} rows")
                    print(f"   Columns: {', '.join(headers[:8])}...")
                    data['available_data']['quarterly_results'] = {
                        'columns': len(headers),
                        'rows': len(rows)
                    }

            # 4. Check for annual results
            print("\n[4] ANNUAL RESULTS:")
            annual_table = soup.find('section', id='profit-loss')
            if annual_table:
                table = annual_table.find('table')
                if table:
                    headers = [th.text.strip() for th in table.find_all('th')]
                    rows = table.find_all('tr')[1:]
                    print(f"   Found annual P&L table with {len(headers)} columns, {len(rows)} rows")
                    data['available_data']['annual_profit_loss'] = {
                        'columns': len(headers),
                        'rows': len(rows)
                    }

            # 5. Check for balance sheet
            print("\n[5] BALANCE SHEET:")
            balance_sheet = soup.find('section', id='balance-sheet')
            if balance_sheet:
                table = balance_sheet.find('table')
                if table:
                    headers = [th.text.strip() for th in table.find_all('th')]
                    rows = table.find_all('tr')[1:]
                    print(f"   Found balance sheet with {len(headers)} columns, {len(rows)} rows")
                    data['available_data']['balance_sheet'] = {
                        'columns': len(headers),
                        'rows': len(rows)
                    }

            # 6. Check for cash flow
            print("\n[6] CASH FLOW:")
            cash_flow = soup.find('section', id='cash-flow')
            if cash_flow:
                table = cash_flow.find('table')
                if table:
                    headers = [th.text.strip() for th in table.find_all('th')]
                    rows = table.find_all('tr')[1:]
                    print(f"   Found cash flow with {len(headers)} columns, {len(rows)} rows")
                    data['available_data']['cash_flow'] = {
                        'columns': len(headers),
                        'rows': len(rows)
                    }

            # 7. Check for ratios table
            print("\n[7] RATIOS TABLE:")
            ratios_section = soup.find('section', id='ratios')
            if ratios_section:
                table = ratios_section.find('table')
                if table:
                    headers = [th.text.strip() for th in table.find_all('th')]
                    rows = table.find_all('tr')[1:]
                    print(f"   Found ratios table with {len(headers)} columns, {len(rows)} rows")
                    data['available_data']['ratios_table'] = {
                        'columns': len(headers),
                        'rows': len(rows)
                    }

            # 8. Check for shareholding pattern
            print("\n[8] SHAREHOLDING PATTERN:")
            shareholding = soup.find('section', id='shareholding')
            if shareholding:
                print(f"   Found shareholding section")
                data['available_data']['shareholding'] = True

            # 9. Check for peer comparison
            print("\n[9] PEER COMPARISON:")
            peers = soup.find('section', id='peers')
            if peers:
                peer_list = peers.find_all('a', class_='company-link')
                print(f"   Found {len(peer_list)} peer companies")
                data['available_data']['peers'] = len(peer_list)

            print(f"\n{'='*80}")
            print("[OK] Exploration complete!")
            print(f"{'='*80}\n")

            return data

        except Exception as e:
            print(f"[ERROR] Error exploring {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'url': url,
                'status': 'error',
                'error': str(e)
            }

    def test_multiple_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """
        Test scraping multiple stocks to see consistency

        Args:
            tickers: List of stock symbols

        Returns:
            DataFrame with exploration results
        """
        results = []

        for ticker in tickers:
            result = self.explore_stock_page(ticker)
            results.append(result)

            # Be respectful - wait between requests
            time.sleep(2)

        return pd.DataFrame(results)

    def check_rate_limiting(self) -> Dict:
        """Check if screener.in has rate limiting"""
        print("\n" + "="*80)
        print("CHECKING RATE LIMITING")
        print("="*80 + "\n")

        test_ticker = "RELIANCE"
        url = f"{self.base_url}/company/{test_ticker}/"

        response_times = []
        statuses = []

        for i in range(5):
            start = time.time()
            try:
                response = self.session.get(url, timeout=10)
                elapsed = time.time() - start

                response_times.append(elapsed)
                statuses.append(response.status_code)

                print(f"Request {i+1}: Status {response.status_code}, Time {elapsed:.2f}s")

                if i < 4:  # Don't wait after last request
                    time.sleep(0.5)

            except Exception as e:
                print(f"Request {i+1}: ERROR - {str(e)}")
                statuses.append(0)

        print(f"\nAverage response time: {sum(response_times)/len(response_times):.2f}s")
        print(f"All successful: {all(s == 200 for s in statuses)}")

        return {
            'response_times': response_times,
            'statuses': statuses,
            'rate_limited': any(s == 429 for s in statuses)
        }


def main():
    """Run exploration"""
    explorer = ScreenerExplorer()

    # Test stocks from different sectors
    test_stocks = [
        'RELIANCE',    # Energy
        'TCS',         # IT
        'HDFCBANK',    # Banking
        'INFY',        # IT
        'ICICIBANK'    # Banking
    ]

    print("="*80)
    print("SCREENER.IN DATA EXPLORATION")
    print("="*80)
    print("\nTesting stocks:", ', '.join(test_stocks))
    print()

    # Explore each stock
    results = explorer.test_multiple_stocks(test_stocks)

    # Check rate limiting
    rate_info = explorer.check_rate_limiting()

    # Save results
    results.to_json('screener_exploration_results.json', indent=2)
    print(f"\n[OK] Results saved to: screener_exploration_results.json")

    # Summary
    print("\n" + "="*80)
    print("EXPLORATION SUMMARY")
    print("="*80)
    print(f"\nStocks explored: {len(results)}")
    print(f"Successful: {sum(results['status'] == 'success')}")
    print(f"Failed: {sum(results['status'] == 'error')}")

    if not rate_info['rate_limited']:
        print("\n[OK] No rate limiting detected (in small sample)")
        print("   Recommended: 1-2 seconds between requests")
    else:
        print("\n[WARNING] Rate limiting detected!")
        print("   Need to implement backoff strategy")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
    1. [OK] Screener.in has rich fundamental data
    2. [OK] Data is available in structured tables
    3. [OK] Multiple data categories available

    Next: Build full scraper to extract:
       - Quarterly results
       - Annual financials
       - Balance sheet
       - Cash flow
       - Ratios
       - Shareholding patterns
    """)


if __name__ == "__main__":
    main()
