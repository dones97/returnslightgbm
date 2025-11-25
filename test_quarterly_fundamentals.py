"""
Test what historical fundamental data is available from yfinance
"""
import yfinance as yf
import pandas as pd

def test_fundamental_history(ticker):
    """Check what historical fundamental data is available"""
    print(f"\n{'='*80}")
    print(f"Testing: {ticker}")
    print(f"{'='*80}")

    stock = yf.Ticker(ticker)

    # 1. Check quarterly financials
    print("\n1. QUARTERLY FINANCIALS (Income Statement):")
    try:
        quarterly_financials = stock.quarterly_financials
        if not quarterly_financials.empty:
            print(f"   Available: YES")
            print(f"   Quarters: {len(quarterly_financials.columns)}")
            print(f"   Date range: {quarterly_financials.columns[-1].strftime('%Y-%m-%d')} to {quarterly_financials.columns[0].strftime('%Y-%m-%d')}")
            print(f"   Rows (metrics): {len(quarterly_financials)}")
            print(f"\n   Sample metrics available:")
            for idx, metric in enumerate(quarterly_financials.index[:10]):
                print(f"      - {metric}")
        else:
            print(f"   Available: NO")
    except Exception as e:
        print(f"   Error: {str(e)}")

    # 2. Check quarterly balance sheet
    print("\n2. QUARTERLY BALANCE SHEET:")
    try:
        quarterly_balance = stock.quarterly_balance_sheet
        if not quarterly_balance.empty:
            print(f"   Available: YES")
            print(f"   Quarters: {len(quarterly_balance.columns)}")
            print(f"   Date range: {quarterly_balance.columns[-1].strftime('%Y-%m-%d')} to {quarterly_balance.columns[0].strftime('%Y-%m-%d')}")
        else:
            print(f"   Available: NO")
    except Exception as e:
        print(f"   Error: {str(e)}")

    # 3. Check quarterly cashflow
    print("\n3. QUARTERLY CASHFLOW:")
    try:
        quarterly_cashflow = stock.quarterly_cashflow
        if not quarterly_cashflow.empty:
            print(f"   Available: YES")
            print(f"   Quarters: {len(quarterly_cashflow.columns)}")
            print(f"   Date range: {quarterly_cashflow.columns[-1].strftime('%Y-%m-%d')} to {quarterly_cashflow.columns[0].strftime('%Y-%m-%d')}")
        else:
            print(f"   Available: NO")
    except Exception as e:
        print(f"   Error: {str(e)}")

    # 4. Check if we can derive historical P/E, P/B, etc.
    print("\n4. DERIVED RATIOS (can we calculate historical P/E, P/B?):")
    try:
        # Get historical prices
        hist = stock.history(period="5y")

        # Get quarterly earnings
        if not quarterly_financials.empty and 'Net Income' in quarterly_financials.index:
            print(f"   Net Income: Available for {len(quarterly_financials.columns)} quarters")
            print(f"   Historical Price: Available for {len(hist)} days")
            print(f"   CAN CALCULATE: Historical P/E ratio (Price / EPS)")

        # Get book value
        if not quarterly_balance.empty and 'Stockholders Equity' in quarterly_balance.index:
            print(f"   Book Value: Available for {len(quarterly_balance.columns)} quarters")
            print(f"   CAN CALCULATE: Historical P/B ratio")

    except Exception as e:
        print(f"   Error: {str(e)}")

    # 5. What can we actually use month-by-month?
    print("\n5. WHAT WE CAN RECONSTRUCT MONTHLY:")
    try:
        if not quarterly_financials.empty:
            # Example: Total Revenue
            if 'Total Revenue' in quarterly_financials.index:
                revenue = quarterly_financials.loc['Total Revenue']
                print(f"   - Quarterly Revenue (forward-fill to monthly)")
                print(f"   - Revenue Growth (YoY, QoQ)")

            if 'Net Income' in quarterly_financials.index:
                print(f"   - Quarterly Net Income (forward-fill to monthly)")
                print(f"   - Profit Margins")

        if not quarterly_balance.empty:
            if 'Total Debt' in quarterly_balance.index:
                print(f"   - Total Debt (forward-fill to monthly)")
            if 'Total Assets' in quarterly_balance.index:
                print(f"   - Total Assets (forward-fill to monthly)")

        print(f"\n   NOTE: Fundamentals update quarterly, so we'd forward-fill to monthly")

    except Exception as e:
        print(f"   Error: {str(e)}")

# Test with a few stocks
test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

print("="*80)
print("TESTING HISTORICAL FUNDAMENTAL DATA AVAILABILITY")
print("="*80)

for ticker in test_stocks:
    test_fundamental_history(ticker)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
KEY FINDINGS:

1. yfinance DOES provide historical financial statements:
   - Quarterly Income Statement (usually 4-8 quarters)
   - Quarterly Balance Sheet (usually 4-8 quarters)
   - Quarterly Cash Flow (usually 4-8 quarters)

2. We CAN calculate historical:
   - Revenue, Net Income, EBITDA
   - Total Debt, Book Value, Assets
   - Operating Cash Flow
   - Derived ratios: P/E, P/B, ROE, Debt/Equity

3. LIMITATION:
   - Only ~2 years of quarterly data (8 quarters)
   - NOT 10 years of historical fundamentals
   - Fundamentals update quarterly (not monthly)

4. SOLUTION APPROACH:
   - Use quarterly fundamentals (2-3 years available)
   - Forward-fill to monthly frequency
   - Combine with 10 years of price/technical data
   - Model will use "most recent fundamental + price history"
""")
