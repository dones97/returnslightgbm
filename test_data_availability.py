"""
Test script to check data availability from yfinance for Indian stocks
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_stock_data(ticker):
    """Test what data is available for a given ticker"""
    print(f"\n{'='*60}")
    print(f"Testing ticker: {ticker}")
    print(f"{'='*60}")

    try:
        stock = yf.Ticker(ticker)

        # Test historical data
        print("\n1. Historical Price Data:")
        hist = stock.history(period="5y")
        if not hist.empty:
            print(f"   [OK] Available from {hist.index[0]} to {hist.index[-1]}")
            print(f"   Columns: {list(hist.columns)}")
        else:
            print("   [X] No historical data available")

        # Test info (fundamental data)
        print("\n2. Fundamental Data (info):")
        info = stock.info
        if info:
            print(f"   [OK] Available fields: {len(info)} fields")
            # Print important fields
            important_fields = [
                'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE',
                'priceToBook', 'profitMargins', 'returnOnEquity', 'returnOnAssets',
                'totalRevenue', 'revenueGrowth', 'grossMargins', 'operatingMargins',
                'debtToEquity', 'currentRatio', 'bookValue', 'priceToSalesTrailing12Months',
                'earningsGrowth', 'beta', 'dividendYield', 'payoutRatio'
            ]
            available = [f for f in important_fields if f in info and info[f] is not None]
            print(f"   Key metrics available: {len(available)}/{len(important_fields)}")
            print(f"   Available: {available[:10]}...")  # Show first 10
        else:
            print("   [X] No fundamental data available")

        # Test financials
        print("\n3. Financial Statements:")
        try:
            financials = stock.financials
            if not financials.empty:
                print(f"   [OK] Income Statement: {financials.shape}")
            else:
                print("   [X] No income statement")
        except:
            print("   [X] No income statement")

        try:
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                print(f"   [OK] Balance Sheet: {balance_sheet.shape}")
            else:
                print("   [X] No balance sheet")
        except:
            print("   [X] No balance sheet")

        try:
            cashflow = stock.cashflow
            if not cashflow.empty:
                print(f"   [OK] Cash Flow: {cashflow.shape}")
            else:
                print("   [X] No cash flow")
        except:
            print("   [X] No cash flow")

        # Test quarterly data
        print("\n4. Quarterly Data:")
        try:
            quarterly_financials = stock.quarterly_financials
            if not quarterly_financials.empty:
                print(f"   [OK] Quarterly Financials: {quarterly_financials.shape}")
            else:
                print("   [X] No quarterly financials")
        except:
            print("   [X] No quarterly financials")

        return True

    except Exception as e:
        print(f"   [X] Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Test a few representative stocks
    test_tickers = [
        "RELIANCE.NS",  # Large cap
        "TCS.NS",       # IT sector
        "HDFCBANK.NS",  # Banking
        "ITC.NS",       # Consumer
        "INFY.NS",      # IT
    ]

    print("Testing yfinance data availability for Indian stocks")
    print("This will help determine what features we can use for the model\n")

    for ticker in test_tickers:
        test_stock_data(ticker)

    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}")
