"""
Check 10-year data availability for Indian stocks
Tests a sample of stocks to determine how many have sufficient historical data
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def check_data_availability(ticker, years=10):
    """Check if stock has required data for given years"""
    try:
        stock = yf.Ticker(ticker)
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch historical data
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {
                'ticker': ticker,
                'has_data': False,
                'months_available': 0,
                'start_date': None,
                'end_date': None,
                'has_fundamentals': False,
                'fundamental_count': 0
            }

        # Calculate months of data
        date_range = (hist.index[-1] - hist.index[0]).days / 30.44

        # Check fundamentals
        info = stock.info
        important_fundamentals = [
            'marketCap', 'trailingPE', 'priceToBook', 'returnOnEquity',
            'debtToEquity', 'profitMargins', 'revenueGrowth'
        ]
        fundamental_count = sum(1 for f in important_fundamentals if f in info and info[f] is not None)

        return {
            'ticker': ticker,
            'has_data': True,
            'months_available': int(date_range),
            'start_date': hist.index[0],
            'end_date': hist.index[-1],
            'has_fundamentals': fundamental_count >= 3,
            'fundamental_count': fundamental_count,
            'records': len(hist)
        }

    except Exception as e:
        return {
            'ticker': ticker,
            'has_data': False,
            'months_available': 0,
            'start_date': None,
            'end_date': None,
            'has_fundamentals': False,
            'fundamental_count': 0,
            'error': str(e)
        }


def main():
    print("="*80)
    print("CHECKING 10-YEAR DATA AVAILABILITY FOR INDIAN STOCKS")
    print("="*80)
    print()

    # Load NSE Universe
    try:
        df = pd.read_csv('NSE_Universe.csv')
        all_tickers = df['Ticker'].dropna().unique().tolist()
        all_tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in all_tickers]
        print(f"Loaded {len(all_tickers)} tickers from NSE_Universe.csv")
    except:
        print("Could not load NSE_Universe.csv, using sample tickers...")
        all_tickers = []

    # Also try backup file
    if not all_tickers:
        try:
            df = pd.read_csv('indian_stocks_tickers.csv', encoding='utf-8-sig')
            backup_tickers = df['Ticker'].dropna().unique().tolist()
            all_tickers.extend(backup_tickers)
            print(f"Loaded {len(all_tickers)} tickers from indian_stocks_tickers.csv")
        except:
            pass

    # If still no tickers, use a comprehensive sample
    if not all_tickers:
        print("Using predefined sample of major stocks...")
        all_tickers = [
            # Large caps
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',

            # Mid caps
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
            'ULTRACEMCO.NS', 'NESTLEIND.NS', 'BAJFINANCE.NS', 'SUNPHARMA.NS', 'WIPRO.NS',

            # More large/mid caps
            'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 'TATAMOTORS.NS',
            'HCLTECH.NS', 'TECHM.NS', 'INDUSINDBK.NS', 'ADANIPORTS.NS', 'DRREDDY.NS',

            # Additional stocks
            'CIPLA.NS', 'DABUR.NS', 'DIVISLAB.NS', 'EICHERMOT.NS', 'GRASIM.NS',
            'HEROMOTOCO.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'M&M.NS', 'BRITANNIA.NS'
        ]

    # Test with a sample (50 stocks for quick check, then we can extrapolate)
    import random
    random.seed(42)
    sample_size = min(50, len(all_tickers))
    test_tickers = random.sample(all_tickers, sample_size) if len(all_tickers) > sample_size else all_tickers

    print(f"\nTesting {len(test_tickers)} stocks for 10-year data availability...")
    print("This may take a few minutes...\n")

    results = []
    for i, ticker in enumerate(test_tickers):
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_tickers)} stocks...")

        result = check_data_availability(ticker, years=10)
        results.append(result)

    # Analyze results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    total_stocks = len(results_df)
    has_data = results_df['has_data'].sum()
    has_10y = (results_df['months_available'] >= 120).sum()
    has_8y = (results_df['months_available'] >= 96).sum()
    has_5y = (results_df['months_available'] >= 60).sum()
    has_fundamentals = results_df['has_fundamentals'].sum()

    print(f"\nTotal stocks tested: {total_stocks}")
    print(f"\nStocks with any data: {has_data} ({has_data/total_stocks*100:.1f}%)")
    print(f"Stocks with 10+ years: {has_10y} ({has_10y/total_stocks*100:.1f}%)")
    print(f"Stocks with 8+ years: {has_8y} ({has_8y/total_stocks*100:.1f}%)")
    print(f"Stocks with 5+ years: {has_5y} ({has_5y/total_stocks*100:.1f}%)")
    print(f"Stocks with fundamentals: {has_fundamentals} ({has_fundamentals/total_stocks*100:.1f}%)")

    # Show distribution
    print("\n" + "-"*80)
    print("DATA AVAILABILITY DISTRIBUTION")
    print("-"*80)

    bins = [0, 24, 60, 96, 120, 180]
    labels = ['<2 years', '2-5 years', '5-8 years', '8-10 years', '10+ years']
    results_df['data_range'] = pd.cut(results_df['months_available'], bins=bins, labels=labels, include_lowest=True)

    distribution = results_df['data_range'].value_counts().sort_index()
    print("\nMonths of data available:")
    for label, count in distribution.items():
        pct = count / total_stocks * 100
        print(f"  {label:15s}: {count:3d} stocks ({pct:5.1f}%)")

    # Stocks with 10+ years
    print("\n" + "-"*80)
    print("STOCKS WITH 10+ YEARS OF DATA")
    print("-"*80)

    good_stocks = results_df[results_df['months_available'] >= 120].copy()
    good_stocks = good_stocks.sort_values('months_available', ascending=False)

    if len(good_stocks) > 0:
        print(f"\nFound {len(good_stocks)} stocks with 10+ years of data:\n")
        for idx, row in good_stocks.head(20).iterrows():
            ticker_clean = row['ticker'].replace('.NS', '')
            years = row['months_available'] / 12
            fundamentals = "Yes" if row['has_fundamentals'] else "No"
            print(f"  {ticker_clean:15s}: {years:.1f} years | Fundamentals: {fundamentals} | Records: {row['records']}")

        if len(good_stocks) > 20:
            print(f"\n  ... and {len(good_stocks) - 20} more stocks")
    else:
        print("\nNo stocks found with 10+ years of data in this sample.")

    # Extrapolate to full universe
    print("\n" + "="*80)
    print("EXTRAPOLATION TO FULL UNIVERSE")
    print("="*80)

    if len(all_tickers) > sample_size:
        pct_10y = has_10y / total_stocks
        pct_8y = has_8y / total_stocks
        pct_5y = has_5y / total_stocks

        estimated_10y = int(len(all_tickers) * pct_10y)
        estimated_8y = int(len(all_tickers) * pct_8y)
        estimated_5y = int(len(all_tickers) * pct_5y)

        print(f"\nTotal stocks in universe: {len(all_tickers)}")
        print(f"\nEstimated stocks with:")
        print(f"  10+ years of data: ~{estimated_10y} stocks")
        print(f"  8+ years of data: ~{estimated_8y} stocks")
        print(f"  5+ years of data: ~{estimated_5y} stocks")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if has_10y >= 30:
        print(f"\n[OK] GOOD NEWS: {has_10y} stocks in sample have 10+ years of data!")
        print(f"  This is sufficient for training a robust model.")
        print(f"\n  Recommended approach:")
        print(f"  - Use 10-year lookback for stocks that have the data")
        print(f"  - This will give you ~120 monthly observations per stock")
        print(f"  - Total training records: {has_10y} stocks × 120 months = ~{has_10y * 120:,} records")
    elif has_8y >= 30:
        print(f"\n[!] MODERATE: Only {has_10y} stocks have 10 years, but {has_8y} have 8+ years")
        print(f"\n  Recommended approach:")
        print(f"  - Use 8-year lookback (96 months)")
        print(f"  - This gives you {has_8y} stocks with sufficient data")
        print(f"  - Total training records: {has_8y} stocks × 96 months = ~{has_8y * 96:,} records")
    else:
        print(f"\n[!] LIMITED: Only {has_10y} stocks have 10 years, {has_8y} have 8 years")
        print(f"\n  Recommended approach:")
        print(f"  - Use 5-year lookback (60 months)")
        print(f"  - This gives you {has_5y} stocks with sufficient data")
        print(f"  - Total training records: {has_5y} stocks × 60 months = ~{has_5y * 60:,} records")

    print("\n  Note: More data is generally better for ML models, but quality matters too.")
    print("  Consider filtering for:")
    print("  - Liquid stocks (high average volume)")
    print("  - Stocks with complete fundamental data")
    print("  - Actively traded stocks (no long suspensions)")

    # Save detailed results
    results_df.to_csv('data_availability_check.csv', index=False)
    print(f"\n  Detailed results saved to: data_availability_check.csv")

    print("\n" + "="*80)
    print("CHECK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
