"""
Annual Classification Screening Script

Uses the screener annual classification model (57.1% accuracy)
to predict 1-year return direction for all stocks.

Leverages existing data_collector_annual.py infrastructure.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collector_annual import AnnualDataCollector
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def screen_all_stocks():
    """Screen all stocks using annual classification model"""

    print("\n" + "#" * 80)
    print("ANNUAL CLASSIFICATION SCREENING")
    print("Predicting 1-Year Return Direction")
    print("#" * 80)

    # Load model
    print("\n[1] Loading model...")
    with open('model_annual_classification.pkl', 'rb') as f:
        model = pickle.load(f)
    print("    Model loaded successfully")
    print(f"    Features: {len(model.feature_name())}")

    # Initialize collector
    collector = AnnualDataCollector(db_path='screener_data.db')

    # Get all stocks with sufficient data
    print("\n[2] Getting stocks with annual data...")
    all_stocks = collector.get_stocks_with_annual_data(min_years=3)
    print(f"    Found {len(all_stocks)} stocks")

    # Collect current year ratios for all stocks
    print("\n[3] Collecting latest annual fundamentals...")

    all_data = []

    for i, ticker in enumerate(all_stocks, 1):
        if i % 100 == 0:
            print(f"    Processed {i}/{len(all_stocks)} stocks...")

        # Get available years for this stock
        import sqlite3
        conn = sqlite3.connect('screener_data.db')
        query = f"SELECT DISTINCT year FROM balance_sheet WHERE ticker = '{ticker}' ORDER BY year DESC LIMIT 1"
        result = pd.read_sql_query(query, conn)
        conn.close()

        if len(result) == 0:
            continue

        latest_year = result.iloc[0]['year']

        # Get ratios for latest year
        ratios = collector.calculate_annual_ratios(ticker, latest_year)

        if ratios and not all(pd.isna(v) for v in ratios.values()):
            ratios['ticker'] = ticker
            ratios['year'] = latest_year
            all_data.append(ratios)

    print(f"\n    Collected data for {len(all_data)} stocks")

    fundamental_df = pd.DataFrame(all_data)

    # Add technical indicators
    print("\n[4] Adding technical indicators...")

    end_date = datetime.now()
    start_date_str = f"{end_date.year - 2}-01-01"

    # Fetch prices
    print("    Fetching price data...")
    price_cache = {}
    tickers_to_fetch = fundamental_df['ticker'].unique().tolist()

    for i, ticker in enumerate(tickers_to_fetch, 1):
        if i % 50 == 0:
            print(f"      Fetched {i}/{len(tickers_to_fetch)} stocks...")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date_str, end=end_date)
            if len(hist) >= 200:  # Need at least 200 days
                price_cache[ticker] = hist
        except:
            continue

    print(f"    Got prices for {len(price_cache)} stocks")

    # Calculate technical indicators
    print("    Calculating technical indicators...")

    rows_with_tech = []

    for idx, row in fundamental_df.iterrows():
        ticker = row['ticker']

        if ticker not in price_cache:
            continue

        prices = price_cache[ticker]
        close = prices['Close']

        if len(close) < 200:
            continue

        try:
            # ROC
            roc_1y = ((close.iloc[-1] / close.iloc[-252]) - 1) * 100 if len(close) >= 252 else np.nan
            roc_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) >= 126 else np.nan

            # Moving averages
            ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else np.nan
            ma_50 = close.rolling(window=50).mean().iloc[-1]
            price_to_ma200 = (close.iloc[-1] / ma_200 - 1) * 100 if not np.isnan(ma_200) else np.nan
            price_to_ma50 = (close.iloc[-1] / ma_50 - 1) * 100

            # Volatility
            returns = close.pct_change()
            volatility_1y = returns.std() * np.sqrt(252) * 100

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_14 = 100 - (100 / (1 + rs.iloc[-1]))

            # Add to row
            row_dict = row.to_dict()
            row_dict.update({
                'roc_1y': roc_1y,
                'roc_6m': roc_6m,
                'price_to_ma200': price_to_ma200,
                'price_to_ma50': price_to_ma50,
                'volatility_1y': volatility_1y,
                'rsi_14': rsi_14
            })

            rows_with_tech.append(row_dict)
        except:
            continue

    complete_data = pd.DataFrame(rows_with_tech)
    print(f"    Complete data for {len(complete_data)} stocks")

    # Prepare features for prediction
    print("\n[5] Making predictions...")

    feature_columns = model.feature_name()

    # Ensure all features exist
    X = complete_data[feature_columns].copy()

    # Handle missing values
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)

    # Predict
    predictions_proba = model.predict(X)
    predictions_direction = (predictions_proba > 0.5).astype(int)

    # Create results
    results = complete_data[['ticker', 'year']].copy()
    results['Predicted_Probability'] = predictions_proba
    results['Predicted_Direction'] = predictions_direction
    results['Confidence'] = abs(predictions_proba - 0.5) * 2

    # Add all features for display
    for col in complete_data.columns:
        if col not in results.columns and col not in ['ticker', 'year']:
            results[col] = complete_data[col]

    # Categorize
    results['Category'] = 'Neutral'
    results.loc[results['Predicted_Probability'] >= 0.65, 'Category'] = 'Strong Buy'
    results.loc[(results['Predicted_Probability'] >= 0.55) & (results['Predicted_Probability'] < 0.65), 'Category'] = 'Buy'
    results.loc[(results['Predicted_Probability'] < 0.45) & (results['Predicted_Probability'] >= 0.35), 'Category'] = 'Sell'
    results.loc[results['Predicted_Probability'] < 0.35, 'Category'] = 'Strong Sell'

    # Rename ticker to Ticker
    results = results.rename(columns={'ticker': 'Ticker'})

    # Sort by probability
    results = results.sort_values('Predicted_Probability', ascending=False).reset_index(drop=True)
    results['Rank'] = range(1, len(results) + 1)

    # Display summary
    print("\n" + "=" * 80)
    print("SCREENING COMPLETE")
    print("=" * 80)
    print(f"\nTotal stocks analyzed: {len(results)}")
    print(f"Strong Buy (prob ≥65%): {sum(results['Category'] == 'Strong Buy')}")
    print(f"Buy (prob 55-65%): {sum(results['Category'] == 'Buy')}")
    print(f"Neutral (prob 45-55%): {sum(results['Category'] == 'Neutral')}")
    print(f"Sell (prob 35-45%): {sum(results['Category'] == 'Sell')}")
    print(f"Strong Sell (prob <35%): {sum(results['Category'] == 'Strong Sell')}")

    print(f"\nMean probability: {results['Predicted_Probability'].mean():.1%}")
    print(f"Predicted UP (next year): {sum(results['Predicted_Direction'] == 1)} ({sum(results['Predicted_Direction'] == 1)/len(results)*100:.1f}%)")

    # Show top 10
    print("\n" + "=" * 80)
    print("TOP 10 PREDICTIONS (1-Year Horizon)")
    print("=" * 80)

    top_10 = results.head(10)
    for i, row in top_10.iterrows():
        direction = "UP" if row['Predicted_Direction'] == 1 else "DOWN"
        roce_val = row.get('roce', 0)
        roce_val = roce_val if pd.notna(roce_val) else 0
        eps_growth_val = row.get('eps_growth', 0)
        eps_growth_val = eps_growth_val if pd.notna(eps_growth_val) else 0
        print(f"{row['Rank']:3d}. {row['Ticker']:15s} {direction:5s} (Prob: {row['Predicted_Probability']:.1%}, "
              f"ROCE: {roce_val:.1f}%, EPS Growth: {eps_growth_val:+.1f}%)")

    # Save results
    print("\n[6] Saving results...")

    # Save as pickle for Streamlit
    with open('screening_results_annual.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("    Saved: screening_results_annual.pkl")

    # Save as CSV
    timestamp = datetime.now().strftime('%Y%m%d')
    csv_filename = f'screening_results_annual_{timestamp}.csv'
    results.to_csv(csv_filename, index=False)
    print(f"    Saved: {csv_filename}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = screen_all_stocks()
