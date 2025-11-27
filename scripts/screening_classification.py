"""
Automated Screening - Classification Model (Screener.in + YFinance)

Screens all stocks using the trained classification model.
Predicts monthly return direction (UP/DOWN) with probability scores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_collector_simple_monthly import SimpleMonthlyCollector
from price_technical_indicators import TechnicalIndicators
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')


def screen_all_stocks():
    """Screen all stocks in database and save classification results"""

    print("\n" + "="*80)
    print("AUTOMATED SCREENING - CLASSIFICATION MODEL")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize
    fundamental_collector = SimpleMonthlyCollector(db_path='screener_data.db')
    technical_collector = TechnicalIndicators(use_cache=True)

    # Load trained model
    print("\n[1] Loading trained classification model...")
    try:
        with open('model_classification.pkl', 'rb') as f:
            model = pickle.load(f)
        print("    [OK] Model loaded successfully")
    except FileNotFoundError:
        print("[ERROR] Model file not found: model_classification.pkl")
        print("Please run model training first!")
        return

    # Get all stocks from database
    print("\n[2] Getting stocks from database...")
    all_stocks = fundamental_collector.get_available_stocks()
    print(f"    Total stocks available: {len(all_stocks)}")

    # Collect most recent data for all stocks
    print("\n[3] Collecting current month data for screening...")
    print("    Getting latest quarterly fundamentals...")

    # Get the most recent month-end date
    current_month_end = pd.Timestamp.now().replace(day=1) - pd.DateOffset(days=1)
    current_month_end_str = current_month_end.strftime('%Y-%m-%d')

    # Collect fundamental data for current month
    fundamental_data = fundamental_collector.prepare_training_data(
        tickers=all_stocks,
        start_date=current_month_end_str,
        end_date=current_month_end_str
    )

    if fundamental_data.empty:
        print("[ERROR] No fundamental data collected!")
        return

    print(f"    Stocks with valid fundamental data: {len(fundamental_data)}")

    # Add technical indicators
    print("\n[4] Adding technical indicators from yfinance...")
    all_data = []

    for i, ticker in enumerate(fundamental_data['ticker'].unique(), 1):
        ticker_data = fundamental_data[fundamental_data['ticker'] == ticker].copy()

        if ticker_data.empty:
            continue

        # Get technical indicators for this stock
        month_dates = pd.Series(ticker_data['month_end_date'].values)

        try:
            # Calculate all technical indicators
            tech_indicators = technical_collector.calculate_indicators(ticker, month_dates)

            # Add technical features to fundamental data
            for col in tech_indicators.columns:
                ticker_data[col] = tech_indicators[col].values

            all_data.append(ticker_data)

            if i % 50 == 0:
                print(f"    Processed {i}/{fundamental_data['ticker'].nunique()} stocks...")

        except Exception as e:
            continue

    if not all_data:
        print("[ERROR] No data with technical indicators!")
        return

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"    Stocks with complete data: {len(combined_data)}")

    # Prepare features
    print("\n[5] Preparing features for prediction...")
    exclude_cols = ['ticker', 'month_end_date', 'quarter_date', 'staleness_days']

    feature_cols = [c for c in combined_data.columns if c not in exclude_cols]
    X = combined_data[feature_cols].copy()

    # Fill NaN values with median
    X = X.fillna(X.median())

    # Make predictions
    print("\n[6] Making predictions...")
    predictions_proba = model.predict(X)  # Probability of UP (1)
    predictions_direction = (predictions_proba > 0.5).astype(int)  # 1=UP, 0=DOWN

    # Create results dataframe
    results = pd.DataFrame({
        'Ticker': combined_data['ticker'].values,
        'Predicted_Direction': predictions_direction,
        'Predicted_Probability': predictions_proba,
        'Quarter_Date': combined_data['quarter_date'].values,
        'Staleness_Days': combined_data['staleness_days'].values,
    })

    # Add fundamental metrics for context
    fundamental_cols = [
        'sales_growth_yoy', 'profit_growth_yoy', 'opm_percent',
        'profit_margin', 'debt_to_equity', 'roce', 'roe'
    ]
    for col in fundamental_cols:
        if col in combined_data.columns:
            results[col] = combined_data[col].values

    # Add technical metrics for context
    technical_cols = ['roc_1m', 'roc_3m', 'volatility_30d', 'rsi_14d']
    for col in technical_cols:
        if col in combined_data.columns:
            results[col] = combined_data[col].values

    # Add .NS suffix for consistency
    results['Ticker'] = results['Ticker'] + '.NS'

    # Sort by predicted probability (highest first)
    results = results.sort_values('Predicted_Probability', ascending=False)

    # Add rank
    results['Rank'] = range(1, len(results) + 1)

    # Categorize predictions
    results['Category'] = 'Neutral'
    results.loc[results['Predicted_Probability'] >= 0.70, 'Category'] = 'Strong Buy'
    results.loc[(results['Predicted_Probability'] >= 0.55) & (results['Predicted_Probability'] < 0.70), 'Category'] = 'Buy'
    results.loc[(results['Predicted_Probability'] < 0.45) & (results['Predicted_Probability'] >= 0.30), 'Category'] = 'Sell'
    results.loc[results['Predicted_Probability'] < 0.30, 'Category'] = 'Strong Sell'

    # Add confidence level
    results['Confidence'] = abs(results['Predicted_Probability'] - 0.5) * 2  # 0 to 1 scale

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as pickle (for Streamlit)
    results.to_pickle('screening_results_classification.pkl')

    # Save as CSV
    csv_filename = f'screening_results_classification_{timestamp}.csv'
    results.to_csv(csv_filename, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("SCREENING COMPLETE")
    print(f"{'='*80}")

    print(f"\n[RESULTS]")
    print(f"  Total stocks analyzed: {len(results)}")
    print(f"  Strong Buy (prob ≥70%): {sum(results['Category'] == 'Strong Buy')}")
    print(f"  Buy (prob 55-70%): {sum(results['Category'] == 'Buy')}")
    print(f"  Neutral (prob 45-55%): {sum(results['Category'] == 'Neutral')}")
    print(f"  Sell (prob 30-45%): {sum(results['Category'] == 'Sell')}")
    print(f"  Strong Sell (prob <30%): {sum(results['Category'] == 'Strong Sell')}")

    print(f"\n[STATISTICS]")
    print(f"  Mean probability: {results['Predicted_Probability'].mean():.2%}")
    print(f"  Std probability: {results['Predicted_Probability'].std():.2%}")
    print(f"  Max probability: {results['Predicted_Probability'].max():.2%}")
    print(f"  Min probability: {results['Predicted_Probability'].min():.2%}")
    print(f"  Predicted UP: {sum(results['Predicted_Direction'] == 1)} ({sum(results['Predicted_Direction'] == 1)/len(results)*100:.1f}%)")
    print(f"  Predicted DOWN: {sum(results['Predicted_Direction'] == 0)} ({sum(results['Predicted_Direction'] == 0)/len(results)*100:.1f}%)")

    print(f"\n[TOP 10 PREDICTIONS]")
    for idx, row in results.head(10).iterrows():
        direction = "UP" if row['Predicted_Direction'] == 1 else "DOWN"
        print(f"  {row['Rank']:3d}. {row['Ticker']:15s} {direction:4s} "
              f"(Prob: {row['Predicted_Probability']:.1%}, "
              f"Conf: {row['Confidence']:.1%}, "
              f"Sales Gr: {row.get('sales_growth_yoy', np.nan):+.1f}%)")

    print(f"\n[FILES SAVED]")
    print(f"  Pickle: screening_results_classification.pkl")
    print(f"  CSV: {csv_filename}")

    return results


if __name__ == "__main__":
    screen_all_stocks()
