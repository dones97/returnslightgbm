"""
Automated Screening - Screener.in Database Version

Screens all stocks using trained model and screener.in database.
Predicts next quarter returns for all available stocks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_collector_screener import ScreenerDataCollector
from model_trainer_screener import ScreenerModelTrainer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def screen_all_stocks():
    """Screen all stocks in database and save results"""

    print("\n" + "="*80)
    print("AUTOMATED SCREENING - SCREENER.IN DATABASE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize
    collector = ScreenerDataCollector(db_path='screener_data.db')
    trainer = ScreenerModelTrainer(db_path='screener_data.db')

    # Load trained model
    print("\n[1] Loading trained model...")
    try:
        trainer.load_model('trained_model_screener.pkl')
    except FileNotFoundError:
        print("[ERROR] Model file not found: trained_model_screener.pkl")
        print("Please run model training first!")
        return

    # Get all stocks
    print("\n[2] Getting stocks from database...")
    all_stocks = collector.get_available_stocks()
    print(f"    Total stocks available: {len(all_stocks)}")

    # Collect current quarter data
    print("\n[3] Collecting current quarter data...")
    screening_data = collector.get_current_data_for_screening(all_stocks)

    if screening_data.empty:
        print("[ERROR] No screening data collected!")
        return

    print(f"    Stocks with valid data: {len(screening_data)}")

    # Prepare features
    print("\n[4] Preparing features for prediction...")
    exclude_cols = ['ticker', 'quarter_date', 'id', 'scraped_at']

    # Ensure we have all required features
    required_features = trainer.feature_names
    available_features = [col for col in screening_data.columns if col not in exclude_cols]

    # Check for missing features
    missing_features = set(required_features) - set(available_features)
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
        # Add missing features as NaN
        for feat in missing_features:
            screening_data[feat] = np.nan

    # Get only required features in correct order
    X = screening_data[required_features].copy()

    # Fill NaN values with 0 (or median - you can improve this)
    X = X.fillna(0)

    # Make predictions
    print("\n[5] Making predictions...")
    predictions = trainer.predict(X)

    # Create results dataframe
    results = pd.DataFrame({
        'Ticker': screening_data['ticker'].values,
        'Predicted_Return': predictions,
        'Quarter_Date': screening_data['quarter_date'].values,
        'Sales': screening_data['sales'].values,
        'Net_Profit': screening_data['net_profit'].values,
        'Sales_Growth_YoY': screening_data['sales_growth_yoy'].values,
        'Profit_Growth_YoY': screening_data['profit_growth_yoy'].values,
        'OPM_Percent': screening_data['opm_percent'].values,
        'Profit_Margin': screening_data['profit_margin'].values,
        'Quality_Score': screening_data['quality_score'].values,
        'Sales_Trend': screening_data['sales_trend'].values,
        'Profit_Trend': screening_data['profit_trend'].values
    })

    # Add .NS suffix for consistency
    results['Ticker'] = results['Ticker'] + '.NS'

    # Sort by predicted return
    results = results.sort_values('Predicted_Return', ascending=False)

    # Add rank
    results['Rank'] = range(1, len(results) + 1)

    # Categorize predictions
    results['Category'] = 'Neutral'
    results.loc[results['Predicted_Return'] >= 5, 'Category'] = 'Strong Buy'
    results.loc[(results['Predicted_Return'] >= 2) & (results['Predicted_Return'] < 5), 'Category'] = 'Buy'
    results.loc[(results['Predicted_Return'] <= -2) & (results['Predicted_Return'] > -5), 'Category'] = 'Sell'
    results.loc[results['Predicted_Return'] <= -5, 'Category'] = 'Strong Sell'

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as pickle (for Streamlit)
    results.to_pickle('screening_results_screener.pkl')

    # Save as CSV
    csv_filename = f'screening_results_screener_{timestamp}.csv'
    results.to_csv(csv_filename, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("SCREENING COMPLETE")
    print(f"{'='*80}")

    print(f"\n[RESULTS]")
    print(f"  Total stocks analyzed: {len(results)}")
    print(f"  Strong Buy (≥5%): {sum(results['Category'] == 'Strong Buy')}")
    print(f"  Buy (2-5%): {sum(results['Category'] == 'Buy')}")
    print(f"  Neutral (-2 to 2%): {sum(results['Category'] == 'Neutral')}")
    print(f"  Sell (-5 to -2%): {sum(results['Category'] == 'Sell')}")
    print(f"  Strong Sell (≤-5%): {sum(results['Category'] == 'Strong Sell')}")

    print(f"\n[STATISTICS]")
    print(f"  Mean predicted return: {results['Predicted_Return'].mean():.2f}%")
    print(f"  Std predicted return: {results['Predicted_Return'].std():.2f}%")
    print(f"  Max predicted return: {results['Predicted_Return'].max():.2f}%")
    print(f"  Min predicted return: {results['Predicted_Return'].min():.2f}%")

    print(f"\n[TOP 10 PREDICTIONS]")
    for idx, row in results.head(10).iterrows():
        print(f"  {row['Rank']:3d}. {row['Ticker']:15s} {row['Predicted_Return']:+7.2f}% "
              f"(Quality: {row['Quality_Score']:.0f}, Sales Growth: {row['Sales_Growth_YoY']:+.1f}%)")

    print(f"\n[FILES SAVED]")
    print(f"  Pickle: screening_results_screener.pkl")
    print(f"  CSV: {csv_filename}")

    return results


if __name__ == "__main__":
    screen_all_stocks()
