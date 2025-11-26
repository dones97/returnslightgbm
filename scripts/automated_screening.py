"""
Automated Stock Screening Script

This script can be run standalone or via GitHub Actions for automated screening.
Uses the pre-trained model from the repository to screen all NSE stocks.
"""

import os
import sys
import pandas as pd
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_trainer import ReturnDirectionModel, StockScorer
from data_collector_enhanced import collect_data_for_universe


def load_universe_tickers():
    """Load all tickers from NSE universe"""
    try:
        df = pd.read_csv('NSE_Universe.csv')
        tickers = df['Ticker'].tolist()
        # Add .NS suffix if not present
        tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]
        return tickers
    except Exception as e:
        print(f"Error loading universe: {e}")
        return []


def run_automated_screening():
    """
    Run automated stock screening workflow

    Steps:
    1. Load pre-trained model
    2. Collect current data for all stocks
    3. Score all stocks
    4. Save results as CSV and pickle
    """

    print("="*80)
    print("AUTOMATED STOCK SCREENING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Load trained model
    print("\n[1/4] Loading pre-trained model...")
    model_file = 'trained_model.pkl'

    if not os.path.exists(model_file):
        print(f"❌ ERROR: Model file '{model_file}' not found!")
        print("Please train a model first or ensure trained_model.pkl is in the repository.")
        return False

    model = ReturnDirectionModel()
    model.load_model(model_file)
    print(f"✅ Model loaded successfully")

    if model.model_metadata:
        trained_date = model.model_metadata.get('trained_at', 'Unknown')
        n_stocks = model.model_metadata.get('n_stocks', 'Unknown')
        print(f"   - Trained on: {trained_date}")
        print(f"   - Training stocks: {n_stocks}")

    # Step 2: Load screening data (use cached if available)
    print("\n[2/4] Loading screening data...")
    screening_cache = 'screening_data_cache.pkl'

    if os.path.exists(screening_cache):
        print(f"✅ Using cached screening data: {screening_cache}")
        with open(screening_cache, 'rb') as f:
            screening_data = pickle.load(f)
        print(f"   - Loaded data for {screening_data['Ticker'].nunique()} stocks")
    else:
        print("⚠️  No cached screening data found. Collecting fresh data...")
        print("   This may take 30-60 minutes and could hit rate limits.")

        # Load all tickers
        all_tickers = load_universe_tickers()
        if not all_tickers:
            print("❌ ERROR: Failed to load stock tickers!")
            return False

        print(f"   - Found {len(all_tickers)} tickers in universe")

        # Collect current data (6 months)
        print("   - Collecting current fundamental data (this will take a while)...")
        screening_data = collect_data_for_universe(all_tickers, lookback_years=0.5)

        if screening_data is None or len(screening_data) == 0:
            print("❌ ERROR: Failed to collect screening data!")
            return False

        # Save for future use
        with open(screening_cache, 'wb') as f:
            pickle.dump(screening_data, f)
        print(f"✅ Collected and cached data for {screening_data['Ticker'].nunique()} stocks")

    # Step 3: Score all stocks
    print("\n[3/4] Scoring all stocks with trained model...")
    scorer = StockScorer(model)
    results = scorer.score_current_universe(screening_data)
    print(f"✅ Scored {len(results)} stocks")

    # Step 4: Save results
    print("\n[4/4] Saving results...")

    # Save as pickle
    results_pkl = 'screening_results.pkl'
    with open(results_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved pickle: {results_pkl}")

    # Save as CSV with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    results_csv = f'screening_results_{timestamp}.csv'
    results.to_csv(results_csv, index=False)
    print(f"✅ Saved CSV: {results_csv}")

    # Print summary
    print("\n" + "="*80)
    print("SCREENING SUMMARY")
    print("="*80)
    print(f"Total stocks analyzed: {len(results)}")
    print(f"\nQuintile Distribution:")
    quintile_counts = results['quality_quintile'].value_counts().sort_index(ascending=False)
    for quintile, count in quintile_counts.items():
        print(f"  {quintile}: {count} stocks")

    # Top 10 by composite score
    print(f"\nTop 10 Stocks by Composite Score:")
    top_10 = results.nlargest(10, 'composite_score')[
        ['Ticker', 'quality_quintile', 'composite_score', 'predicted_probability']
    ]
    for idx, row in top_10.iterrows():
        ticker = row['Ticker'].replace('.NS', '')
        quintile = row['quality_quintile']
        score = row['composite_score']
        prob = row['predicted_probability']
        print(f"  {ticker:<15} {quintile:<15} Score: {score:>6.1f}  Prob: {prob:>6.1%}")

    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return True


if __name__ == "__main__":
    success = run_automated_screening()
    sys.exit(0 if success else 1)
