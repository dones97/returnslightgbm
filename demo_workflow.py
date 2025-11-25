"""
Demo Workflow - Test the entire pipeline with a small sample
This script demonstrates the complete workflow without the Streamlit UI
"""
import pandas as pd
import numpy as np
from data_collector import StockDataCollector, collect_data_for_universe
from model_trainer import ReturnDirectionModel, StockScorer
import warnings
warnings.filterwarnings('ignore')


def demo_workflow():
    """Run a complete demo of the workflow"""

    print("="*80)
    print("INDIAN STOCK SCREENER - DEMO WORKFLOW")
    print("="*80)
    print()

    # Step 1: Data Collection
    print("STEP 1: DATA COLLECTION")
    print("-"*80)

    # Use a small sample of popular stocks for demo
    demo_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS'
    ]

    print(f"Collecting data for {len(demo_tickers)} stocks...")
    print("This may take a few minutes...\n")

    # Collect data (5 years of history)
    data = collect_data_for_universe(demo_tickers, lookback_years=5)

    if data.empty:
        print("ERROR: Failed to collect data!")
        return

    print(f"\nData Collection Complete!")
    print(f"  - Stocks processed: {data['Ticker'].nunique()}")
    print(f"  - Total records: {len(data)}")
    print(f"  - Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"  - Features: {len(data.columns)}")
    print()

    # Show sample data
    print("Sample data (first 5 rows):")
    print(data.head()[['Date', 'Ticker', 'Close', 'RSI', 'ROC_1M', 'trailing_pe', 'roe']].to_string())
    print()

    # Step 2: Model Training
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("-"*80)

    print("Training LightGBM model to predict monthly return direction...")
    print()

    # Initialize and train model
    model = ReturnDirectionModel(test_size=0.2, random_state=42)

    try:
        metrics = model.train_model(data, use_time_series_split=True)
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        return

    print("\nModel Training Complete!")
    print()

    # Show feature importance
    print("Top 15 Most Important Features:")
    print("-"*80)
    top_features = model.get_feature_importance(top_n=15)
    for idx, row in top_features.iterrows():
        print(f"  {idx+1:2d}. {row['feature']:30s} - Importance: {row['importance']:,.0f}")
    print()

    # Step 3: Stock Scoring
    print("\n" + "="*80)
    print("STEP 3: STOCK SCREENING & SCORING")
    print("-"*80)

    print("Scoring stocks and assigning quality quintiles...")
    print()

    # Initialize scorer
    scorer = StockScorer(model)

    # Score stocks
    try:
        results = scorer.score_current_universe(data)
    except Exception as e:
        print(f"ERROR during scoring: {str(e)}")
        return

    print("Screening Complete!")
    print()

    # Show results by quintile
    print("Results by Quality Quintile:")
    print("-"*80)

    for quintile in ['Q5 (Highest)', 'Q4 (High)', 'Q3 (Medium)', 'Q2 (Low)', 'Q1 (Lowest)']:
        quintile_stocks = results[results['quality_quintile'] == quintile]
        if len(quintile_stocks) > 0:
            print(f"\n{quintile}:")
            print(f"  Count: {len(quintile_stocks)}")
            print(f"  Avg Composite Score: {quintile_stocks['composite_score'].mean():.1f}")
            print(f"  Avg Return Probability: {quintile_stocks['return_probability'].mean():.2%}")
            print(f"  Stocks: {', '.join(quintile_stocks['Ticker'].str.replace('.NS', '').tolist())}")

    print()

    # Show top 10 stocks
    print("\n" + "="*80)
    print("TOP 10 STOCKS BY COMPOSITE SCORE")
    print("-"*80)

    top_10 = results.nlargest(10, 'composite_score')[
        ['Ticker', 'quality_quintile', 'composite_score', 'return_probability',
         'trailing_pe', 'roe', 'ROC_12M']
    ].copy()

    # Clean ticker names
    top_10['Ticker'] = top_10['Ticker'].str.replace('.NS', '')

    # Format display
    print()
    print(f"{'Rank':<5} {'Ticker':<12} {'Quintile':<15} {'Score':<8} {'Ret Prob':<10} {'P/E':<8} {'ROE':<10} {'12M Mom':<10}")
    print("-"*90)

    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        ticker = row['Ticker']
        quintile = row['quality_quintile']
        score = f"{row['composite_score']:.1f}"
        prob = f"{row['return_probability']:.2%}"
        pe = f"{row['trailing_pe']:.2f}" if pd.notna(row['trailing_pe']) else "N/A"
        roe = f"{row['roe']*100:.1f}%" if pd.notna(row['roe']) else "N/A"
        mom = f"{row['ROC_12M']*100:.1f}%" if pd.notna(row['ROC_12M']) else "N/A"

        print(f"{idx:<5} {ticker:<12} {quintile:<15} {score:<8} {prob:<10} {pe:<8} {roe:<10} {mom:<10}")

    print()
    print("="*80)
    print("DEMO COMPLETE!")
    print()
    print("To run the full interactive application, use:")
    print("  streamlit run streamlit_app.py")
    print("="*80)


if __name__ == "__main__":
    demo_workflow()
