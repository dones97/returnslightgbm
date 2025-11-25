"""
Quick test to verify the KeyError fix
"""
from data_collector_enhanced import EnhancedStockDataCollector
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TESTING ENHANCED DATA COLLECTOR FIX")
print("=" * 80)

# Test with a few stocks that might have incomplete data
test_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']

collector = EnhancedStockDataCollector(lookback_years=5)

for ticker in test_tickers:
    print(f"\nTesting {ticker}...")
    try:
        df = collector.create_feature_dataframe(ticker)
        if df is not None and not df.empty:
            print(f"  [OK] Successfully collected {len(df)} months of data")
            print(f"  [OK] Features: {len(df.columns)} columns")

            # Check for key columns
            key_features = ['ROE', 'F_Score', 'Z_Score', 'Operating_Margin']
            present = [f for f in key_features if f in df.columns]
            print(f"  [OK] Key features present: {', '.join(present)}")
        else:
            print(f"  [WARN] No data returned")
    except KeyError as e:
        print(f"  [ERROR] KeyError: {e}")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:100]}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nIf all tests show [OK], the fix is working correctly.")
print("If you see [ERROR], there are still issues to fix.")
