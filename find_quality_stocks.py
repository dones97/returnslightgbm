"""
Find stocks with good historical coverage for training

Strategy: Train on high-quality stocks (8+ years data)
         Then predict on ALL stocks (even with limited data)
"""
import sqlite3
import pandas as pd
import numpy as np

print("="*80)
print("FINDING STOCKS WITH QUALITY HISTORICAL DATA")
print("="*80)

conn = sqlite3.connect('screener_data.db')

# Get quarterly data coverage
print("\n[1] Analyzing quarterly data coverage...")
quarterly = pd.read_sql_query(
    "SELECT ticker, quarter_date FROM quarterly_results",
    conn
)
quarterly['quarter_date'] = pd.to_datetime(quarterly['quarter_date'], format='%b %Y', errors='coerce')

# Count quarters per stock
quarters_per_stock = quarterly.groupby('ticker').agg({
    'quarter_date': ['count', 'min', 'max']
}).reset_index()
quarters_per_stock.columns = ['ticker', 'num_quarters', 'first_quarter', 'last_quarter']

# Calculate years of coverage
quarters_per_stock['years_coverage'] = (
    (quarters_per_stock['last_quarter'] - quarters_per_stock['first_quarter']).dt.days / 365.25
)

print(f"Total stocks: {len(quarters_per_stock)}")
print(f"\nQuarterly data distribution:")
print(f"  Mean quarters: {quarters_per_stock['num_quarters'].mean():.1f}")
print(f"  Median quarters: {quarters_per_stock['num_quarters'].median():.0f}")
print(f"  Max quarters: {quarters_per_stock['num_quarters'].max()}")
print(f"\nYears coverage distribution:")
print(f"  Mean: {quarters_per_stock['years_coverage'].mean():.1f} years")
print(f"  Median: {quarters_per_stock['years_coverage'].median():.1f} years")
print(f"  Max: {quarters_per_stock['years_coverage'].max():.1f} years")

# Get annual data coverage
print("\n[2] Analyzing annual data coverage...")

# Check each annual table
annual_coverage = {}
for table in ['annual_profit_loss', 'balance_sheet', 'cash_flow', 'annual_ratios']:
    df = pd.read_sql_query(f"SELECT ticker, year FROM {table}", conn)
    coverage = df.groupby('ticker')['year'].count().reset_index()
    coverage.columns = ['ticker', f'{table}_years']
    annual_coverage[table] = coverage

# Merge all annual coverage
all_coverage = quarters_per_stock.copy()
for table, coverage in annual_coverage.items():
    all_coverage = all_coverage.merge(coverage, on='ticker', how='left')

# Fill NaN with 0
for table in annual_coverage.keys():
    all_coverage[f'{table}_years'] = all_coverage[f'{table}_years'].fillna(0)

# Calculate minimum annual coverage (bottleneck)
annual_cols = [f'{table}_years' for table in annual_coverage.keys()]
all_coverage['min_annual_years'] = all_coverage[annual_cols].min(axis=1)

print(f"\nAnnual data availability:")
for table in annual_coverage.keys():
    col = f'{table}_years'
    print(f"  {table}: {all_coverage[col].mean():.1f} years (mean)")

# Find stocks with good coverage
print("\n[3] Identifying stocks with 8+ years of data...")

# Criteria for "good" stock:
# - At least 28 quarters (7 years of quarterly data)
# - At least 6 years of annual data across all tables
good_stocks_quarterly = all_coverage[all_coverage['num_quarters'] >= 28]
good_stocks_annual = all_coverage[all_coverage['min_annual_years'] >= 6]

# Combined: stocks with EITHER good quarterly OR good annual
good_stocks = all_coverage[
    (all_coverage['num_quarters'] >= 28) |
    (all_coverage['min_annual_years'] >= 6)
]

print(f"\nStocks with 28+ quarters (7 years): {len(good_stocks_quarterly)}")
print(f"Stocks with 6+ years annual data: {len(good_stocks_annual)}")
print(f"Stocks with good coverage (either): {len(good_stocks)}")

# Sort by total data quality
all_coverage['data_quality_score'] = (
    all_coverage['num_quarters'] * 0.3 +  # Quarterly weight
    all_coverage['min_annual_years'] * 10   # Annual weight (more valuable)
)
all_coverage = all_coverage.sort_values('data_quality_score', ascending=False)

# Show top stocks
print(f"\n[4] Top 20 stocks by data quality:")
print("-" * 80)
top_20 = all_coverage.head(20)
for i, row in top_20.iterrows():
    print(f"{row['ticker']:15s} | Quarters: {row['num_quarters']:2.0f} | "
          f"Annual: {row['min_annual_years']:.0f}y | "
          f"Coverage: {row['years_coverage']:.1f}y | "
          f"Score: {row['data_quality_score']:.0f}")

# Find stocks with 8+ years of coverage
stocks_8y = all_coverage[
    (all_coverage['years_coverage'] >= 8) |
    (all_coverage['min_annual_years'] >= 8)
]

print(f"\n[5] Stocks with 8+ years of data: {len(stocks_8y)}")

if len(stocks_8y) >= 100:
    print(f"   [OK] Enough stocks for good training ({len(stocks_8y)} stocks)")
    recommended = min(len(stocks_8y), 500)
    print(f"   RECOMMENDATION: Use top {recommended} stocks by data quality")
elif len(stocks_8y) >= 50:
    print(f"   [WARNING] Limited stocks, but workable ({len(stocks_8y)} stocks)")
    print(f"   RECOMMENDATION: Use all {len(stocks_8y)} stocks")
else:
    print(f"   [WARNING] Very few stocks with 8+ years ({len(stocks_8y)} stocks)")
    print(f"   FALLBACK: Lower threshold to 6 years")

    stocks_6y = all_coverage[
        (all_coverage['years_coverage'] >= 6) |
        (all_coverage['min_annual_years'] >= 6)
    ]
    print(f"   Stocks with 6+ years: {len(stocks_6y)}")
    recommended = min(len(stocks_6y), 500)
    print(f"   RECOMMENDATION: Use top {recommended} stocks by data quality")

# Save list of good stocks
good_stock_list = all_coverage.head(500)['ticker'].tolist()

print(f"\n[6] Creating training stock list...")
print(f"   Saving top 500 stocks by data quality to: training_stocks.txt")

with open('training_stocks.txt', 'w') as f:
    for ticker in good_stock_list:
        f.write(f"{ticker}\n")

# Show breakdown by years
print(f"\n[7] Data coverage breakdown:")
print("-" * 80)
bins = [0, 2, 4, 6, 8, 10, 15]
labels = ['<2y', '2-4y', '4-6y', '6-8y', '8-10y', '10+y']
all_coverage['coverage_bin'] = pd.cut(all_coverage['years_coverage'], bins=bins, labels=labels)
print(all_coverage['coverage_bin'].value_counts().sort_index())

# Show actual date ranges for top stocks
print(f"\n[8] Sample stocks - actual date ranges:")
print("-" * 80)
sample = all_coverage.head(10)
for i, row in sample.iterrows():
    print(f"{row['ticker']:15s} | {row['first_quarter'].date()} to {row['last_quarter'].date()} "
          f"({row['years_coverage']:.1f} years)")

# Estimate training samples with good stocks
print(f"\n[9] Estimated training samples:")
print("-" * 80)

# Use top stocks by data quality
n_stocks_to_use = min(500, len(all_coverage))
top_stocks = all_coverage.head(n_stocks_to_use)

# Estimate months per stock (conservative)
avg_years = top_stocks['years_coverage'].mean()
avg_months = avg_years * 12
estimated_samples = n_stocks_to_use * avg_months * 0.8  # 80% success rate

print(f"Top {n_stocks_to_use} stocks by data quality:")
print(f"  Average coverage: {avg_years:.1f} years ({avg_months:.0f} months)")
print(f"  Estimated samples: {estimated_samples:.0f}")
print(f"  (Assuming 80% data availability)")

if estimated_samples >= 40000:
    print(f"\n[OK] Excellent! Enough samples for robust training")
elif estimated_samples >= 20000:
    print(f"\n[OK] Good! Sufficient samples for training")
else:
    print(f"\n[WARNING] Limited samples, may need to lower data quality threshold")

conn.close()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\nRECOMMENDATION:")
print(f"  Use top {n_stocks_to_use} stocks by data quality")
print(f"  Expected samples: {estimated_samples:.0f}")
print(f"  Average coverage: {avg_years:.1f} years")
print(f"\n  Stock list saved to: training_stocks.txt")
print(f"\nNEXT STEP:")
print(f"  Update model_trainer_monthly.py to use stocks from training_stocks.txt")
print(f"  Or use top N stocks by data quality score")
