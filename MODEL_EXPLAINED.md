# LightGBM Model - Complete Explanation

## Model Overview

### What It Does
**Predicts the DIRECTION of next month's stock return** (Up or Down)

**Output**: Probability between 0 and 1
- Probability > 0.5 = Predicts positive return next month
- Probability < 0.5 = Predicts negative return next month

### Model Type
**Binary Classification using Gradient Boosted Decision Trees**

LightGBM builds an ensemble of decision trees:
- Each tree learns patterns from features
- Trees combine to make final prediction
- Automatically finds which features matter most
- Handles non-linear relationships
- Fast training even with many features

---

## Training Data Requirements

### With 5-Year Lookback (Default)

**Per Stock**:
- Total months: 60
- Usable for training: 59 (need next month return as target)
- Months with time-varying fundamentals: ~30 (50%)

### Expected Dataset Sizes

| Stocks | Total Observations | Training (80%) | Test (20%) | Status |
|--------|-------------------|----------------|------------|--------|
| 50 | ~3,000 | ~2,400 | ~600 | Marginal |
| **200** | **~12,000** | **~9,600** | **~2,400** | **Good (Recommended)** |
| 500 | ~30,000 | ~24,000 | ~6,000 | Excellent |
| 1,500 | ~90,000 | ~72,000 | ~18,000 | Excellent |

**Recommendation**: Start with **200 stocks** (~12,000 observations)
- Sufficient for LightGBM to find robust patterns
- Fast enough for testing (10-15 min data collection)
- Can scale to 500-1500 later

---

## Features Used (~50 Total)

### 1. Technical Features (~7 features - 20% weight)

Simplified for fundamental focus:

1. **ROC_1M** - 1-month price momentum
2. **ROC_3M** - 3-month price momentum
3. **ROC_6M** - 6-month price momentum
4. **ROC_12M** - 12-month price momentum
5. **Volatility_60** - 60-day volatility
6. **RSI** - Relative Strength Index
7. **Volume_Ratio** - Volume vs average

### 2. Quarterly Fundamental Features (~40 features - 80% weight)

**THESE ARE TIME-VARYING** - They change each quarter based on latest financials!

#### Income Statement Metrics (6)
- Total_Revenue
- Gross_Profit
- Operating_Income
- EBITDA
- EBIT
- Net_Income

#### Margins (4) - **TIME-VARYING**
- **Gross_Margin**
- **Operating_Margin**
- **Net_Margin**
- **EBITDA_Margin**

These are calculated quarterly: Profit / Revenue
- Jan 2020: Uses Q4 2019 data
- Apr 2020: Uses Q1 2020 data (CHANGES!)
- Jul 2020: Uses Q2 2020 data (CHANGES!)

#### Growth Rates (5) - **TIME-VARYING**
- **Revenue_Growth_QoQ** - Quarter-over-quarter
- **Revenue_Growth_YoY** - Year-over-year
- **Net_Income_Growth_QoQ**
- **Net_Income_Growth_YoY**
- **EBITDA_Growth_YoY**

These change as company grows/shrinks!

#### Balance Sheet (6)
- Total_Assets
- Total_Debt
- Stockholders_Equity
- Current_Assets
- Current_Liabilities
- Cash

#### Financial Ratios (4) - **TIME-VARYING**
- **Debt_to_Equity** - Leverage
- **Debt_to_Assets** - Debt burden
- **Current_Ratio** - Liquidity
- **Quick_Ratio** - Quick liquidity

These ratios change as balance sheet changes!

#### Profitability Ratios (3) - **TIME-VARYING**
- **ROE** (Return on Equity) - Calculated with TTM earnings
- **ROA** (Return on Assets) - Calculated with TTM earnings
- **ROIC** (Return on Invested Capital) - Calculated with TTM operating income

These are the KEY metrics you want to analyze!

#### Trailing 12-Month Metrics (5)
- TTM_Revenue - Last 4 quarters summed
- TTM_Net_Income - Last 4 quarters summed
- TTM_Operating_Income
- TTM_EBITDA
- TTM_Operating_Cash_Flow

#### Cash Flow (3)
- Operating_Cash_Flow
- Free_Cash_Flow
- Capital_Expenditure

#### Advanced Scores (2) - **IMPORTANT**
- **F_Score** - Piotroski F-Score (0-9)
  - Higher = Better quality
  - Measures: Profitability, Leverage, Operating efficiency
  - TIME-VARYING based on quarterly data

- **Z_Score** - Altman Z-Score
  - Higher = Lower bankruptcy risk
  - Z > 2.99: Safe zone
  - 1.81 < Z < 2.99: Grey zone
  - Z < 1.81: Distress zone
  - TIME-VARYING based on quarterly data

### 3. Current Snapshot Features (3)
- market_cap_current - Current market cap
- beta - Current beta
- dividend_yield - Current dividend yield

---

## How Features Change Over Time (Example)

### RELIANCE.NS - ROE Over Time

| Month | Quarter Data Used | Net Income (TTM) | Equity | ROE | Change |
|-------|------------------|------------------|--------|-----|--------|
| Jan 2020 | Q4 2019 | 36,000 Cr | 3,00,000 Cr | 12.0% | - |
| Apr 2020 | Q1 2020 | 34,000 Cr | 3,05,000 Cr | 11.1% | Down |
| Jul 2020 | Q2 2020 | 28,000 Cr | 3,15,000 Cr | 8.9% | Down |
| Oct 2020 | Q3 2020 | 32,000 Cr | 3,20,000 Cr | 10.0% | Up |
| Jan 2021 | Q4 2020 | 38,000 Cr | 3,25,000 Cr | 11.7% | Up |

**This is TRUE time-varying data!**

Model learns: "When ROE increases quarter-over-quarter, does next month's return tend to be positive?"

---

## Time Alignment (No Circularity!)

**Critical**: We prevent look-ahead bias using proper time shifts:

```
Month 1 Features (Jan 2020):
  - Price/Technical data: Up to Jan 31, 2020
  - Fundamental data: From Q4 2019 (reported before Jan)
  - Target: Feb 2020 return (shift(-1))

Month 2 Features (Feb 2020):
  - Price/Technical data: Up to Feb 29, 2020
  - Fundamental data: Still Q4 2019 (forward-filled)
  - Target: Mar 2020 return (shift(-1))

Month 4 Features (Apr 2020):
  - Price/Technical data: Up to Apr 30, 2020
  - Fundamental data: From Q1 2020 (NEW!)
  - Target: May 2020 return (shift(-1))
```

**We NEVER use future data to predict past!**

---

## Training Process (Time-Series Split)

### 5-Year Data (60 months per stock)

```
Months 1-48 (4 years) => TRAINING SET (80%)
├─ Model learns patterns here
├─ Builds decision trees
└─ Finds which features predict returns

Months 49-60 (1 year) => TEST SET (20%)
├─ Model has never seen this data
├─ Simulates real prediction
└─ Evaluates performance

NO SHUFFLING - maintains temporal order
```

### With 200 Stocks:

**Training Set**:
- 200 stocks × 48 months = 9,600 observations
- Model learns from these

**Test Set**:
- 200 stocks × 12 months = 2,400 observations
- Model evaluated on these

**Total**: 12,000 observations - **SUFFICIENT for LightGBM!**

---

## LightGBM Hyperparameters

Optimized configuration in `config.py`:

```python
{
    'objective': 'binary',              # Binary classification (up/down)
    'metric': 'auc',                    # Area Under ROC Curve
    'boosting_type': 'gbdt',           # Gradient Boosting Decision Trees
    'num_leaves': 31,                   # Max leaves per tree
    'learning_rate': 0.05,              # Step size (conservative)
    'feature_fraction': 0.8,            # Use 80% of features per tree
    'bagging_fraction': 0.8,            # Use 80% of data per tree
    'bagging_freq': 5,                  # Bagging every 5 iterations
    'max_depth': -1,                    # No depth limit
    'min_child_samples': 20,            # Min samples per leaf
    'reg_alpha': 0.1,                   # L1 regularization
    'reg_lambda': 0.1,                  # L2 regularization
    'num_boost_round': 1000,            # Max 1000 trees
    'early_stopping_rounds': 50         # Stop if no improvement for 50 rounds
}
```

**What this means**:
- Builds up to 1000 decision trees
- Stops early if validation performance plateaus
- Uses regularization to prevent overfitting
- Typical training: 200-400 trees before early stop

---

## What The Model Learns

LightGBM builds decision trees that capture patterns like:

### Example Tree 1:
```
If F_Score >= 7 AND ROE > 15% AND Debt_to_Equity < 0.5:
    => 72% probability of positive return
```

### Example Tree 2:
```
If Revenue_Growth_YoY < -10% AND Operating_Margin decreasing:
    => 28% probability of positive return (AVOID!)
```

### Example Tree 3:
```
If ROC_12M > 20% AND ROE improving AND Z_Score > 2.5:
    => 78% probability of positive return
```

### Example Tree 4:
```
If Debt_to_Equity increasing AND Current_Ratio < 1.0:
    => 35% probability of positive return (RISKY!)
```

Combines hundreds of such trees => Final probability

**Model automatically discovers**:
- Which fundamentals matter most
- Optimal thresholds (e.g., ROE > 15%)
- Interaction effects (e.g., high ROE + low debt)
- Non-linear relationships

---

## Feature Importance - What You'll See

After training, you'll see which features predicted returns best.

### Expected Top Features (Fundamental-Focused):

1. **F_Score** - Quality score
2. **ROE** - Profitability
3. **Revenue_Growth_YoY** - Growth momentum
4. **Debt_to_Equity** - Leverage risk
5. **Operating_Margin** - Efficiency
6. **ROC_12M** - Price momentum
7. **Net_Margin** - Profitability
8. **Z_Score** - Financial health
9. **ROIC** - Capital efficiency
10. **Current_Ratio** - Liquidity

**If fundamental features dominate top 10**: ✓ Working as intended!
**If technical features dominate**: Model not finding fundamental signals (may need more data)

---

## Expected Performance

### Realistic Expectations:

| Metric | Expected Range | What It Means |
|--------|---------------|---------------|
| **Accuracy** | 55-60% | Better than random (50%) |
| **ROC AUC** | 0.55-0.65 | Modest predictive power |
| **Precision** | 55-65% | When predicts up, correct 55-65% |
| **Recall** | 55-65% | Catches 55-65% of upward moves |

### Why Not Higher?

1. **Market prediction is inherently difficult**
   - Many unpredictable factors (news, macro, sentiment)
   - Even professional investors struggle

2. **Monthly returns are noisy**
   - Short-term randomness dominates
   - Long-term patterns are subtle

3. **This is actually GOOD performance**
   - 55-60% accuracy means statistical edge
   - Can be profitable with proper risk management
   - Focus on relative rankings, not absolute predictions

### What Matters More:

**Feature importance** > Accuracy
- Understanding WHICH fundamentals predict returns
- Building investment strategy around findings
- Relative stock rankings (quintiles)

---

## Research Questions You Can Answer

With this model and data, you can analyze:

### 1. Which Fundamental Characteristics Predict Outperformance?

**Check feature importance to see**:
- Is high ROE predictive?
- Does low P/E matter?
- F-Score effectiveness?
- Revenue growth importance?
- Optimal debt levels?

### 2. Quality vs Value vs Growth

**Compare top quintile characteristics**:
- Q5 (best): High ROE + Low Debt + High Growth?
- Q1 (worst): Low ROE + High Debt + Negative Growth?
- Which factors are most discriminating?

### 3. Improving vs Static Fundamentals

**Analyze trends**:
- Do improving margins predict returns?
- Revenue acceleration matter?
- Debt reduction signal?

### 4. Factor Combinations

**Optimal combinations**:
- Best 3-5 metrics?
- Complementary factors?
- Weighting scheme?

### 5. Risk vs Return

**Using Z-Score and debt ratios**:
- Can you get high returns with low risk?
- Trade-off between quality and growth?

---

## Data Flow Summary

### 1. Data Collection (10-15 min for 200 stocks)
```
data_collector_enhanced.py:
├─ Fetch 5 years price data (daily OHLCV)
├─ Fetch quarterly financials (6-8 quarters)
├─ Calculate quarterly ratios (ROE, margins, etc.)
├─ Calculate growth rates (YoY, QoQ)
├─ Calculate advanced scores (F-Score, Z-Score)
├─ Resample to monthly frequency
└─ Forward-fill quarterly fundamentals to monthly
```

### 2. Feature Preparation (automatic)
```
model_trainer.py:
├─ Load monthly data with features
├─ Remove rows with missing target
├─ Select feature columns (~50 features)
├─ Handle missing values (median imputation)
└─ Create training/test split (time-series aware)
```

### 3. Model Training (30-60 seconds)
```
model_trainer.py:
├─ Initialize LightGBM with hyperparameters
├─ Train on training set (80% of data)
├─ Validate on test set (20% of data)
├─ Early stopping if no improvement
├─ Calculate feature importance
└─ Evaluate performance metrics
```

### 4. Stock Scoring & Ranking (instant)
```
model_trainer.py:
├─ Predict probability for all stocks
├─ Calculate percentile scores for features
├─ Compute composite score
├─ Assign quality quintiles (Q1-Q5)
└─ Export results
```

---

## Minimum Data Requirements

### For LightGBM to Work:

**Absolute Minimum** (not recommended):
- 1,000 observations
- Can train but results unreliable

**Recommended Minimum**:
- 10,000+ observations ✓
- Provides reasonable patterns
- **Our 200-stock default: ~12,000 obs** ✓

**Optimal**:
- 50,000+ observations
- Very robust patterns
- **500-1500 stocks: 30,000-90,000 obs** ✓

### Why 200 Stocks is Good Enough:

1. **LightGBM is efficient**
   - Designed for smaller datasets
   - Regularization prevents overfitting
   - Works well with 10,000+ observations

2. **Cross-sectional + Time-series**
   - 200 stocks × 60 months = 12,000 observations
   - Captures stock differences AND time patterns
   - Sufficient for robust patterns

3. **Can always scale up**
   - Start with 200 (test quickly)
   - If results promising, scale to 500-1500
   - More stocks = more robust, but diminishing returns

---

## How to Interpret Results

### After Training:

**1. Check Feature Importance**
- Are fundamental features in top 10? ✓ Good
- Which specific fundamentals matter most?
- This answers your research question!

**2. Check Model Performance**
- Accuracy 55-60%? ✓ Normal
- ROC AUC > 0.55? ✓ Has predictive power
- Don't expect 80%+ accuracy (unrealistic)

**3. Analyze Quintiles**
- Q5 (highest) characteristics?
- Q1 (lowest) characteristics?
- Clear differentiation? ✓ Model working

**4. Export and Backtest**
- Download Q4-Q5 stocks
- Track performance over time
- Refine strategy based on learnings

---

## Recommended Starting Point

### First Run Configuration:

**Data Collection**:
- ✓ Historical period: **5 years**
- ✓ Number of stocks: **200**
- ✓ Processing time: 10-15 minutes
- ✓ Expected observations: ~12,000

**Model Training**:
- ✓ Test size: **20%**
- ✓ Time-series split: **YES** (critical!)
- ✓ Training observations: ~9,600
- ✓ Test observations: ~2,400
- ✓ Training time: 30-60 seconds

**Expected Results**:
- ✓ Accuracy: 55-60%
- ✓ Top features: Fundamentals
- ✓ Clear quintile differentiation
- ✓ Actionable insights

### After First Run:

**If results are good**:
- Scale to 500-1500 stocks
- More robust patterns
- Better coverage

**If results are weak**:
- Consider Alpha Vantage (test)
- Or build Screener.in scraper
- Get more fundamental history

---

## Summary

**Your LightGBM model will**:
- ✓ Train on ~12,000 observations (200 stocks × 60 months)
- ✓ Use ~50 features (80% fundamental, 20% technical)
- ✓ Learn from 2-3 years of time-varying fundamentals
- ✓ Predict next month's return direction
- ✓ Show which fundamentals predict outperformance
- ✓ Rank stocks into quality quintiles

**This is sufficient to answer**:
> "What fundamental characteristics lead to outperformance?"

**Ready to start**:
```bash
streamlit run streamlit_app.py
```

Then:
1. Collect data (200 stocks, 5 years)
2. Train model
3. Check feature importance ← **Key insight here!**
4. Screen stocks
5. Export results

---

*Model: LightGBM Binary Classifier*
*Data: 5 years, 200 stocks, ~50 features*
*Focus: Time-varying fundamentals*
*Goal: Understand what drives outperformance*
