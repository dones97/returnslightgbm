# Indian Stock Screener - Complete Project Overview

## Project Summary

A sophisticated machine learning application for screening Indian stocks using LightGBM to predict monthly return direction and assign quality scores. The system processes historical price data, fundamental metrics, and technical indicators to rank stocks into quality quintiles.

## Key Features

✅ **Comprehensive Data Collection** from Yahoo Finance
✅ **30+ Technical Indicators** (RSI, MACD, Bollinger Bands, Momentum)
✅ **25+ Fundamental Metrics** (Valuation, Profitability, Growth, Risk)
✅ **LightGBM ML Model** for return direction prediction
✅ **Percentile Scoring System** for all stock characteristics
✅ **Quality Quintile Assignment** (Q1-Q5)
✅ **Interactive Streamlit Dashboard**
✅ **Time-Series Aware Training** to avoid look-ahead bias
✅ **Export Results to CSV**

## Project Files

### Core Application Files

1. **streamlit_app.py** (Main Application)
   - Three-page Streamlit interface
   - Data Collection page
   - Model Training page
   - Stock Screening page
   - Interactive filtering and visualization

2. **data_collector.py** (Data Pipeline)
   - `StockDataCollector` class
   - Fetches data from yfinance
   - Computes 20+ technical indicators
   - Extracts fundamental metrics
   - Converts to monthly frequency
   - Handles missing data

3. **model_trainer.py** (ML & Scoring)
   - `ReturnDirectionModel` class for training
   - `StockScorer` class for scoring
   - LightGBM binary classification
   - Feature importance analysis
   - Percentile score calculation
   - Quintile assignment

4. **config.py** (Configuration)
   - All adjustable parameters
   - Model hyperparameters
   - Feature engineering settings
   - UI configuration
   - Easily customizable

### Utility Files

5. **demo_workflow.py**
   - Complete workflow demo
   - Runs without Streamlit UI
   - Uses 15 popular stocks
   - Shows results in terminal
   - Great for testing

6. **test_data_availability.py**
   - Tests yfinance data access
   - Checks 5 major stocks
   - Validates fundamental data
   - Ensures API working

### Documentation

7. **README.md**
   - Comprehensive documentation
   - Architecture overview
   - Feature descriptions
   - Technical details
   - Troubleshooting guide

8. **QUICKSTART.md**
   - 5-minute setup guide
   - Step-by-step instructions
   - Common issues & solutions
   - Performance benchmarks
   - Example workflow

9. **PROJECT_OVERVIEW.md** (This file)
   - High-level summary
   - File descriptions
   - Architecture diagram
   - Technical approach

### Configuration Files

10. **requirements.txt**
    - Python dependencies
    - streamlit, pandas, numpy
    - yfinance, lightgbm
    - scikit-learn, plotly

11. **run_app.bat**
    - Windows launch script
    - One-click app start

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                  (Streamlit Web App)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────┐
│  Data   │  │  Model  │  │  Stock   │
│Collection│  │Training │  │Screening │
│  Page   │  │  Page   │  │   Page   │
└────┬────┘  └────┬────┘  └────┬─────┘
     │            │             │
     │            │             │
     ▼            ▼             ▼
┌─────────────────────────────────────────┐
│         DATA COLLECTOR MODULE           │
│  - Fetch from Yahoo Finance             │
│  - Compute technical indicators         │
│  - Extract fundamentals                 │
│  - Monthly resampling                   │
└──────────┬──────────────────────────────┘
           │
           ▼
      ┌─────────┐
      │  Cache  │ (stock_data_cache.pkl)
      └────┬────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      MODEL TRAINER MODULE               │
│  - Feature preparation                  │
│  - Train/test split (time-aware)        │
│  - LightGBM training                    │
│  - Feature importance                   │
└──────────┬──────────────────────────────┘
           │
           ▼
      ┌─────────┐
      │ Model   │ (trained_model.pkl)
      └────┬────┘
           │
           ▼
┌─────────────────────────────────────────┐
│        STOCK SCORER MODULE              │
│  - Calculate percentile scores          │
│  - Compute composite score              │
│  - Assign quality quintiles             │
└──────────┬──────────────────────────────┘
           │
           ▼
      ┌─────────┐
      │Results  │ (screening_results.pkl)
      └────┬────┘
           │
           ▼
    ┌────────────┐
    │   Export   │ (CSV Download)
    │   Results  │
    └────────────┘
```

## Data Flow

### 1. Data Collection Phase

```
NSE_Universe.csv
     │
     ├──> Load Tickers
     │
     ├──> For each ticker:
     │       │
     │       ├──> yfinance API
     │       │       │
     │       │       ├──> Historical Prices (OHLCV)
     │       │       └──> Fundamental Data (info)
     │       │
     │       ├──> Technical Indicators
     │       │       │
     │       │       ├──> Moving Averages (SMA, EMA)
     │       │       ├──> MACD, RSI
     │       │       ├──> Bollinger Bands
     │       │       ├──> Momentum (1M, 3M, 6M, 12M)
     │       │       ├──> Volatility
     │       │       └──> Volume indicators
     │       │
     │       ├──> Monthly Resampling
     │       │       │
     │       │       └──> Compute forward returns
     │       │
     │       └──> Combine Features
     │
     └──> Save to cache
```

### 2. Model Training Phase

```
Cached Data
     │
     ├──> Feature Preparation
     │       │
     │       ├──> Remove NaN targets
     │       ├──> Select feature columns
     │       ├──> Handle infinite values
     │       └──> Impute missing values
     │
     ├──> Train/Test Split (Time-Series)
     │       │
     │       ├──> Sort by date
     │       ├──> 80% train, 20% test
     │       └──> No shuffle (temporal order)
     │
     ├──> LightGBM Training
     │       │
     │       ├──> Binary classification
     │       ├──> AUC optimization
     │       ├──> 1000 rounds + early stopping
     │       └──> Feature importance tracking
     │
     ├──> Model Evaluation
     │       │
     │       ├──> Accuracy, Precision, Recall
     │       ├──> F1 Score, ROC AUC
     │       └──> Confusion Matrix
     │
     └──> Save Model
```

### 3. Stock Screening Phase

```
Latest Data + Trained Model
     │
     ├──> Model Predictions
     │       │
     │       └──> Probability of positive return
     │
     ├──> Factor Scoring
     │       │
     │       ├──> Select top 15 features
     │       ├──> Calculate percentile ranks
     │       ├──> Invert where lower is better
     │       └──> Handle missing values
     │
     ├──> Composite Score
     │       │
     │       ├──> Weighted average of percentiles
     │       └──> Scale to 0-100
     │
     ├──> Quintile Assignment
     │       │
     │       ├──> Sort by composite score
     │       ├──> Divide into 5 groups
     │       └──> Label Q1-Q5
     │
     └──> Display & Export
```

## Technical Approach

### Avoiding Circularity

✅ **Forward Returns Properly Shifted**
- Target is next month's return
- Never use future data for current prediction

✅ **Time-Series Split**
- Training data always before test data
- No random shuffling across time

✅ **Technical Indicators Use Past Only**
- Moving averages look backward
- Momentum is historical
- No look-ahead bias

### Feature Engineering

**Technical Features (20+)**
- Trend: SMAs, EMAs, MACD
- Momentum: ROC 1M/3M/6M/12M
- Volatility: Rolling std, ATR
- Oscillators: RSI, Stochastic
- Volume: Relative volume
- Position: Distance from 52W high/low

**Fundamental Features (25+)**
- Valuation: P/E, P/B, P/S, EV/EBITDA
- Profitability: ROE, ROA, margins
- Growth: Revenue growth, earnings growth
- Health: Debt/equity, current ratio
- Cash: Operating CF, free CF
- Income: Dividend yield
- Risk: Beta

### Model Choice: LightGBM

**Why LightGBM?**
- Fast training on large datasets
- Handles mixed features well
- Built-in missing value handling
- Great feature importance
- Less prone to overfitting
- Efficient memory usage

**Model Configuration**
- Objective: Binary classification
- Metric: AUC (area under ROC)
- Boosting: GBDT
- Regularization: L1 + L2
- Early stopping: 50 rounds

### Scoring System

**Percentile Approach**
1. Rank each feature across all stocks
2. Convert to percentile (0-100)
3. Invert for "lower is better" features
4. Calculate weighted average
5. Assign to quintiles

**Benefits**
- Robust to outliers
- Easy to interpret
- Comparable across features
- Relative ranking focus

## Performance Expectations

### Model Accuracy

**Realistic Expectations:**
- Accuracy: 55-60% (slightly better than random)
- ROC AUC: 0.55-0.65
- Precision: 55-65%

**Why Low?**
- Market prediction is inherently difficult
- Monthly returns are noisy
- Many unpredictable factors
- Model captures edges, not certainty

**How to Use:**
- Focus on relative rankings
- Combine with other analysis
- Use as screening tool, not oracle
- Track performance over time

### Processing Time

**Data Collection** (100 stocks, 5 years)
- Time: 5-10 minutes
- Bottleneck: API rate limits
- Mitigation: Caching

**Model Training**
- Time: 30-60 seconds
- Scales linearly with data
- GPU not required

**Screening**
- Time: <10 seconds
- Very fast inference
- Real-time filtering

## Data Requirements

### Minimum Requirements
- At least 12 months of price history
- Basic OHLCV data available
- Preferably some fundamental data

### Ideal Requirements
- 5+ years of price history
- Complete fundamental data
- Regular trading volume
- No extended suspensions

### Data Quality
- Not all stocks have full data
- Small caps often missing fundamentals
- Recently listed stocks have less history
- Filter based on your requirements

## Customization Options

### Easy Customizations (config.py)

1. **Data Parameters**
   - Lookback period (years)
   - Maximum stocks to process
   - Cache settings

2. **Model Parameters**
   - Test size percentage
   - Random seed
   - Time-series split toggle

3. **LightGBM Hyperparameters**
   - Learning rate
   - Number of leaves
   - Regularization strength
   - Number of rounds

4. **Feature Engineering**
   - Moving average periods
   - Momentum timeframes
   - Technical indicator parameters

5. **Scoring Weights**
   - Feature importance weights
   - Composite score calculation
   - Quintile thresholds

### Advanced Customizations (Code)

1. **Add New Features**
   - Edit `compute_technical_indicators()`
   - Add to `extract_fundamental_features()`

2. **Change Model**
   - Swap LightGBM for XGBoost, CatBoost
   - Try ensemble methods
   - Experiment with neural networks

3. **Alternative Targets**
   - Predict magnitude instead of direction
   - Multi-class (strong up/neutral/strong down)
   - Regression for exact returns

4. **Enhanced Screening**
   - Sector-relative scoring
   - Market cap weighted
   - Custom factor models

## Deployment Options

### Local Use (Current)
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment

**Streamlit Cloud** (Free)
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Auto-deploys on push

**AWS/GCP/Azure**
1. Containerize with Docker
2. Deploy to cloud platform
3. Set up scheduled retraining

**Heroku**
1. Add Procfile
2. Deploy via git
3. Scale dynos as needed

## Future Enhancements

### Potential Improvements

1. **Data Sources**
   - Add NSE/BSE direct APIs
   - Incorporate news sentiment
   - Include macroeconomic indicators

2. **Features**
   - Sector rotation signals
   - Peer comparisons
   - Analyst ratings

3. **Models**
   - Ensemble of multiple models
   - Separate models by sector
   - Sequential models (LSTM)

4. **Backtesting**
   - Walk-forward validation
   - Portfolio simulation
   - Risk-adjusted returns

5. **Alerts**
   - Email notifications
   - Telegram bot integration
   - Price alerts

6. **Advanced Analysis**
   - Factor attribution
   - Risk decomposition
   - Correlation analysis

## Limitations & Disclaimers

### Known Limitations

1. **Data Freshness**: Fundamentals update quarterly
2. **Data Quality**: Varies by stock
3. **Model Accuracy**: Modest (55-60%)
4. **Look-Ahead Risk**: Minimized but possible
5. **Survivorship Bias**: Only current stocks
6. **API Limits**: Yahoo Finance rate limits

### Important Disclaimers

⚠️ **Not Financial Advice**
- Educational purpose only
- Not investment recommendations
- Past performance ≠ future results

⚠️ **Do Your Own Research**
- Use as screening tool only
- Verify all data independently
- Consult financial advisors

⚠️ **Market Risks**
- Models can be wrong
- Markets are unpredictable
- Risk of capital loss

⚠️ **Data Accuracy**
- Free data may have errors
- Check important figures
- Use official sources for decisions

## Support & Resources

### Documentation
- README.md - Full documentation
- QUICKSTART.md - Quick setup guide
- Code comments - Inline documentation

### Testing
- test_data_availability.py - Verify data access
- demo_workflow.py - Test complete pipeline

### Configuration
- config.py - All settings in one place

### Community
- GitHub Issues - Report bugs
- Discussions - Ask questions
- Pull Requests - Contribute improvements

## License & Usage

This project is for educational and research purposes.

**Allowed:**
✅ Personal use and learning
✅ Modification and customization
✅ Research and analysis
✅ Educational purposes

**Not Allowed:**
❌ Commercial redistribution without permission
❌ Claiming as your own work
❌ Providing as financial advice service

## Credits

**Built with:**
- Streamlit (UI)
- LightGBM (ML)
- yfinance (Data)
- pandas/numpy (Processing)
- plotly (Visualization)

**Inspired by:**
- Quantitative finance research
- Factor investing principles
- Machine learning in finance

---

## Getting Started

Ready to begin? See [QUICKSTART.md](QUICKSTART.md) for setup instructions!

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Happy screening! 📈
