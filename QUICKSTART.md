# Quick Start Guide - Indian Stock Screener

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Step 1: Install Dependencies

Open a terminal/command prompt in this folder and run:

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (Web UI framework)
- pandas & numpy (Data manipulation)
- yfinance (Market data)
- lightgbm (Machine learning)
- scikit-learn (ML utilities)
- plotly (Interactive charts)

### Step 2: Verify Installation

Run the data availability test:

```bash
python test_data_availability.py
```

This will check if yfinance can fetch data for major Indian stocks.

## Running the Application

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run streamlit_app.py
```

Or on Windows, double-click:
```
run_app.bat
```

Your browser will open automatically with the application.

### Option 2: Command Line Demo

Run a quick demo with 15 popular stocks:

```bash
python demo_workflow.py
```

This runs the complete pipeline without the UI and shows results in the terminal.

## Using the Application

### Step 1: Data Collection

1. Open the Streamlit app
2. Go to "Data Collection" page
3. Select:
   - Historical period: 5 years (recommended)
   - Max stocks: Start with 50-100 for testing
4. Click "Start Data Collection"
5. Wait for data to download (5-10 minutes for 100 stocks)

**Tip**: Check "Use cached data" to skip re-downloading on subsequent runs.

### Step 2: Model Training

1. Go to "Model Training" page
2. Select:
   - Test set size: 20% (default)
   - Use time-series split: ✓ (recommended)
3. Click "Train Model"
4. Review performance metrics
5. Check feature importance chart

**Expected Performance**:
- Accuracy: 55-60%
- ROC AUC: 0.55-0.65

Note: Predicting market direction is inherently difficult!

### Step 3: Stock Screening

1. Go to "Stock Screening" page
2. Click "Run Screening"
3. Filter results:
   - Select quality quintiles (Q4, Q5 for best stocks)
   - Set minimum composite score
   - Set minimum return probability
4. View ranked stocks
5. Download results as CSV

## Understanding the Results

### Quality Quintiles

- **Q5 (Highest)**: Top 20% of stocks - Best characteristics
- **Q4 (High)**: Next 20% - Good characteristics
- **Q3 (Medium)**: Middle 20% - Average characteristics
- **Q2 (Low)**: Lower 20% - Below average
- **Q1 (Lowest)**: Bottom 20% - Worst characteristics

### Key Metrics

- **Composite Score**: Overall quality score (0-100)
  - Higher = Better quality

- **Return Probability**: Model's predicted probability of positive next-month return
  - 0.6+ = Model thinks likely to go up
  - 0.4- = Model thinks likely to go down

- **Trailing P/E**: Price to Earnings ratio
  - Lower often = Better value
  - Compare within same sector

- **ROE**: Return on Equity
  - Higher = Better profitability
  - 15%+ is generally good

- **12M Momentum (ROC_12M)**: Price change over last year
  - Positive = Stock trending up
  - Negative = Stock trending down

## Tips for Best Results

### Data Collection

1. **Start Small**: Test with 50 stocks before processing full universe
2. **Check Coverage**: Not all stocks have complete data
3. **Use Cache**: Enable caching to avoid re-downloading
4. **Stable Connection**: Ensure good internet during data collection

### Model Training

1. **Sufficient Data**: Use at least 3 years of history
2. **Time-Series Split**: Always use this to avoid look-ahead bias
3. **Regular Retraining**: Markets change - retrain monthly
4. **Validate Results**: Don't trust model blindly

### Stock Selection

1. **Focus on Q4-Q5**: These are highest quality stocks
2. **Combine with Research**: Use this as a starting point
3. **Sector Diversification**: Don't pick all from one sector
4. **Check Liquidity**: Ensure stocks have good trading volume
5. **Verify Data**: Double-check important metrics

## Common Issues & Solutions

### Issue: Data collection is slow
**Solution**:
- Reduce max_stocks parameter
- Check internet connection
- Some stocks may be delisted (normal failures)

### Issue: Model accuracy is low
**Solution**:
- This is normal! Predicting market direction is hard
- Focus on relative rankings, not absolute predictions
- Consider it one factor among many

### Issue: Missing fundamental data
**Solution**:
- Common for small-cap stocks
- Filter for larger companies
- Use NSE_Universe.csv which has more liquid stocks

### Issue: App crashes during training
**Solution**:
- Reduce number of stocks
- Close other applications
- Check if data has enough valid records

## Next Steps

### After First Run

1. **Analyze Results**: Look at top quintile stocks
2. **Backtest**: Track predictions over time
3. **Refine**: Adjust filters based on your strategy
4. **Expand**: Process full universe once comfortable

### Customization

1. Edit `config.py` to change parameters
2. Modify feature weights in scoring
3. Add custom technical indicators
4. Adjust LightGBM hyperparameters

## Support & Documentation

- **Full README**: See README.md for detailed documentation
- **Code Comments**: All modules are well-commented
- **Configuration**: Check config.py for all settings

## Important Disclaimers

⚠️ **This is not financial advice**
⚠️ **Past performance ≠ future results**
⚠️ **Always do your own research**
⚠️ **Consult a financial advisor before investing**

Use this tool to screen and research stocks, but make investment decisions based on comprehensive analysis, not just model predictions.

## Example Workflow

Here's a typical workflow:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test data availability
python test_data_availability.py

# 3. Run demo (optional)
python demo_workflow.py

# 4. Launch app
streamlit run streamlit_app.py

# 5. In the app:
#    - Collect data for 100 stocks (5 years)
#    - Train model
#    - Screen stocks
#    - Filter for Q5 quintile
#    - Download top stocks as CSV
```

## Performance Benchmarks

Approximate processing times (may vary):

| Operation | 50 Stocks | 100 Stocks | 500 Stocks |
|-----------|-----------|------------|------------|
| Data Collection | 3-5 min | 5-10 min | 30-60 min |
| Model Training | 10-30 sec | 30-60 sec | 2-5 min |
| Screening | <5 sec | <10 sec | <30 sec |

---

**Ready to start? Run:**
```bash
streamlit run streamlit_app.py
```

Good luck with your stock screening! 📈
