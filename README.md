# Indian Stock Screener - ML Powered

A comprehensive machine learning-powered stock screening application for Indian equity markets. Uses LightGBM to predict monthly return direction and assigns quality scores to stocks.

## Features

- **Comprehensive Data Collection**: Fetches historical price data and fundamental metrics from Yahoo Finance
- **Advanced Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, momentum factors
- **Fundamental Analysis**: P/E ratios, ROE, debt metrics, profitability margins, and more
- **ML-Based Predictions**: LightGBM model trained to predict direction of monthly returns
- **Percentile Scoring**: Each stock characteristic converted to percentile score
- **Quality Quintiles**: Stocks categorized into 5 quality tiers (Q1-Q5)
- **Interactive Dashboard**: Streamlit-based UI for easy exploration and analysis

## Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── data_collector.py          # Data fetching and feature engineering
├── model_trainer.py           # LightGBM model training and scoring
├── test_data_availability.py  # Data availability testing script
├── NSE_Universe.csv           # Universe of NSE stocks
├── indian_stocks_tickers.csv  # Backup ticker list
└── requirements.txt           # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Workflow

1. **Data Collection (Step 1)**
   - Select historical period (3-10 years)
   - Choose number of stocks to process
   - Fetches price data and fundamentals from Yahoo Finance
   - Computes technical indicators
   - Caches data for future use

2. **Model Training (Step 2)**
   - Train LightGBM model on collected data
   - Uses 80/20 train-test split
   - Predicts direction of next month's returns (binary classification)
   - Displays performance metrics and feature importance
   - Model saved for reuse

3. **Stock Screening (Step 3)**
   - Scores all stocks using trained model
   - Calculates percentile ranks for key characteristics
   - Assigns composite score and quality quintile
   - Interactive filtering and sorting
   - Export results to CSV

## Data Features

### Technical Indicators
- Moving Averages (SMA, EMA)
- MACD and Signal Line
- RSI (14-day)
- Bollinger Bands
- ATR (Average True Range)
- Volume indicators
- Momentum (1M, 3M, 6M, 12M)
- Volatility measures
- 52-week high/low distance

### Fundamental Metrics
- Valuation: P/E, P/B, P/S, EV/EBITDA
- Profitability: ROE, ROA, margins
- Growth: Revenue growth, earnings growth
- Financial Health: Debt/Equity, current ratio
- Cash Flow metrics
- Dividend yield and payout ratio
- Beta (risk measure)

## Model Details

### Algorithm
- **LightGBM** (Gradient Boosting Decision Trees)
- Binary classification for return direction
- 1000 boosting rounds with early stopping
- Optimized for AUC

### Training
- Time-series aware splitting to avoid look-ahead bias
- 80/20 train-test split
- Monthly frequency (predicts next month's return direction)
- Feature importance analysis using gain metric

### Scoring System
1. Model predictions (probability of positive return)
2. Percentile scores for top features
3. Composite score (weighted average of percentiles)
4. Quintile assignment (Q1-Q5)

## Data Sources

- **Price Data**: Yahoo Finance (yfinance library)
- **Fundamental Data**: Yahoo Finance stock info API
- **Stock Universe**: NSE_Universe.csv (user-provided)

## Important Considerations

### Data Quality
- Not all stocks have complete fundamental data
- Historical data availability varies by stock
- Missing values are handled via median imputation

### Model Limitations
- Past performance doesn't guarantee future results
- Model predicts direction, not magnitude
- Fundamentals may be stale (quarterly updates)
- Market conditions change over time

### Circularity Prevention
- Forward returns are properly shifted to avoid data leakage
- Time-series split prevents training on future data
- Technical indicators use only past information

## Configuration

### Adjustable Parameters

**Data Collection:**
- `lookback_years`: Historical period (3-10 years)
- `max_stocks`: Limit for processing

**Model Training:**
- `test_size`: Proportion for testing (10-30%)
- `use_time_series_split`: Time-aware vs random split

**Screening:**
- Quintile filters
- Minimum composite score
- Minimum return probability

## Performance Tips

1. **Start Small**: Test with 50-100 stocks first
2. **Cache Data**: Use cached data to avoid re-downloading
3. **Save Models**: Trained models are cached automatically
4. **Filter Wisely**: Use NSE_Universe.csv for liquid stocks only

## Troubleshooting

### Common Issues

**Slow Data Collection:**
- Reduce `max_stocks` parameter
- Use cached data when available
- Check internet connection

**Missing Data:**
- Some stocks may not have full history
- Fundamental data varies by company
- Filter for stocks with minimum data requirements

**Model Performance:**
- Market regime changes affect predictions
- Retrain periodically with fresh data
- Consider ensemble of models

## Testing

Test data availability for specific stocks:
```bash
python test_data_availability.py
```

## License

This project is for educational and research purposes only. Not financial advice.

## Disclaimer

This tool is for informational purposes only. Do not use it as the sole basis for investment decisions. Always conduct your own research and consult with financial advisors before investing.

---

**Author**: Stock Screening ML Project
**Version**: 1.0
**Last Updated**: November 2025
