"""
Configuration file for Indian Stock Screener
Adjust these parameters to customize the behavior
"""

# Data Collection Settings
DATA_CONFIG = {
    'default_lookback_years': 5,  # 5 years gives 40-60% fundamental coverage vs 15% with 10 years
    'max_stocks_default': 500,     # Default maximum stocks to process (increased for robustness)
    'cache_data': True,            # Whether to cache collected data
    'data_cache_file': 'stock_data_cache.pkl',
    'use_quarterly_fundamentals': True,  # Extract and use quarterly fundamental data
    'fundamental_lookback_quarters': 20,  # Number of quarters to extract (5 years = 20 quarters)
    'use_quality_filters': True,   # Apply quality filters when selecting stocks
    'min_market_cap': 1e9,         # Minimum market cap (1 billion INR)
    'min_avg_volume': 100000,      # Minimum average daily volume
    'min_data_months': 48          # Minimum months of historical data (4 years)
}

# Model Training Settings
MODEL_CONFIG = {
    'test_size': 0.2,              # Proportion of data for testing (0.2 = 20%)
    'random_state': 42,            # Random seed for reproducibility
    'use_time_series_split': True, # Use time-aware split (recommended)
    'model_cache_file': 'trained_model.pkl'
}

# LightGBM Hyperparameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'num_boost_round': 1000,
    'early_stopping_rounds': 50
}

# Feature Engineering Settings
TECHNICAL_INDICATORS = {
    'sma_periods': [20, 50, 200],  # Simple moving average periods
    'ema_periods': [12, 26],        # Exponential moving average periods
    'rsi_period': 14,                # RSI calculation period
    'bb_period': 20,                 # Bollinger Bands period
    'atr_period': 14,                # Average True Range period
    'volume_sma_period': 20,         # Volume SMA period
    'momentum_periods': [21, 63, 126, 252]  # 1M, 3M, 6M, 12M in trading days
}

# Fundamental Metrics to Extract
FUNDAMENTAL_METRICS = [
    # Valuation
    'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE', 'pegRatio',
    'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseToRevenue',
    'enterpriseToEbitda',

    # Profitability
    'profitMargins', 'operatingMargins', 'grossMargins',
    'returnOnEquity', 'returnOnAssets',

    # Growth
    'revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth',

    # Financial Health
    'totalCash', 'totalDebt', 'debtToEquity', 'currentRatio', 'quickRatio',

    # Cash Flow
    'operatingCashflow', 'freeCashflow',

    # Dividends
    'dividendYield', 'payoutRatio',

    # Risk
    'beta'
]

# Scoring Settings
SCORING_CONFIG = {
    'top_features_for_scoring': 15,  # Number of top features to use for percentile scoring
    'equal_weights': True,            # Use equal weights for composite score
    'custom_weights': None,           # Dict of feature: weight (if not using equal weights)
    'results_cache_file': 'screening_results.pkl'
}

# Features where LOWER is BETTER (will invert percentile)
INVERT_FEATURES = [
    'trailing_pe', 'forward_pe', 'debt_to_equity',
    'Volatility_20', 'Volatility_60', 'peg_ratio'
]

# Stock Universe Files
UNIVERSE_FILES = {
    'primary': 'NSE_Universe.csv',
    'backup': 'indian_stocks_tickers.csv'
}

# Streamlit UI Settings
UI_CONFIG = {
    'page_title': 'Indian Stock Screener - ML Powered',
    'page_icon': '📈',
    'layout': 'wide',
    'default_quintile_filter': ['Q4 (High)', 'Q5 (Highest)'],
    'min_composite_score': 0,
    'min_return_probability': 0.5
}

# Display Columns for Results Table
DISPLAY_COLUMNS = [
    'Ticker',
    'quality_quintile',
    'composite_score',
    'return_probability',
    'trailing_pe',
    'price_to_book',
    'roe',
    'revenue_growth',
    'debt_to_equity',
    'dividend_yield',
    'beta',
    'ROC_12M',
    'RSI',
    'Volatility_20'
]

# Column Formatting
COLUMN_FORMATS = {
    'composite_score': '{:.1f}',
    'return_probability': '{:.2%}',
    'trailing_pe': '{:.2f}',
    'price_to_book': '{:.2f}',
    'roe': '{:.2%}',
    'revenue_growth': '{:.2%}',
    'debt_to_equity': '{:.2f}',
    'dividend_yield': '{:.2%}',
    'beta': '{:.2f}',
    'ROC_12M': '{:.2%}',
    'RSI': '{:.1f}',
    'Volatility_20': '{:.2%}'
}

# Data Quality Filters (minimum requirements)
DATA_QUALITY_FILTERS = {
    'min_history_months': 12,        # Minimum months of price history
    'min_volume': 100000,            # Minimum average daily volume
    'min_market_cap': 1e9,           # Minimum market cap (1 billion)
    'require_fundamentals': True     # Whether to require fundamental data
}

# Advanced Settings
ADVANCED_CONFIG = {
    'handle_missing_data': 'median',  # 'median', 'mean', 'drop', or 'forward_fill'
    'remove_outliers': True,          # Remove statistical outliers
    'outlier_std_threshold': 3,       # Standard deviations for outlier detection
    'normalize_features': False,      # Whether to normalize features (not needed for tree-based models)
    'save_intermediate_results': True # Save intermediate processing results
}

# Logging Settings
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'stock_screener.log',
    'enable_logging': True
}
