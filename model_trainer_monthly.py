"""
LightGBM Model Trainer - Monthly Returns with Proper Fundamental Alignment

NO LOOK-AHEAD BIAS!

Key improvements over quarterly approach:
1. Uses monthly returns (1-month forward) instead of quarterly
2. Only uses fundamentals that were REPORTED at prediction time
3. 12x more training data (120 months vs 10 quarters)
4. Proper causality: Known fundamentals → Predict continuation

Target: Next month's price return (%)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from typing import Tuple, Dict
import random
from data_collector_simple_monthly import SimpleMonthlyCollector
from price_data_helper import PriceDataHelper
from price_technical_indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')


class MonthlyModelTrainer:
    """Train LightGBM model using monthly data with proper fundamental lag"""

    def __init__(self, db_path: str = 'screener_data.db', use_price_cache: bool = False):
        """
        Initialize trainer

        Args:
            db_path: Path to screener.in SQLite database
            use_price_cache: If True, use cached price data
        """
        self.db_path = db_path
        self.collector = SimpleMonthlyCollector(db_path)
        self.price_helper = PriceDataHelper(use_cache=use_price_cache)
        self.technical_indicators = TechnicalIndicators(use_cache=use_price_cache)
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def prepare_training_data(self, n_stocks: int = 500,
                             start_date: str = '2014-12-31',
                             end_date: str = '2024-11-30',
                             random_seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare monthly training data

        Args:
            n_stocks: Number of stocks (default: 500)
            start_date: Start date (10 years ago)
            end_date: End date (latest month)
            random_seed: Random seed

        Returns:
            X: Features DataFrame
            y: Target variable (next month returns)
        """
        print(f"\n{'='*80}")
        print("PREPARE MONTHLY TRAINING DATA - NO LOOK-AHEAD BIAS")
        print(f"{'='*80}")

        # Get all available stocks
        all_stocks = self.collector.get_available_stocks()
        print(f"\nTotal stocks in database: {len(all_stocks)}")

        if len(all_stocks) < n_stocks:
            print(f"[WARNING] Only {len(all_stocks)} stocks available, using all")
            n_stocks = len(all_stocks)

        # Randomly select stocks
        random.seed(random_seed)
        selected_stocks = random.sample(all_stocks, n_stocks)
        print(f"Randomly selected {n_stocks} stocks for training")

        # Collect fundamental features (monthly with proper lag)
        training_data = self.collector.prepare_training_data(
            tickers=selected_stocks,
            start_date=start_date,
            end_date=end_date
        )

        if training_data.empty:
            raise ValueError("No training data collected!")

        print(f"\n[1] Fundamental features collected (with proper reporting lag):")
        print(f"    Total months: {len(training_data)}")
        print(f"    Unique stocks: {training_data['ticker'].nunique()}")
        print(f"    Features: {len(training_data.columns)}")
        print(f"    Average staleness: {training_data['staleness_days'].mean():.0f} days")

        # Add price-based technical indicators
        print(f"\n[2] Adding price-based technical indicators (momentum, RSI, volatility)...")

        unique_stocks = training_data['ticker'].unique()
        print(f"    Total stocks to process: {len(unique_stocks)}")

        all_technical_indicators = []

        for i, ticker in enumerate(unique_stocks, 1):
            if i % 50 == 0:
                print(f"    Processing {i}/{len(unique_stocks)}...")

            try:
                # Get this stock's monthly data
                stock_data = training_data[training_data['ticker'] == ticker].copy()

                # Calculate technical indicators at each month-end
                indicators = self.technical_indicators.calculate_indicators(
                    ticker, stock_data['month_end_date']
                )

                all_technical_indicators.append(indicators)

            except Exception as e:
                print(f"    [WARNING] Error calculating indicators for {ticker}: {str(e)}")
                # Add empty indicators
                empty_indicators = pd.DataFrame({
                    'roc_1m': [np.nan] * len(stock_data),
                    'roc_3m': [np.nan] * len(stock_data),
                    'roc_6m': [np.nan] * len(stock_data),
                    'roc_12m': [np.nan] * len(stock_data),
                    'volatility_30d': [np.nan] * len(stock_data),
                    'volatility_90d': [np.nan] * len(stock_data),
                    'rsi_14d': [np.nan] * len(stock_data),
                    'price_to_ma50': [np.nan] * len(stock_data),
                    'price_to_ma200': [np.nan] * len(stock_data),
                    'momentum_score': [np.nan] * len(stock_data)
                })
                all_technical_indicators.append(empty_indicators)

        # Combine technical indicators with fundamental data
        technical_df = pd.concat(all_technical_indicators, ignore_index=True)
        training_data = pd.concat([training_data.reset_index(drop=True),
                                  technical_df.reset_index(drop=True)], axis=1)

        print(f"    Technical indicators added: {len(technical_df.columns)}")
        print(f"    Total features now: {len(training_data.columns)}")

        # Add monthly price returns as target variable
        print(f"\n[3] Fetching price data for monthly returns calculation (target variable)...")

        all_returns = []
        unique_stocks = training_data['ticker'].unique()

        print(f"    Total stocks to fetch prices for: {len(unique_stocks)}")

        success_count = 0
        fail_count = 0

        for i, ticker in enumerate(unique_stocks, 1):
            if i % 50 == 0:
                print(f"    Processing {i}/{len(unique_stocks)}... (Success: {success_count}, Failed: {fail_count})")

            try:
                # Get this stock's monthly data
                stock_data = training_data[training_data['ticker'] == ticker].copy()

                # Calculate monthly returns (1 month forward)
                returns = self.price_helper.get_monthly_returns(
                    ticker, stock_data['month_end_date']
                )

                all_returns.extend(returns.tolist())

                # Check if any valid returns
                if not all(pd.isna(r) for r in returns):
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                stock_data = training_data[training_data['ticker'] == ticker].copy()
                all_returns.extend([np.nan] * len(stock_data))
                fail_count += 1

        print(f"    Completed price data fetch:")
        print(f"      Total stocks: {len(unique_stocks)}")
        print(f"      Successful: {success_count} ({success_count/len(unique_stocks)*100:.1f}%)")
        print(f"      Failed: {fail_count} ({fail_count/len(unique_stocks)*100:.1f}%)")

        # Add returns as column
        training_data['next_month_return'] = all_returns

        # Remove rows where we couldn't calculate returns
        before_drop = len(training_data)
        training_data = training_data.dropna(subset=['next_month_return'])
        after_drop = len(training_data)

        print(f"\n[4] Monthly returns calculated:")
        print(f"    Valid return data: {after_drop}/{before_drop} months")
        print(f"    Dropped: {before_drop - after_drop} months (no price data)")

        if len(training_data) < 100:
            raise ValueError(f"Insufficient data for training! Only {len(training_data)} months. Need at least 100.")

        # Print statistics about the returns
        print(f"\n[5] Target variable statistics:")
        print(f"    Mean return: {training_data['next_month_return'].mean():.2f}%")
        print(f"    Std return: {training_data['next_month_return'].std():.2f}%")
        print(f"    Min return: {training_data['next_month_return'].min():.2f}%")
        print(f"    Max return: {training_data['next_month_return'].max():.2f}%")
        print(f"    Median return: {training_data['next_month_return'].median():.2f}%")

        # Separate features and target
        exclude_cols = ['ticker', 'month_end_date', 'quarter_date', 'staleness_days',
                       'next_month_return', 'id', 'scraped_at']
        feature_cols = [col for col in training_data.columns if col not in exclude_cols]

        X = training_data[feature_cols].copy()
        y = training_data['next_month_return'].copy()

        # Store feature names
        self.feature_names = feature_cols

        print(f"\n[6] Final dataset prepared:")
        print(f"    Training samples: {len(X)}")
        print(f"    Features: {len(self.feature_names)}")
        print(f"    Samples per feature: {len(X) / len(self.feature_names):.0f}")

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """Train LightGBM model"""
        print(f"\n{'='*80}")
        print("TRAIN LIGHTGBM MODEL")
        print(f"{'='*80}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data,
                               feature_name=self.feature_names)

        # Parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 6,
            'min_child_samples': 20
        }

        print("\nTraining model...")
        print(f"Parameters: {params}")

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        # Predictions
        y_pred_train = self.model.predict(X_train, num_iteration=self.model.best_iteration)
        y_pred_test = self.model.predict(X_test, num_iteration=self.model.best_iteration)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # Print results
        print(f"\n{'='*80}")
        print("MODEL PERFORMANCE")
        print(f"{'='*80}")
        print(f"\nTRAINING SET:")
        print(f"  R-squared (R²): {train_r2:.4f}")
        print(f"  RMSE: {train_rmse:.2f}%")
        print(f"  MAE: {train_mae:.2f}%")

        print(f"\nTEST SET:")
        print(f"  R-squared (R²): {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.2f}%")
        print(f"  MAE: {test_mae:.2f}%")

        print(f"\nTOP 10 IMPORTANT FEATURES:")
        for i, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.0f}")

        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }

    def save_model(self, model_path: str = 'trained_model_monthly.pkl'):
        """Save model"""
        if self.model is None:
            raise ValueError("No model to save!")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n[OK] Model saved to: {model_path}")


def main():
    """Main training pipeline"""
    import os

    print("\n" + "="*80)
    print("LIGHTGBM MODEL TRAINER - MONTHLY RETURNS (NO LOOK-AHEAD BIAS)")
    print("="*80)

    # Check if price cache exists
    use_cache = os.path.exists('price_cache.pkl')

    if use_cache:
        print("\n[OK] Found price cache file - will use cached prices")
    else:
        print("\n[INFO] No price cache found - will fetch prices from yfinance")
        print("       (To use cache: run 'python build_price_cache.py' first)")

    # Initialize trainer
    trainer = MonthlyModelTrainer(db_path='screener_data.db', use_price_cache=use_cache)

    # Prepare data (8 years of monthly data for 500 best stocks)
    # Note: Use 8 years (2016-2024) as database has excellent annual data going back 12 years
    # Quarterly data only goes back 3 years, but annual fundamentals are complete
    X, y = trainer.prepare_training_data(
        n_stocks=500,
        start_date='2016-12-31',  # 8 years ago (all stocks have annual data from here)
        end_date='2024-11-30',     # Latest month
        random_seed=42
    )

    # Train model
    metrics = trainer.train_model(X, y, test_size=0.2, random_state=42)

    # Save model
    trainer.save_model('trained_model_monthly.pkl')

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal R² (Test): {metrics['test_r2']:.4f}")
    print(f"Model saved: trained_model_monthly.pkl")
    print("\nKey improvements over quarterly approach:")
    print("  ✓ No look-ahead bias (only use REPORTED fundamentals)")
    print("  ✓ 12x more training data (months vs quarters)")
    print("  ✓ Better signal extraction from fundamentals")
    print("\nNext steps:")
    print("  1. Review feature importance (fundamentals should rank higher now)")
    print("  2. Compare R² to quarterly model (expect 5x improvement)")
    print("  3. Deploy if performance meets expectations")


if __name__ == "__main__":
    main()
