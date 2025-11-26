"""
LightGBM Model Trainer - Screener.in Database Version

Trains model to predict next quarter returns using rich fundamental data
from screener.in database.

Key improvements:
- 10 years of quarterly fundamental data
- 500 random stocks for training
- Rich features: growth rates, margins, quality metrics, trends
- Predicts next quarter returns (aligned with quarterly reporting)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from typing import Tuple, List, Dict
import random
from data_collector_screener import ScreenerDataCollector
from price_data_helper import PriceDataHelper
from price_technical_indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')


class ScreenerModelTrainer:
    """Train LightGBM model using screener.in database + price-based technical indicators"""

    def __init__(self, db_path: str = 'screener_data.db'):
        """
        Initialize trainer

        Args:
            db_path: Path to screener.in SQLite database
        """
        self.db_path = db_path
        self.collector = ScreenerDataCollector(db_path)
        self.price_helper = PriceDataHelper()
        self.technical_indicators = TechnicalIndicators()
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def prepare_training_data(self, n_stocks: int = 500,
                             lookback_quarters: int = 10,
                             random_seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training dataset

        Args:
            n_stocks: Number of stocks to use for training (default: 500)
            lookback_quarters: Number of quarters per stock (default: 10)
            random_seed: Random seed for reproducibility

        Returns:
            X: Features DataFrame
            y: Target variable (next quarter returns)
        """
        print(f"\n{'='*80}")
        print("PREPARE TRAINING DATA FROM SCREENER.IN DATABASE")
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

        # Collect fundamental features
        training_data = self.collector.prepare_training_data(
            tickers=selected_stocks,
            lookback_quarters=lookback_quarters
        )

        if training_data.empty:
            raise ValueError("No training data collected!")

        print(f"\n[1] Fundamental features collected:")
        print(f"    Total quarters: {len(training_data)}")
        print(f"    Features: {len(training_data.columns)}")

        # Add price-based technical indicators
        print(f"\n[2] Adding price-based technical indicators (momentum, RSI, volatility)...")

        unique_stocks = training_data['ticker'].unique()
        print(f"    Total stocks to fetch price data for: {len(unique_stocks)}")

        all_technical_indicators = []

        for i, ticker in enumerate(unique_stocks, 1):
            if i % 50 == 0:
                print(f"    Processing {i}/{len(unique_stocks)}...")

            try:
                # Get this stock's data
                stock_data = training_data[training_data['ticker'] == ticker].copy()

                # Calculate technical indicators
                indicators = self.technical_indicators.calculate_indicators(
                    ticker, stock_data['quarter_date']
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
        training_data = pd.concat([training_data.reset_index(drop=True), technical_df.reset_index(drop=True)], axis=1)

        print(f"    Technical indicators added: {len(technical_df.columns)}")
        print(f"    Total features now: {len(training_data.columns)}")

        # Add price returns as target variable
        print(f"\n[3] Fetching price data for returns calculation (target variable)...")

        all_returns = []
        unique_stocks = training_data['ticker'].unique()

        print(f"    Total stocks to fetch prices for: {len(unique_stocks)}")

        success_count = 0
        fail_count = 0

        for i, ticker in enumerate(unique_stocks, 1):
            if i % 50 == 0:
                print(f"    Processing {i}/{len(unique_stocks)}... (Success: {success_count}, Failed: {fail_count})")

            try:
                # Get this stock's data
                stock_data = training_data[training_data['ticker'] == ticker].copy()

                # Calculate returns
                returns = self.price_helper.get_quarterly_returns(
                    ticker, stock_data['quarter_date']
                )

                all_returns.extend(returns.tolist())

                # Check if any valid returns (not all NaN)
                if not all(pd.isna(r) for r in returns):
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                print(f"    [WARNING] Error fetching prices for {ticker}: {str(e)}")
                # Get this stock's data
                stock_data = training_data[training_data['ticker'] == ticker].copy()
                # Add NaN for this stock's returns
                all_returns.extend([np.nan] * len(stock_data))
                fail_count += 1

        print(f"    Completed price data fetch:")
        print(f"      Total stocks: {len(unique_stocks)}")
        print(f"      Successful: {success_count} ({success_count/len(unique_stocks)*100:.1f}%)")
        print(f"      Failed: {fail_count} ({fail_count/len(unique_stocks)*100:.1f}%)")

        # Add returns as column
        training_data['next_quarter_return'] = all_returns

        # Remove rows where we couldn't calculate returns
        before_drop = len(training_data)
        training_data = training_data.dropna(subset=['next_quarter_return'])
        after_drop = len(training_data)

        print(f"\n[4] Returns calculated:")
        print(f"    Valid return data: {after_drop}/{before_drop} quarters")
        print(f"    Dropped: {before_drop - after_drop} quarters (no price data)")

        if len(training_data) < 100:
            raise ValueError(f"Insufficient data for training! Only {len(training_data)} quarters with valid returns. Need at least 100.")

        # Print statistics about the returns
        print(f"\n[5] Target variable statistics:")
        print(f"    Mean return: {training_data['next_quarter_return'].mean():.2f}%")
        print(f"    Std return: {training_data['next_quarter_return'].std():.2f}%")
        print(f"    Min return: {training_data['next_quarter_return'].min():.2f}%")
        print(f"    Max return: {training_data['next_quarter_return'].max():.2f}%")
        print(f"    Median return: {training_data['next_quarter_return'].median():.2f}%")

        # Separate features and target
        exclude_cols = ['ticker', 'quarter_date', 'id', 'scraped_at', 'next_quarter_return']
        feature_cols = [col for col in training_data.columns if col not in exclude_cols]

        X = training_data[feature_cols].copy()
        y = training_data['next_quarter_return'].copy()

        # Store feature names
        self.feature_names = feature_cols

        print(f"\n[6] Final dataset prepared:")
        print(f"    Training samples: {len(X)}")
        print(f"    Features: {len(self.feature_names)} (fundamentals + technical indicators)")
        print(f"    Sample features: {', '.join(self.feature_names[:15])}...")

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train LightGBM model

        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            random_state: Random seed

        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*80}")
        print("TRAIN LIGHTGBM MODEL")
        print(f"{'='*80}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=self.feature_names)

        # Parameters optimized for regression
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

        # Calculate metrics
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

        # Return metrics
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'n_features': len(self.feature_names),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        return metrics

    def save_model(self, model_path: str = 'trained_model_screener.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save! Train model first.")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n[OK] Model saved to: {model_path}")

    def load_model(self, model_path: str = 'trained_model_screener.pkl'):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']

        print(f"[OK] Model loaded from: {model_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("No model loaded! Train or load model first.")

        # Ensure features match training
        X = X[self.feature_names]

        return self.model.predict(X, num_iteration=self.model.best_iteration)


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("LIGHTGBM MODEL TRAINER - SCREENER.IN DATABASE VERSION")
    print("="*80)

    # Initialize trainer
    trainer = ScreenerModelTrainer(db_path='screener_data.db')

    # Prepare data
    X, y = trainer.prepare_training_data(
        n_stocks=500,  # 500 random stocks
        lookback_quarters=10,  # 10 quarters (2.5 years)
        random_seed=42
    )

    # Train model
    metrics = trainer.train_model(X, y, test_size=0.2, random_state=42)

    # Save model
    trainer.save_model('trained_model_screener.pkl')

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal R² (Test): {metrics['test_r2']:.4f}")
    print(f"Model saved: trained_model_screener.pkl")
    print("\nNext steps:")
    print("  1. Review feature importance")
    print("  2. Run screening workflow")
    print("  3. Analyze predictions")


if __name__ == "__main__":
    main()
