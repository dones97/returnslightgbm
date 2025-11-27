"""
Classification Model - Predict Monthly Return Direction

Strategy: Use screener.in fundamentals + yfinance technicals to predict UP/DOWN

Key Features:
- Binary classification (UP vs DOWN next month)
- Size-agnostic fundamental ratios from screener.in
- Technical indicators from yfinance
- 3 years of data (2022-2024)
- 45-day reporting lag for fundamentals
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import pickle
from typing import Tuple, Dict
from data_collector_simple_monthly import SimpleMonthlyCollector
from price_technical_indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')


class MonthlyDirectionClassifier:
    """LightGBM classifier for monthly return direction using fundamentals + technicals"""

    def __init__(self, db_path: str = 'screener_data.db', use_price_cache: bool = True):
        self.db_path = db_path
        self.fundamental_collector = SimpleMonthlyCollector(db_path)
        self.technical_collector = TechnicalIndicators(use_cache=use_price_cache)
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def prepare_training_data(self, n_stocks: int = 500,
                             start_date: str = '2022-01-31',
                             end_date: str = '2024-10-31',
                             random_seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with fundamentals + technicals + return direction

        Args:
            n_stocks: Number of stocks to use
            start_date: Start date for monthly data
            end_date: End date (one month before latest to allow forward return)
            random_seed: Random seed

        Returns:
            X: Features DataFrame
            y: Target (1=UP, 0=DOWN)
        """
        print(f"\n{'='*80}")
        print("PREPARE CLASSIFICATION TRAINING DATA")
        print(f"{'='*80}")

        # Get available stocks
        all_stocks = self.fundamental_collector.get_available_stocks()
        print(f"\nTotal stocks in database: {len(all_stocks)}")

        # Randomly select stocks
        import random
        random.seed(random_seed)
        if len(all_stocks) > n_stocks:
            selected_stocks = random.sample(all_stocks, n_stocks)
        else:
            selected_stocks = all_stocks
            print(f"[WARNING] Only {len(all_stocks)} stocks available, using all")

        print(f"Selected {len(selected_stocks)} stocks")

        # Collect fundamental features
        print(f"\n[1] Collecting fundamental ratios (with 45-day reporting lag)...")
        fundamental_data = self.fundamental_collector.prepare_training_data(
            tickers=selected_stocks,
            start_date=start_date,
            end_date=end_date
        )

        if fundamental_data.empty:
            raise ValueError("No fundamental data collected!")

        print(f"\n[2] Adding technical indicators from yfinance...")

        # Add technical indicators for each stock-month
        all_data = []

        for i, ticker in enumerate(fundamental_data['ticker'].unique(), 1):
            ticker_data = fundamental_data[fundamental_data['ticker'] == ticker].copy()

            if ticker_data.empty:
                continue

            # Get technical indicators for this stock
            month_dates = pd.Series(ticker_data['month_end_date'].values)

            try:
                # Calculate all technical indicators
                tech_indicators = self.technical_collector.calculate_indicators(ticker, month_dates)

                # Add technical features to fundamental data
                for col in tech_indicators.columns:
                    ticker_data[col] = tech_indicators[col].values

                all_data.append(ticker_data)

                if i % 50 == 0:
                    print(f"  Processed {i}/{fundamental_data['ticker'].nunique()} stocks...")

            except Exception as e:
                print(f"  [WARNING] {ticker}: {str(e)[:60]}")
                continue

        if not all_data:
            raise ValueError("No data with technical indicators!")

        combined_data = pd.concat(all_data, ignore_index=True)

        print(f"\n[3] Calculating forward returns and direction...")

        # Calculate next month's return for each stock-month
        returns = []
        directions = []

        for ticker in combined_data['ticker'].unique():
            ticker_df = combined_data[combined_data['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('month_end_date').reset_index(drop=True)

            # Calculate 1-month forward returns using PriceDataHelper
            month_returns = self.technical_collector.price_helper.get_monthly_returns(
                ticker,
                pd.Series(ticker_df['month_end_date'].values),
                forward_months=1
            )

            ticker_df['forward_return'] = month_returns.values
            ticker_df['direction'] = (month_returns > 0).astype(int)  # 1=UP, 0=DOWN

            # Add to results
            for _, row in ticker_df.iterrows():
                returns.append(row['forward_return'])
                directions.append(row['direction'])

        combined_data['forward_return'] = returns
        combined_data['direction'] = directions

        # Remove rows with missing direction (last month for each stock)
        valid_data = combined_data.dropna(subset=['direction'])

        print(f"\n[4] Final dataset prepared")
        print(f"    Total samples: {len(valid_data)}")
        print(f"    UP months: {(valid_data['direction'] == 1).sum()} ({(valid_data['direction'] == 1).mean()*100:.1f}%)")
        print(f"    DOWN months: {(valid_data['direction'] == 0).sum()} ({(valid_data['direction'] == 0).mean()*100:.1f}%)")
        print(f"    Features: {len([c for c in valid_data.columns if c not in ['ticker', 'month_end_date', 'quarter_date', 'staleness_days', 'forward_return', 'direction']])}")

        # Prepare features and target
        feature_cols = [c for c in valid_data.columns
                       if c not in ['ticker', 'month_end_date', 'quarter_date',
                                   'staleness_days', 'forward_return', 'direction']]

        X = valid_data[feature_cols].copy()
        y = valid_data['direction'].copy()

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train LightGBM classification model

        Args:
            X: Features
            y: Target (1=UP, 0=DOWN)
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*80}")
        print("TRAIN LIGHTGBM CLASSIFIER")
        print(f"{'='*80}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTrain set: {len(X_train)} samples ({(y_train == 1).mean()*100:.1f}% UP)")
        print(f"Test set: {len(X_test)} samples ({(y_test == 1).mean()*100:.1f}% UP)")

        # Handle missing values
        print(f"\n[1] Handling missing values...")
        for col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)

        # Train LightGBM
        print(f"\n[2] Training LightGBM classifier...")

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'max_depth': 6
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        # Predictions
        y_train_pred_proba = self.model.predict(X_train)
        y_test_pred_proba = self.model.predict(X_test)

        y_train_pred = (y_train_pred_proba > 0.5).astype(int)
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)

        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)

        # Feature importance
        self.feature_names = X_train.columns.tolist()
        self.feature_importance = self.model.feature_importance(importance_type='gain')

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        print(f"\n{'='*80}")
        print("CLASSIFICATION MODEL PERFORMANCE")
        print(f"{'='*80}")
        print(f"\nTRAIN SET:")
        print(f"  Accuracy: {train_accuracy:.1%}")
        print(f"  AUC:      {train_auc:.4f}")
        print(f"\nTEST SET:")
        print(f"  Accuracy:  {test_accuracy:.1%}  <- % of correct predictions")
        print(f"  Precision: {test_precision:.1%}  <- When predict UP, % correct")
        print(f"  Recall:    {test_recall:.1%}  <- % of actual UPs caught")
        print(f"  F1 Score:  {test_f1:.4f}")
        print(f"  AUC-ROC:   {test_auc:.4f}  <- 0.5=random, 1.0=perfect")

        print(f"\nCONFUSION MATRIX:")
        print(f"                 Predicted DOWN  Predicted UP")
        print(f"  Actual DOWN         {cm[0,0]:4d}           {cm[0,1]:4d}")
        print(f"  Actual UP           {cm[1,0]:4d}           {cm[1,1]:4d}")

        print(f"\n{'='*80}")
        print("TOP 15 PREDICTIVE FEATURES")
        print(f"{'='*80}")
        for i, row in importance_df.head(15).iterrows():
            print(f"{i+1:2d}. {row['feature']:40s} {row['importance']:>12,.0f}")

        # Interpretation
        print(f"\n{'='*80}")
        print("INTERPRETATION:")
        print(f"{'='*80}")

        if test_accuracy > 0.60:
            print("✓✓✓ EXCELLENT - Accuracy >60% is rare in stock prediction!")
        elif test_accuracy > 0.55:
            print("✓✓ VERY GOOD - Accuracy >55% provides strong edge")
        elif test_accuracy > 0.52:
            print("✓ USABLE - Accuracy >52% beats random (50%)")
        else:
            print("⚠ WEAK - Accuracy ≤52% barely better than random")

        if test_auc > 0.65:
            print("✓✓✓ EXCELLENT - AUC >0.65 is strong signal!")
        elif test_auc > 0.57:
            print("✓✓ GOOD - AUC >0.57 provides usable signal")
        elif test_auc > 0.52:
            print("✓ WEAK - AUC >0.52 slight edge over random")
        else:
            print("⚠ NO SIGNAL - AUC ≤0.52 essentially random")

        print(f"\nBASELINE: Random guessing = 50% accuracy, 0.5 AUC")
        print(f"YOUR MODEL: {(test_accuracy - 0.5) * 100:.1f} percentage points above random")
        print(f"{'='*80}")

        # Save model
        print(f"\n[3] Saving model...")
        with open('model_classification.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("    Model saved to: model_classification.pkl")

        importance_df.to_csv('feature_importance_classification.csv', index=False)
        print("    Feature importance saved to: feature_importance_classification.csv")

        return {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'confusion_matrix': cm,
            'feature_importance': importance_df
        }


if __name__ == '__main__':
    print(f"\n{'#'*80}")
    print("MONTHLY DIRECTION CLASSIFIER - FUNDAMENTALS + TECHNICALS")
    print(f"{'#'*80}")

    # Initialize trainer
    trainer = MonthlyDirectionClassifier(use_price_cache=True)

    # Prepare data (500 stocks, 2022-2024)
    X, y = trainer.prepare_training_data(
        n_stocks=500,
        start_date='2022-01-31',
        end_date='2024-10-31',  # End Oct to allow Nov forward return
        random_seed=42
    )

    # Train model (80/20 split)
    metrics = trainer.train_model(X, y, test_size=0.2, random_state=42)

    print(f"\n{'#'*80}")
    print("TRAINING COMPLETE!")
    print(f"{'#'*80}")
