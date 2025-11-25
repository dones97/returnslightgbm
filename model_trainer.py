"""
LightGBM Model Training Module
Trains a gradient boosting model to predict monthly return direction
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from typing import List, Tuple, Dict
import pickle
import warnings
warnings.filterwarnings('ignore')


class ReturnDirectionModel:
    """LightGBM model for predicting monthly return direction"""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize model

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for modeling

        Args:
            df: Raw feature dataframe

        Returns:
            Tuple of (processed_dataframe, feature_names)
        """
        # Remove rows with NaN in target
        df = df.dropna(subset=['Return_Direction'])

        # Define feature columns (exclude target, date, ticker, and intermediate columns)
        exclude_cols = [
            'Date', 'Ticker', 'Return_Next_Month', 'Return_Direction',
            'Open', 'High', 'Low', 'Close', 'Volume',  # Raw price data
            'Dividends', 'Stock Splits',  # Event data
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',  # Keep derived ratios instead
            'BB_Middle', 'BB_Upper', 'BB_Lower',  # Keep BB_Width and BB_Position
            'High_52W', 'Low_52W',  # Keep Distance_from_High and Distance_from_Low
            'Volume_SMA_20'  # Keep Volume_Ratio
        ]

        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with median for each feature
        for col in feature_cols:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        return df, feature_cols

    def train_model(self, df: pd.DataFrame, use_time_series_split: bool = False) -> Dict:
        """
        Train LightGBM model

        Args:
            df: Feature dataframe with target variable
            use_time_series_split: Whether to use time series cross-validation

        Returns:
            Dictionary with training metrics and feature importance
        """
        # Prepare features
        df_processed, feature_cols = self.prepare_features(df)
        self.feature_names = feature_cols

        X = df_processed[feature_cols]
        y = df_processed['Return_Direction']

        # Train-test split
        if use_time_series_split:
            # For time series, split by date to avoid look-ahead bias
            df_processed = df_processed.sort_values('Date')
            split_idx = int(len(df_processed) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Standard train-test split (stratified)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

        # LightGBM parameters
        params = {
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
            'random_state': self.random_state,
            'verbose': -1
        }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train model
        print("Training LightGBM model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # Make predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

        print("\n" + "="*60)
        print("Model Performance Metrics")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("="*60)

        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive returns

        Args:
            df: Feature dataframe

        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        df_processed, _ = self.prepare_features(df)
        X = df_processed[self.feature_names]

        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)

        return predictions

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        return self.feature_importance.head(top_n)

    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']

        print(f"Model loaded from {filepath}")


class StockScorer:
    """
    Creates percentile scores for stock characteristics and assigns quality quintiles
    """

    def __init__(self, model: ReturnDirectionModel):
        """
        Initialize scorer

        Args:
            model: Trained ReturnDirectionModel
        """
        self.model = model

    def calculate_factor_scores(self, df: pd.DataFrame, features_to_score: List[str] = None) -> pd.DataFrame:
        """
        Calculate percentile scores for important features

        Args:
            df: Feature dataframe
            features_to_score: List of features to score (if None, uses top model features)

        Returns:
            DataFrame with percentile scores added
        """
        df = df.copy()

        if features_to_score is None:
            # Use top features from model
            top_features = self.model.get_feature_importance(top_n=15)
            features_to_score = top_features['feature'].tolist()

        # Filter to features that exist in dataframe
        features_to_score = [f for f in features_to_score if f in df.columns]

        # Calculate percentile rank for each feature
        for feature in features_to_score:
            # Handle features where higher is better vs lower is better
            # Assume higher is better for most features, but invert for specific ones
            invert_features = ['trailing_pe', 'forward_pe', 'debt_to_equity', 'Volatility_20', 'Volatility_60']

            if feature in invert_features:
                # For these features, lower is better, so invert the percentile
                df[f'{feature}_percentile'] = 100 - df[feature].rank(pct=True) * 100
            else:
                # Higher is better
                df[f'{feature}_percentile'] = df[feature].rank(pct=True) * 100

        return df

    def calculate_composite_score(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Calculate composite score from percentile scores

        Args:
            df: DataFrame with percentile scores
            weights: Dictionary of feature weights (if None, uses equal weights)

        Returns:
            DataFrame with composite score
        """
        df = df.copy()

        # Get all percentile columns
        percentile_cols = [col for col in df.columns if col.endswith('_percentile')]

        if not percentile_cols:
            raise ValueError("No percentile columns found. Run calculate_factor_scores() first.")

        if weights is None:
            # Equal weights
            weights = {col: 1.0 / len(percentile_cols) for col in percentile_cols}

        # Calculate weighted average
        df['composite_score'] = 0
        for col in percentile_cols:
            weight = weights.get(col, 1.0 / len(percentile_cols))
            df['composite_score'] += df[col].fillna(50) * weight  # Fill NaN with median (50th percentile)

        return df

    def assign_quintiles(self, df: pd.DataFrame, score_column: str = 'composite_score') -> pd.DataFrame:
        """
        Assign quality quintiles based on scores

        Args:
            df: DataFrame with scores
            score_column: Column name to use for quintile assignment

        Returns:
            DataFrame with quintile assignments
        """
        df = df.copy()

        # Assign quintiles (1 = bottom 20%, 5 = top 20%)
        df['quality_quintile'] = pd.qcut(
            df[score_column],
            q=5,
            labels=['Q1 (Lowest)', 'Q2 (Low)', 'Q3 (Medium)', 'Q4 (High)', 'Q5 (Highest)'],
            duplicates='drop'
        )

        return df

    def score_current_universe(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Score entire universe of stocks based on most recent data

        Args:
            current_data: DataFrame with current stock data

        Returns:
            DataFrame with scores and quintiles
        """
        # Get most recent data for each stock
        if 'Date' in current_data.columns:
            latest_data = current_data.sort_values('Date').groupby('Ticker').tail(1)
        else:
            latest_data = current_data.copy()

        # Get model predictions (probability of positive return)
        try:
            latest_data['return_probability'] = self.model.predict_proba(latest_data)
        except:
            latest_data['return_probability'] = np.nan

        # Calculate factor scores
        latest_data = self.calculate_factor_scores(latest_data)

        # Calculate composite score
        latest_data = self.calculate_composite_score(latest_data)

        # Assign quintiles
        latest_data = self.assign_quintiles(latest_data)

        return latest_data
