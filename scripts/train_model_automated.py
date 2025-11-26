"""
Automated Model Training Script

Trains LightGBM model on collected training data.
Runs via GitHub Actions monthly.
Model is committed back to repository for use in screening.
"""

import os
import sys
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_trainer import ReturnDirectionModel


def train_model():
    """
    Train LightGBM model on training data

    Uses:
    - Data from stock_data_cache.pkl
    - Time-series split (80/20)
    - Saves to trained_model.pkl
    """

    print("="*80)
    print("MODEL TRAINING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Load training data
    print("\n[1/3] Loading training data...")
    cache_file = 'stock_data_cache.pkl'

    if not os.path.exists(cache_file):
        print(f"❌ ERROR: Training data not found: {cache_file}")
        print("Run collect_training_data.py first!")
        sys.exit(1)

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    print(f"✅ Loaded {len(data)} rows for {data['Ticker'].nunique()} stocks")
    print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Step 2: Train model
    print("\n[2/3] Training LightGBM model...")
    print("⏱️  This will take 2-3 minutes...")

    model = ReturnDirectionModel(test_size=0.2, random_state=42)
    metrics = model.train_model(data, use_time_series_split=True)

    # Step 3: Save model
    print("\n[3/3] Saving model...")
    model_file = 'trained_model.pkl'
    model.save_model(model_file)

    # Create training log
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL TRAINING LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Stocks: {data['Ticker'].nunique()}\n")
        f.write(f"Training Samples: {model.model_metadata.get('n_training_samples', 'N/A')}\n")
        f.write(f"Test Samples: {model.model_metadata.get('n_test_samples', 'N/A')}\n")
        f.write(f"Features: {model.model_metadata.get('n_features', 'N/A')}\n")
        f.write(f"Data Range: {model.model_metadata.get('data_date_range', 'N/A')}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 FEATURES BY IMPORTANCE\n")
        f.write("="*80 + "\n")
        top_features = model.get_feature_importance(top_n=20)
        for idx, row in top_features.iterrows():
            f.write(f"{row['feature']:<30} {row['importance']:>10.0f}\n")
        f.write("="*80 + "\n")

    print("\n" + "="*80)
    print("MODEL TRAINING SUMMARY")
    print("="*80)
    print(f"Model saved to: {model_file}")
    print(f"Training log: {log_file}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nTop 5 Features:")
    top5 = model.get_feature_importance(top_n=5)
    for idx, row in top5.iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.0f}")
    print("="*80)

    return True


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
