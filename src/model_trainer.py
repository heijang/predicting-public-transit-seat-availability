"""
Model Training for RQ1: Multi-Horizon Prediction

This module trains RandomForest models for different prediction horizons
and compares performance with/without external factors.

Usage:
    # Train single horizon
    python -m src.model_trainer --horizon 30min --features base
    
    # Train all horizons (default)
    python -m src.model_trainer
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_engineer import prepare_modeling_data


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train RandomForest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        random_state: Random seed
    
    Returns:
        Trained model
    """
    print(f"\n  Training RandomForest ({n_estimators} trees)...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    print(f"  ✓ Training completed")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Returns:
        dict: Performance metrics
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'samples': len(y_test)
    }
    
    print(f"\n  Evaluation Results:")
    print(f"    MAE:  {mae:.3f}")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    R²:   {r2:.3f}")
    
    return metrics


def train_horizon(data_path, horizon, feature_set='base', test_size=0.2, random_state=42):
    """
    Train model for a specific horizon and feature set.
    
    Args:
        data_path: Path to engineered data CSV
        horizon: Target horizon ('30min', '1h', '3h', 'next_day')
        feature_set: 'base' or 'extended'
        test_size: Test set ratio
        random_state: Random seed
    
    Returns:
        dict: Training results and metrics
    """
    print("=" * 70)
    print(f"TRAINING: {horizon} horizon | {feature_set} features")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['trip_start_date', 'trip_dep_time', 'trip_arr_time'])
    
    # Prepare X, y
    X, y, features = prepare_modeling_data(df, horizon=horizon, feature_set=feature_set)
    
    # Train/test split (time-based would be better, but random for simplicity)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n  Train samples: {len(X_train):,}")
    print(f"  Test samples:  {len(X_test):,}")
    
    # Train
    model = train_model(X_train, y_train, random_state=random_state)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 3 Important Features:")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_dir = Path('outputs/rq1_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'rf_{horizon}_{feature_set}.joblib'
    joblib.dump(model, model_path)
    print(f"\n  ✓ Model saved: {model_path}")
    
    return {
        'horizon': horizon,
        'feature_set': feature_set,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_path': str(model_path)
    }


def train_all_horizons(data_path='data/processed/re5_with_targets.csv', 
                       feature_sets=['base']):
    """
    Train models for all horizons and feature sets.
    
    Returns:
        pd.DataFrame: Results summary
    """
    horizons = ['30min', '1h', '3h', 'next_day']
    
    results = []
    
    for horizon in horizons:
        for feature_set in feature_sets:
            try:
                result = train_horizon(data_path, horizon, feature_set)
                results.append({
                    'horizon': horizon,
                    'feature_set': feature_set,
                    'MAE': result['metrics']['MAE'],
                    'RMSE': result['metrics']['RMSE'],
                    'R²': result['metrics']['R²'],
                    'samples': result['metrics']['samples']
                })
            except Exception as e:
                print(f"\n  ✗ Error training {horizon} {feature_set}: {e}")
                continue
    
    return pd.DataFrame(results)


def main():
    """Main execution with argument parsing."""
    parser = argparse.ArgumentParser(description='Train RQ1 models')
    parser.add_argument('--horizon', type=str, default='all',
                       choices=['30min', '1h', '3h', 'next_day', 'all'],
                       help='Prediction horizon')
    parser.add_argument('--features', type=str, default='base',
                       choices=['base', 'extended', 'both'],
                       help='Feature set')
    parser.add_argument('--data', type=str, default='data/processed/re5_with_targets.csv',
                       help='Path to engineered data')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ1 MODEL TRAINING")
    print("=" * 70)
    
    if args.horizon == 'all':
        # Train all horizons
        feature_sets = ['base'] if args.features == 'base' else ['base', 'extended']
        results_df = train_all_horizons(args.data, feature_sets)
        
        # Save results
        output_path = 'outputs/rq1_results/training_results.csv'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(results_df.to_string(index=False))
        print(f"\n✓ Results saved: {output_path}")
    else:
        # Train single horizon
        result = train_horizon(args.data, args.horizon, args.features)
        print(f"\n✓ Training completed for {args.horizon}")


if __name__ == '__main__':
    main()
