"""
Feature Engineering for RQ1: Multi-Horizon Seat Availability Prediction
(Stop-based time series approach)

This module creates target variables for different prediction horizons
based on stop-level time series (each stop has a sequence of trains).

Usage:
    python -m src.feature_engineer
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_processed_data(re5_path='data/processed/re5_processed.csv',
                       re6_path='data/processed/re6_processed.csv'):
    """Load processed RE5 and RE6 data."""
    print("Loading processed data...")
    
    re5 = pd.read_csv(re5_path, parse_dates=['trip_start_date', 'trip_dep_time', 'trip_arr_time'])
    re6 = pd.read_csv(re6_path, parse_dates=['trip_start_date', 'trip_dep_time', 'trip_arr_time'])
    
    print(f"  ✓ RE5: {len(re5):,} rows")
    print(f"  ✓ RE6: {len(re6):,} rows")
    
    return re5, re6


def create_horizon_targets(df, horizons={'30min': 1, '1h': 2, '3h': 6, 'next_day': 48}):
    """
    Create target variables based on stop-level time series.
    
    Each stop (dep_id) has a sequence of trains arriving over time.
    We predict the occupancy of future trains at the same stop.
    
    Args:
        df: Input dataframe
        horizons: Dict of {name: number_of_trains_ahead}
                  Assuming ~30min average interval between trains
                  - 30min: next train (shift -1)
                  - 1h: 2nd train (shift -2)
                  - 3h: 6th train (shift -6)
                  - next_day: 48th train (shift -48)
    
    Returns:
        DataFrame with target columns
    """
    print("\nCreating stop-based horizon targets...")
    print("  Approach: Each stop's train sequence (dep_id + trip_dep_time)")
    
    df = df.copy()
    df = df.sort_values(['dep_id', 'trip_dep_time']).reset_index(drop=True)
    
    for name, shift_periods in horizons.items():
        print(f"  Processing {name} ({shift_periods} trains ahead)...")
        df[f'occ_{name}_ahead'] = df.groupby('dep_id')['occ_pct'].shift(-shift_periods)
    
    # Keep only rows with at least one valid target
    target_cols = [f'occ_{name}_ahead' for name in horizons.keys()]
    df_clean = df.dropna(subset=target_cols, how='all')
    
    print(f"\n  ✓ Original rows: {len(df):,}")
    print(f"  ✓ Rows with valid targets: {len(df_clean):,}")
    print(f"  ✓ Dropped: {len(df) - len(df_clean):,}")
    
    return df_clean


def select_features(df, feature_set='base'):
    """
    Select feature columns.
    
    Args:
        df: Input dataframe
        feature_set: 'base' or 'extended'
    
    Returns:
        list: Feature column names
    """
    base_features = [
        'hour',
        'dow',
        'is_rush_hour',
        'is_weekend',
        'is_night',
        'is_match_day',
        'occ_pct',  # current occupancy at this stop
    ]
    
    if feature_set == 'extended':
        # TODO: Add external factors later
        return base_features
    
    return base_features


def prepare_modeling_data(df, horizon='30min', feature_set='base'):
    """
    Prepare X, y for modeling.
    
    Args:
        df: Dataframe with features and targets
        horizon: Target horizon
        feature_set: 'base' or 'extended'
    
    Returns:
        tuple: (X, y, feature_names)
    """
    features = select_features(df, feature_set)
    target = f'occ_{horizon}_ahead'
    
    df_clean = df.dropna(subset=[target] + features)
    
    X = df_clean[features]
    y = df_clean[target]
    
    print(f"\nPrepared data for {horizon} ({feature_set}):")
    print(f"  Features: {features}")
    print(f"  Target: {target}")
    print(f"  Samples: {len(X):,}")
    
    return X, y, features


def save_engineered_data(df, output_path='data/processed/re5_with_targets.csv'):
    """Save engineered data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("FEATURE ENGINEERING FOR RQ1 (Stop-based)")
    print("=" * 70)
    
    # Load
    re5, re6 = load_processed_data()
    
    # Create targets
    re5_targets = create_horizon_targets(re5)
    
    # Save
    save_engineered_data(re5_targets)
    
    # Sample
    print("\nSample (first stop's train sequence):")
    sample_stop = re5_targets['dep_id'].iloc[0]
    sample = re5_targets[re5_targets['dep_id'] == sample_stop].head(10)
    print(sample[['dep_id', 'trip_dep_time', 'occ_pct', 'occ_30min_ahead', 'occ_1h_ahead']])
    
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
