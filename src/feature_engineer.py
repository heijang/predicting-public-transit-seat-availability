import pandas as pd
import numpy as np
import holidays
from pathlib import Path

"""
Feature Engineering for RQ1: Multi-Horizon Seat Availability Prediction
(Stop-based time series approach)

This module creates target variables for different prediction horizons
based on stop-level time series (each stop has a sequence of trains).

Usage:
    python -m src.feature_engineer
"""

def load_processed_data(re5_path='data/processed/re5_processed.csv',
                       re6_path='data/processed/re6_processed.csv'):
    """Load processed RE5 and RE6 data."""
    
    re5 = pd.read_csv(re5_path, parse_dates=['trip_start_date', 'trip_dep_time', 'trip_arr_time'])
    re6 = pd.read_csv(re6_path, parse_dates=['trip_start_date', 'trip_dep_time', 'trip_arr_time'])
    
    return re5, re6


def add_holiday_flag(df, country='DE', prov='BB'):
    """
    Add is_holiday column using python-holidays.
    
    Germany (DE), Brandenburg (BB) for Potsdam/Berlin region.
    """
    if not np.issubdtype(df['trip_start_date'].dtype, np.datetime64):
        df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])

    years = sorted(df['trip_start_date'].dt.year.unique())
    de_holidays = holidays.country_holidays(country, subdiv=prov, years=years)

    df['is_holiday'] = df['trip_start_date'].dt.date.apply(lambda d: d in de_holidays)

    return df


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

    df = df.copy()
    df = df.sort_values(['dep_id', 'trip_dep_time']).reset_index(drop=True)
    
    for name, shift_periods in horizons.items():
        print(f"  Processing {name} ({shift_periods} trains ahead)...")
        df[f'occ_{name}_ahead'] = df.groupby('dep_id')['occ_pct'].shift(-shift_periods)
    
    # Keep only rows with at least one valid target
    target_cols = [f'occ_{name}_ahead' for name in horizons.keys()]
    df_clean = df.dropna(subset=target_cols, how='all')

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

    # external factors
    external_flags = [
        'is_match_day',
        'is_holiday',
    ]
    
    if feature_set == 'extended':
        return base_features + external_flags
    
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
    
    return X, y, features


def save_engineered_data(df, output_path='data/processed/re5_with_targets.csv'):
    """Save engineered data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    """Main execution."""
    # Load
    re5, re6 = load_processed_data()
    
    # Add Holiday flag
    re5 = add_holiday_flag(re5)

    # Create targets
    re5_targets = create_horizon_targets(re5)
    
    # Save
    save_engineered_data(re5_targets)
    
    # Sample
    sample_stop = re5_targets['dep_id'].iloc[0]
    sample = re5_targets[re5_targets['dep_id'] == sample_stop].head(10)


if __name__ == '__main__':
    main()
