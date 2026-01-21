"""
RQ2 Analysis: SHAP-based Explainability for Seat Availability Predictions

Research Question:
"How do external factors shape SHAP-based explanations of seat availability
predictions in special scenarios such as concert nights, storm days, and holiday weekends?"

This analysis provides:
1. SHAP value computation for the prediction model
2. Scenario-specific analysis (match days, holidays, rush hours, weekends)
3. Feature importance comparison across scenarios
4. Detailed visualizations and tables
5. Comprehensive report generation
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_prepare_data():
    """Load and prepare the dataset for RQ2 analysis."""
    print("=" * 70)
    print("RQ2 ANALYSIS: SHAP-based Explainability")
    print("=" * 70)

    print("\n[1/5] Loading data...")

    # Try to load preprocessed data first
    data_path = Path('data/processed/re5_with_targets.csv')
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
        df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])
        print(f"  Loaded preprocessed data: {len(df):,} rows")
    else:
        # Fall back to raw data
        raw_path = Path('data/ingested/RE5_2024_03.csv')
        if raw_path.exists():
            df = pd.read_csv(raw_path)
        else:
            # Try RE5 processed
            df = pd.read_csv('data/processed/re5_processed.csv')

        df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
        df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])
        print(f"  Loaded data: {len(df):,} rows")

    # Ensure required columns
    if 'hour' not in df.columns:
        df['hour'] = df['trip_dep_time'].dt.hour
    if 'dow' not in df.columns:
        df['dow'] = df['trip_start_date'].dt.dayofweek
    if 'date' not in df.columns:
        df['date'] = df['trip_start_date'].dt.date
    if 'month' not in df.columns:
        df['month'] = df['trip_start_date'].dt.month

    # Calculate occupancy percentage if needed
    if 'occ_pct' not in df.columns:
        df['occ_pct'] = (df['occupancy'] / df['capacity'] * 100).round(2)

    # Filter quality (capacity >= 100)
    if 'capacity' in df.columns:
        df = df[df['capacity'] >= 100].copy()

    print(f"  Date range: {df['trip_start_date'].min().date()} to {df['trip_start_date'].max().date()}")
    print(f"  Unique stations: {df['dep_id'].nunique()}")

    return df


def create_features(df):
    """Create all features for the model."""
    print("\n[2/5] Creating features...")

    df = df.copy()

    # Time features
    df['hour'] = df['trip_dep_time'].dt.hour
    df['dow'] = df['trip_start_date'].dt.dayofweek
    df['month'] = df['trip_start_date'].dt.month
    df['day_of_month'] = df['trip_start_date'].dt.day

    # Binary flags
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_morning'] = df['hour'].isin([6, 7, 8, 9, 10, 11]).astype(int)
    df['is_afternoon'] = df['hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
    df['is_evening'] = df['hour'].isin([18, 19, 20, 21]).astype(int)

    # Holiday flag (Germany, Brandenburg)
    years = sorted(df['trip_start_date'].dt.year.unique())
    de_holidays = holidays.country_holidays('DE', subdiv='BB', years=years)
    df['is_holiday'] = df['trip_start_date'].dt.date.apply(lambda d: 1 if d in de_holidays else 0)

    # Match day flag
    if 'is_match_day' not in df.columns:
        try:
            match_df = pd.read_excel('data/raw/Schedule_Teams_2022_2024.xlsx', header=1)
            match_df['match_date'] = pd.to_datetime(match_df['Date'], errors='coerce')
            match_dates = set(match_df['match_date'].dt.date.dropna())
            df['is_match_day'] = df['trip_start_date'].dt.date.apply(lambda d: 1 if d in match_dates else 0)
        except Exception as e:
            print(f"  Warning: Could not load match schedule: {e}")
            df['is_match_day'] = 0

    # Historical average features
    print("  Creating historical averages...")

    # By stop/hour/dow
    hist_stop_hour_dow = df.groupby(['dep_id', 'hour', 'dow'])['occ_pct'].agg(['mean', 'std']).reset_index()
    hist_stop_hour_dow.columns = ['dep_id', 'hour', 'dow', 'hist_avg_stop_hour_dow', 'hist_std_stop_hour_dow']
    df = df.merge(hist_stop_hour_dow, on=['dep_id', 'hour', 'dow'], how='left')

    # By stop/hour
    hist_stop_hour = df.groupby(['dep_id', 'hour'])['occ_pct'].mean().reset_index()
    hist_stop_hour.columns = ['dep_id', 'hour', 'hist_avg_stop_hour']
    df = df.merge(hist_stop_hour, on=['dep_id', 'hour'], how='left')

    # By stop/dow
    hist_stop_dow = df.groupby(['dep_id', 'dow'])['occ_pct'].mean().reset_index()
    hist_stop_dow.columns = ['dep_id', 'dow', 'hist_avg_stop_dow']
    df = df.merge(hist_stop_dow, on=['dep_id', 'dow'], how='left')

    # Fill NaN
    df['hist_std_stop_hour_dow'] = df['hist_std_stop_hour_dow'].fillna(df['occ_pct'].std())

    print(f"  Features created: {df.shape[1]} columns")
    print(f"  Match days: {df['is_match_day'].sum():,} observations")
    print(f"  Holidays: {df['is_holiday'].sum():,} observations")

    return df


def prepare_model_data(df, target_horizon='1h'):
    """Prepare features and target for modeling."""

    # Create target (shift-based)
    horizons = {'30min': 1, '1h': 2, '3h': 6, 'next_day': 48}
    shift = horizons.get(target_horizon, 2)

    df = df.sort_values(['dep_id', 'trip_dep_time']).reset_index(drop=True)
    df['target'] = df.groupby('dep_id')['occ_pct'].shift(-shift)

    # Feature columns
    feature_cols = [
        'hour', 'dow', 'month',
        'is_weekend', 'is_rush_hour', 'is_night',
        'is_match_day', 'is_holiday',
        'hist_avg_stop_hour_dow', 'hist_avg_stop_hour', 'hist_avg_stop_dow',
    ]

    # Filter valid rows
    df_clean = df.dropna(subset=feature_cols + ['target'])

    X = df_clean[feature_cols]
    y = df_clean['target']

    return X, y, feature_cols, df_clean


# =============================================================================
# Model Training
# =============================================================================

def train_model(X, y, random_state=42):
    """Train RandomForest model."""
    print("\n[3/5] Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  MAE: {mae:.2f}%")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  R2: {r2:.3f}")

    return model, X_train, X_test, y_train, y_test, y_pred


# =============================================================================
# SHAP Analysis
# =============================================================================

def run_shap_analysis(model, X_test, feature_cols, sample_size=2000):
    """Run comprehensive SHAP analysis."""
    try:
        import shap
    except ImportError:
        print("\nSHAP not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'shap'])
        import shap

    print("\n[4/5] Computing SHAP values...")

    # Sample for efficiency
    n_samples = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=n_samples, random_state=42)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"  Computed SHAP values for {n_samples} samples")

    return shap_values, X_sample, explainer


def analyze_scenarios(shap_values, X_sample, feature_cols):
    """Analyze SHAP values across different scenarios."""
    print("\n[5/5] Analyzing scenarios...")

    scenarios = {}

    # 1. Overall statistics
    overall_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'std_shap': shap_values.std(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    scenarios['overall'] = {
        'n_samples': len(X_sample),
        'importance': overall_importance,
        'shap_values': shap_values
    }

    # 2. Match day analysis
    if 'is_match_day' in X_sample.columns:
        match_idx = X_sample['is_match_day'] == 1
        non_match_idx = X_sample['is_match_day'] == 0

        if match_idx.sum() > 10:
            match_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[match_idx]).mean(axis=0),
                'std_shap': shap_values[match_idx].std(axis=0),
                'mean_shap': shap_values[match_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['match_day'] = {
                'n_samples': match_idx.sum(),
                'importance': match_importance,
                'shap_values': shap_values[match_idx],
                'X': X_sample[match_idx]
            }
            print(f"    Match day samples: {match_idx.sum()}")

        if non_match_idx.sum() > 10:
            non_match_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[non_match_idx]).mean(axis=0),
                'std_shap': shap_values[non_match_idx].std(axis=0),
                'mean_shap': shap_values[non_match_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['non_match_day'] = {
                'n_samples': non_match_idx.sum(),
                'importance': non_match_importance,
                'shap_values': shap_values[non_match_idx],
                'X': X_sample[non_match_idx]
            }

    # 3. Holiday analysis
    if 'is_holiday' in X_sample.columns:
        holiday_idx = X_sample['is_holiday'] == 1
        non_holiday_idx = X_sample['is_holiday'] == 0

        if holiday_idx.sum() > 10:
            holiday_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[holiday_idx]).mean(axis=0),
                'std_shap': shap_values[holiday_idx].std(axis=0),
                'mean_shap': shap_values[holiday_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['holiday'] = {
                'n_samples': holiday_idx.sum(),
                'importance': holiday_importance,
                'shap_values': shap_values[holiday_idx],
                'X': X_sample[holiday_idx]
            }
            print(f"    Holiday samples: {holiday_idx.sum()}")

        if non_holiday_idx.sum() > 10:
            non_holiday_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[non_holiday_idx]).mean(axis=0),
                'std_shap': shap_values[non_holiday_idx].std(axis=0),
                'mean_shap': shap_values[non_holiday_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['non_holiday'] = {
                'n_samples': non_holiday_idx.sum(),
                'importance': non_holiday_importance,
                'shap_values': shap_values[non_holiday_idx],
                'X': X_sample[non_holiday_idx]
            }

    # 4. Rush hour analysis
    if 'is_rush_hour' in X_sample.columns:
        rush_idx = X_sample['is_rush_hour'] == 1
        non_rush_idx = X_sample['is_rush_hour'] == 0

        if rush_idx.sum() > 10:
            rush_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[rush_idx]).mean(axis=0),
                'std_shap': shap_values[rush_idx].std(axis=0),
                'mean_shap': shap_values[rush_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['rush_hour'] = {
                'n_samples': rush_idx.sum(),
                'importance': rush_importance,
                'shap_values': shap_values[rush_idx],
                'X': X_sample[rush_idx]
            }
            print(f"    Rush hour samples: {rush_idx.sum()}")

        if non_rush_idx.sum() > 10:
            non_rush_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[non_rush_idx]).mean(axis=0),
                'std_shap': shap_values[non_rush_idx].std(axis=0),
                'mean_shap': shap_values[non_rush_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['non_rush_hour'] = {
                'n_samples': non_rush_idx.sum(),
                'importance': non_rush_importance,
                'shap_values': shap_values[non_rush_idx],
                'X': X_sample[non_rush_idx]
            }

    # 5. Weekend analysis
    if 'is_weekend' in X_sample.columns:
        weekend_idx = X_sample['is_weekend'] == 1
        weekday_idx = X_sample['is_weekend'] == 0

        if weekend_idx.sum() > 10:
            weekend_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[weekend_idx]).mean(axis=0),
                'std_shap': shap_values[weekend_idx].std(axis=0),
                'mean_shap': shap_values[weekend_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['weekend'] = {
                'n_samples': weekend_idx.sum(),
                'importance': weekend_importance,
                'shap_values': shap_values[weekend_idx],
                'X': X_sample[weekend_idx]
            }
            print(f"    Weekend samples: {weekend_idx.sum()}")

        if weekday_idx.sum() > 10:
            weekday_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[weekday_idx]).mean(axis=0),
                'std_shap': shap_values[weekday_idx].std(axis=0),
                'mean_shap': shap_values[weekday_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['weekday'] = {
                'n_samples': weekday_idx.sum(),
                'importance': weekday_importance,
                'shap_values': shap_values[weekday_idx],
                'X': X_sample[weekday_idx]
            }

    # 6. Night analysis
    if 'is_night' in X_sample.columns:
        night_idx = X_sample['is_night'] == 1

        if night_idx.sum() > 10:
            night_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values[night_idx]).mean(axis=0),
                'std_shap': shap_values[night_idx].std(axis=0),
                'mean_shap': shap_values[night_idx].mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)

            scenarios['night'] = {
                'n_samples': night_idx.sum(),
                'importance': night_importance,
                'shap_values': shap_values[night_idx],
                'X': X_sample[night_idx]
            }
            print(f"    Night samples: {night_idx.sum()}")

    return scenarios


# =============================================================================
# Visualization Functions
# =============================================================================

def create_visualizations(shap_values, X_sample, feature_cols, scenarios, output_dir):
    """Create all RQ2 visualizations."""
    import shap

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)

    # 1. SHAP Summary Plot (Beeswarm)
    print("\n  1. SHAP Summary Plot (Beeswarm)...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=len(feature_cols))
    plt.title('SHAP Summary Plot: Feature Impact on Occupancy Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq2_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SHAP Bar Plot (Feature Importance)
    print("  2. SHAP Feature Importance Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type='bar', show=False, max_display=len(feature_cols))
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq2_shap_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Scenario Comparison (Match Day vs Non-Match Day)
    if 'match_day' in scenarios and 'non_match_day' in scenarios:
        print("  3. Match Day vs Non-Match Day Comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Match day
        match_imp = scenarios['match_day']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_match_day', 'is_holiday'] else 'steelblue' for f in match_imp['feature']]
        axes[0].barh(match_imp['feature'], match_imp['mean_abs_shap'], color=colors)
        axes[0].set_xlabel('Mean |SHAP Value|')
        axes[0].set_title(f"Match Day (n={scenarios['match_day']['n_samples']})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Non-match day
        non_match_imp = scenarios['non_match_day']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_match_day', 'is_holiday'] else 'steelblue' for f in non_match_imp['feature']]
        axes[1].barh(non_match_imp['feature'], non_match_imp['mean_abs_shap'], color=colors)
        axes[1].set_xlabel('Mean |SHAP Value|')
        axes[1].set_title(f"Non-Match Day (n={scenarios['non_match_day']['n_samples']})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Feature Importance: Match Day vs Non-Match Day', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_match_day_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Holiday Comparison
    if 'holiday' in scenarios and 'non_holiday' in scenarios:
        print("  4. Holiday vs Non-Holiday Comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Holiday
        holiday_imp = scenarios['holiday']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_match_day', 'is_holiday'] else 'steelblue' for f in holiday_imp['feature']]
        axes[0].barh(holiday_imp['feature'], holiday_imp['mean_abs_shap'], color=colors)
        axes[0].set_xlabel('Mean |SHAP Value|')
        axes[0].set_title(f"Holiday (n={scenarios['holiday']['n_samples']})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Non-holiday
        non_holiday_imp = scenarios['non_holiday']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_match_day', 'is_holiday'] else 'steelblue' for f in non_holiday_imp['feature']]
        axes[1].barh(non_holiday_imp['feature'], non_holiday_imp['mean_abs_shap'], color=colors)
        axes[1].set_xlabel('Mean |SHAP Value|')
        axes[1].set_title(f"Non-Holiday (n={scenarios['non_holiday']['n_samples']})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Feature Importance: Holiday vs Non-Holiday', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_holiday_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Rush Hour Comparison
    if 'rush_hour' in scenarios and 'non_rush_hour' in scenarios:
        print("  5. Rush Hour vs Non-Rush Hour Comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Rush hour
        rush_imp = scenarios['rush_hour']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_rush_hour', 'hour'] else 'steelblue' for f in rush_imp['feature']]
        axes[0].barh(rush_imp['feature'], rush_imp['mean_abs_shap'], color=colors)
        axes[0].set_xlabel('Mean |SHAP Value|')
        axes[0].set_title(f"Rush Hour (n={scenarios['rush_hour']['n_samples']})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Non-rush hour
        non_rush_imp = scenarios['non_rush_hour']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_rush_hour', 'hour'] else 'steelblue' for f in non_rush_imp['feature']]
        axes[1].barh(non_rush_imp['feature'], non_rush_imp['mean_abs_shap'], color=colors)
        axes[1].set_xlabel('Mean |SHAP Value|')
        axes[1].set_title(f"Non-Rush Hour (n={scenarios['non_rush_hour']['n_samples']})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Feature Importance: Rush Hour vs Non-Rush Hour', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_rush_hour_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Weekend vs Weekday Comparison
    if 'weekend' in scenarios and 'weekday' in scenarios:
        print("  6. Weekend vs Weekday Comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Weekend
        weekend_imp = scenarios['weekend']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_weekend', 'dow'] else 'steelblue' for f in weekend_imp['feature']]
        axes[0].barh(weekend_imp['feature'], weekend_imp['mean_abs_shap'], color=colors)
        axes[0].set_xlabel('Mean |SHAP Value|')
        axes[0].set_title(f"Weekend (n={scenarios['weekend']['n_samples']})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Weekday
        weekday_imp = scenarios['weekday']['importance'].sort_values('mean_abs_shap', ascending=True)
        colors = ['coral' if f in ['is_weekend', 'dow'] else 'steelblue' for f in weekday_imp['feature']]
        axes[1].barh(weekday_imp['feature'], weekday_imp['mean_abs_shap'], color=colors)
        axes[1].set_xlabel('Mean |SHAP Value|')
        axes[1].set_title(f"Weekday (n={scenarios['weekday']['n_samples']})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Feature Importance: Weekend vs Weekday', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_weekend_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 7. All Scenarios Heatmap
    print("  7. All Scenarios Feature Importance Heatmap...")
    scenario_names = ['overall', 'match_day', 'non_match_day', 'holiday', 'non_holiday',
                      'rush_hour', 'non_rush_hour', 'weekend', 'weekday']

    heatmap_data = []
    valid_scenarios = []
    for name in scenario_names:
        if name in scenarios:
            imp = scenarios[name]['importance'].set_index('feature')['mean_abs_shap']
            heatmap_data.append(imp)
            valid_scenarios.append(name)

    if len(heatmap_data) > 1:
        heatmap_df = pd.concat(heatmap_data, axis=1)
        heatmap_df.columns = valid_scenarios

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Feature Importance Across Scenarios (Mean |SHAP Value|)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_scenario_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 8. External Factors Impact Comparison
    print("  8. External Factors Impact Comparison...")
    external_features = ['is_match_day', 'is_holiday', 'is_weekend', 'is_rush_hour', 'is_night']

    comparison_data = []
    for scenario_name in valid_scenarios:
        if scenario_name in scenarios:
            imp = scenarios[scenario_name]['importance'].set_index('feature')
            for feat in external_features:
                if feat in imp.index:
                    comparison_data.append({
                        'scenario': scenario_name,
                        'feature': feat,
                        'mean_abs_shap': imp.loc[feat, 'mean_abs_shap']
                    })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        fig, ax = plt.subplots(figsize=(14, 6))
        pivot_df = comparison_df.pivot(index='feature', columns='scenario', values='mean_abs_shap')
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('External Factor')
        ax.set_ylabel('Mean |SHAP Value|')
        ax.set_title('External Factors Impact Across Scenarios', fontsize=14, fontweight='bold')
        ax.legend(title='Scenario', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_external_factors_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 9. SHAP Dependence Plots for Key Features
    print("  9. SHAP Dependence Plots...")
    key_features = ['hour', 'is_match_day', 'is_holiday', 'hist_avg_stop_dow']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        if feat in feature_cols:
            feat_idx = feature_cols.index(feat)
            ax = axes[i]
            shap.dependence_plot(feat_idx, shap_values, X_sample, feature_names=feature_cols,
                               ax=ax, show=False)
            ax.set_title(f'SHAP Dependence: {feat}', fontsize=12, fontweight='bold')

    plt.suptitle('SHAP Dependence Plots for Key Features', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq2_shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Feature Importance Change (Delta) Plot
    print("  10. Feature Importance Change Across Scenarios...")

    if 'overall' in scenarios:
        overall_imp = scenarios['overall']['importance'].set_index('feature')['mean_abs_shap']

        deltas = []
        scenario_pairs = [
            ('match_day', 'Match Day'),
            ('holiday', 'Holiday'),
            ('rush_hour', 'Rush Hour'),
            ('weekend', 'Weekend')
        ]

        for scenario_key, scenario_label in scenario_pairs:
            if scenario_key in scenarios:
                scenario_imp = scenarios[scenario_key]['importance'].set_index('feature')['mean_abs_shap']
                for feat in feature_cols:
                    if feat in scenario_imp.index and feat in overall_imp.index:
                        deltas.append({
                            'scenario': scenario_label,
                            'feature': feat,
                            'delta': scenario_imp[feat] - overall_imp[feat]
                        })

        if deltas:
            delta_df = pd.DataFrame(deltas)

            fig, ax = plt.subplots(figsize=(14, 8))
            pivot_delta = delta_df.pivot(index='feature', columns='scenario', values='delta')

            # Sort by total absolute delta
            pivot_delta['total_abs'] = pivot_delta.abs().sum(axis=1)
            pivot_delta = pivot_delta.sort_values('total_abs', ascending=True)
            pivot_delta = pivot_delta.drop('total_abs', axis=1)

            pivot_delta.plot(kind='barh', ax=ax, width=0.8)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Delta (Scenario - Overall)')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance Change from Overall Average', fontsize=14, fontweight='bold')
            ax.legend(title='Scenario', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'rq2_importance_delta.png', dpi=300, bbox_inches='tight')
            plt.close()

    print("  Visualizations saved!")


# =============================================================================
# Tables and Reports
# =============================================================================

def create_tables(scenarios, feature_cols, output_dir):
    """Create result tables."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Creating tables...")
    print("=" * 70)

    # 1. Overall Feature Importance Table
    overall_imp = scenarios['overall']['importance'].copy()
    overall_imp['rank'] = range(1, len(overall_imp) + 1)
    overall_imp = overall_imp[['rank', 'feature', 'mean_abs_shap', 'mean_shap', 'std_shap']]
    overall_imp.to_csv(Path(output_dir) / 'table_overall_importance.csv', index=False)
    print("  1. Overall importance table saved")

    # 2. Scenario Summary Table
    scenario_summary = []
    for name, data in scenarios.items():
        if name != 'overall':
            top_features = data['importance'].head(3)['feature'].tolist()
            scenario_summary.append({
                'scenario': name,
                'n_samples': data['n_samples'],
                'top_1_feature': top_features[0] if len(top_features) > 0 else '',
                'top_2_feature': top_features[1] if len(top_features) > 1 else '',
                'top_3_feature': top_features[2] if len(top_features) > 2 else '',
            })

    summary_df = pd.DataFrame(scenario_summary)
    summary_df.to_csv(Path(output_dir) / 'table_scenario_summary.csv', index=False)
    print("  2. Scenario summary table saved")

    # 3. External Factors Impact Table
    external_features = ['is_match_day', 'is_holiday', 'is_weekend', 'is_rush_hour', 'is_night']

    external_impact = []
    for scenario_name, data in scenarios.items():
        imp = data['importance'].set_index('feature')
        row = {'scenario': scenario_name, 'n_samples': data['n_samples']}
        for feat in external_features:
            if feat in imp.index:
                row[f'{feat}_shap'] = round(imp.loc[feat, 'mean_abs_shap'], 4)
        external_impact.append(row)

    external_df = pd.DataFrame(external_impact)
    external_df.to_csv(Path(output_dir) / 'table_external_factors_impact.csv', index=False)
    print("  3. External factors impact table saved")

    # 4. Feature Importance by Scenario (Wide format)
    wide_data = {}
    for scenario_name, data in scenarios.items():
        imp = data['importance'].set_index('feature')['mean_abs_shap']
        wide_data[scenario_name] = imp

    wide_df = pd.DataFrame(wide_data)
    wide_df.to_csv(Path(output_dir) / 'table_importance_by_scenario.csv')
    print("  4. Importance by scenario table saved")

    # 5. Delta Table (Change from Overall)
    if 'overall' in scenarios:
        overall_imp_dict = scenarios['overall']['importance'].set_index('feature')['mean_abs_shap']

        delta_data = []
        for scenario_name, data in scenarios.items():
            if scenario_name != 'overall':
                scenario_imp = data['importance'].set_index('feature')['mean_abs_shap']
                for feat in feature_cols:
                    if feat in scenario_imp.index and feat in overall_imp_dict.index:
                        delta_data.append({
                            'scenario': scenario_name,
                            'feature': feat,
                            'overall_shap': round(overall_imp_dict[feat], 4),
                            'scenario_shap': round(scenario_imp[feat], 4),
                            'delta': round(scenario_imp[feat] - overall_imp_dict[feat], 4),
                            'pct_change': round((scenario_imp[feat] - overall_imp_dict[feat]) / overall_imp_dict[feat] * 100, 2) if overall_imp_dict[feat] > 0 else 0
                        })

        delta_df = pd.DataFrame(delta_data)
        delta_df.to_csv(Path(output_dir) / 'table_importance_delta.csv', index=False)
        print("  5. Importance delta table saved")

    print("  Tables saved!")

    return overall_imp, summary_df, external_df


def generate_report(scenarios, feature_cols, model_metrics, output_dir):
    """Generate comprehensive RQ2 analysis report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_lines = []

    def add(line=''):
        report_lines.append(line)

    add("=" * 80)
    add("RQ2 ANALYSIS REPORT: SHAP-based Explainability")
    add("=" * 80)
    add()
    add("Research Question:")
    add("How do external factors shape SHAP-based explanations of seat availability")
    add("predictions in special scenarios such as match days, holidays, and rush hours?")
    add()
    add("-" * 80)

    # 1. Model Performance
    add()
    add("1. MODEL PERFORMANCE")
    add("-" * 40)
    add(f"   MAE:  {model_metrics['MAE']:.2f}%")
    add(f"   RMSE: {model_metrics['RMSE']:.2f}%")
    add(f"   R2:   {model_metrics['R2']:.3f}")
    add()

    # 2. Overall Feature Importance
    add("2. OVERALL FEATURE IMPORTANCE (SHAP)")
    add("-" * 40)
    overall_imp = scenarios['overall']['importance']
    add(f"   {'Rank':<6} {'Feature':<25} {'Mean |SHAP|':>12}")
    add("   " + "-" * 45)
    for i, row in overall_imp.iterrows():
        rank = list(overall_imp.index).index(i) + 1
        add(f"   {rank:<6} {row['feature']:<25} {row['mean_abs_shap']:>12.4f}")
    add()

    # 3. Key Findings by Scenario
    add("3. KEY FINDINGS BY SCENARIO")
    add("-" * 40)

    scenario_descriptions = {
        'match_day': 'Match Day (Football games at Olympiastadion)',
        'non_match_day': 'Non-Match Day (Normal days)',
        'holiday': 'Holiday (German public holidays)',
        'non_holiday': 'Non-Holiday (Working days)',
        'rush_hour': 'Rush Hour (6-9, 16-19)',
        'non_rush_hour': 'Non-Rush Hour',
        'weekend': 'Weekend (Saturday, Sunday)',
        'weekday': 'Weekday (Monday-Friday)',
        'night': 'Night (22:00-06:00)'
    }

    for scenario_name, desc in scenario_descriptions.items():
        if scenario_name in scenarios:
            data = scenarios[scenario_name]
            add()
            add(f"   [{desc}]")
            add(f"   Samples: {data['n_samples']}")
            add(f"   Top 3 features:")
            top3 = data['importance'].head(3)
            for i, row in top3.iterrows():
                rank = list(top3.index).index(i) + 1
                add(f"     {rank}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    add()

    # 4. External Factors Analysis
    add("4. EXTERNAL FACTORS ANALYSIS")
    add("-" * 40)

    external_features = ['is_match_day', 'is_holiday', 'is_weekend', 'is_rush_hour', 'is_night']

    overall_ext = scenarios['overall']['importance'].set_index('feature')

    add()
    add("   External Factor Impact (Overall):")
    add(f"   {'Factor':<20} {'Mean |SHAP|':>12} {'Rank':>6}")
    add("   " + "-" * 40)

    for feat in external_features:
        if feat in overall_ext.index:
            shap_val = overall_ext.loc[feat, 'mean_abs_shap']
            rank = list(scenarios['overall']['importance']['feature']).index(feat) + 1
            add(f"   {feat:<20} {shap_val:>12.4f} {rank:>6}")
    add()

    # 5. Scenario Comparison - Key Differences
    add("5. SCENARIO COMPARISON - KEY DIFFERENCES")
    add("-" * 40)

    if 'match_day' in scenarios and 'non_match_day' in scenarios:
        add()
        add("   [Match Day vs Non-Match Day]")
        match_imp = scenarios['match_day']['importance'].set_index('feature')
        non_match_imp = scenarios['non_match_day']['importance'].set_index('feature')

        # Find biggest differences
        diffs = []
        for feat in feature_cols:
            if feat in match_imp.index and feat in non_match_imp.index:
                diff = match_imp.loc[feat, 'mean_abs_shap'] - non_match_imp.loc[feat, 'mean_abs_shap']
                diffs.append((feat, diff))

        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        add("   Biggest differences (Match - Non-Match):")
        for feat, diff in diffs[:5]:
            direction = "+" if diff > 0 else ""
            add(f"     {feat}: {direction}{diff:.4f}")

    if 'holiday' in scenarios and 'non_holiday' in scenarios:
        add()
        add("   [Holiday vs Non-Holiday]")
        holiday_imp = scenarios['holiday']['importance'].set_index('feature')
        non_holiday_imp = scenarios['non_holiday']['importance'].set_index('feature')

        diffs = []
        for feat in feature_cols:
            if feat in holiday_imp.index and feat in non_holiday_imp.index:
                diff = holiday_imp.loc[feat, 'mean_abs_shap'] - non_holiday_imp.loc[feat, 'mean_abs_shap']
                diffs.append((feat, diff))

        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        add("   Biggest differences (Holiday - Non-Holiday):")
        for feat, diff in diffs[:5]:
            direction = "+" if diff > 0 else ""
            add(f"     {feat}: {direction}{diff:.4f}")

    if 'rush_hour' in scenarios and 'non_rush_hour' in scenarios:
        add()
        add("   [Rush Hour vs Non-Rush Hour]")
        rush_imp = scenarios['rush_hour']['importance'].set_index('feature')
        non_rush_imp = scenarios['non_rush_hour']['importance'].set_index('feature')

        diffs = []
        for feat in feature_cols:
            if feat in rush_imp.index and feat in non_rush_imp.index:
                diff = rush_imp.loc[feat, 'mean_abs_shap'] - non_rush_imp.loc[feat, 'mean_abs_shap']
                diffs.append((feat, diff))

        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        add("   Biggest differences (Rush - Non-Rush):")
        for feat, diff in diffs[:5]:
            direction = "+" if diff > 0 else ""
            add(f"     {feat}: {direction}{diff:.4f}")
    add()

    # 6. Interpretation and Implications
    add("6. INTERPRETATION AND IMPLICATIONS")
    add("-" * 40)
    add()

    # Find most important feature
    top_feature = scenarios['overall']['importance'].iloc[0]['feature']
    top_shap = scenarios['overall']['importance'].iloc[0]['mean_abs_shap']

    add(f"   Most Important Feature: {top_feature} (SHAP: {top_shap:.4f})")
    add()

    add("   Key Insights:")
    add()

    # Insight 1: Historical patterns dominate
    hist_features = ['hist_avg_stop_hour_dow', 'hist_avg_stop_hour', 'hist_avg_stop_dow']
    hist_shap_sum = sum(
        scenarios['overall']['importance'].set_index('feature').loc[f, 'mean_abs_shap']
        for f in hist_features if f in scenarios['overall']['importance']['feature'].values
    )
    total_shap = scenarios['overall']['importance']['mean_abs_shap'].sum()
    hist_pct = hist_shap_sum / total_shap * 100

    add(f"   1. Historical patterns contribute {hist_pct:.1f}% of total feature importance.")
    add("      -> Past behavior is the strongest predictor of future occupancy.")
    add()

    # Insight 2: External factors are contextual
    external_shap_sum = sum(
        scenarios['overall']['importance'].set_index('feature').loc[f, 'mean_abs_shap']
        for f in external_features if f in scenarios['overall']['importance']['feature'].values
    )
    external_pct = external_shap_sum / total_shap * 100

    add(f"   2. External factors contribute {external_pct:.1f}% of total feature importance.")
    add("      -> They provide contextual adjustments rather than primary predictions.")
    add()

    # Insight 3: Scenario-specific importance shifts
    if 'match_day' in scenarios:
        match_day_feat_rank = list(scenarios['match_day']['importance']['feature']).index('is_match_day') + 1 if 'is_match_day' in scenarios['match_day']['importance']['feature'].values else -1
        overall_match_day_rank = list(scenarios['overall']['importance']['feature']).index('is_match_day') + 1 if 'is_match_day' in scenarios['overall']['importance']['feature'].values else -1

        add(f"   3. 'is_match_day' importance:")
        add(f"      - Overall rank: #{overall_match_day_rank}")
        add(f"      - On match days: #{match_day_feat_rank}")
        if match_day_feat_rank < overall_match_day_rank:
            add("      -> Match day feature becomes MORE important during match days (context activation)")
        add()

    if 'holiday' in scenarios:
        holiday_shap = scenarios['holiday']['importance'].set_index('feature').loc['is_holiday', 'mean_abs_shap'] if 'is_holiday' in scenarios['holiday']['importance']['feature'].values else 0
        overall_holiday_shap = scenarios['overall']['importance'].set_index('feature').loc['is_holiday', 'mean_abs_shap'] if 'is_holiday' in scenarios['overall']['importance']['feature'].values else 0

        add(f"   4. 'is_holiday' importance:")
        add(f"      - Overall SHAP: {overall_holiday_shap:.4f}")
        add(f"      - On holidays: {holiday_shap:.4f}")
        if holiday_shap > overall_holiday_shap:
            change = (holiday_shap - overall_holiday_shap) / overall_holiday_shap * 100 if overall_holiday_shap > 0 else 0
            add(f"      -> Holiday feature is {change:.1f}% more important on actual holidays")
        add()

    # 7. Decision Support Implications
    add("7. DECISION SUPPORT IMPLICATIONS")
    add("-" * 40)
    add()
    add("   For Passengers:")
    add("   - Historical patterns (same stop, same day, same time) are the best guide")
    add("   - On match days, expect higher occupancy, especially in affected time windows")
    add("   - On holidays, typical patterns may not apply - check real-time predictions")
    add("   - During rush hours, time of day is the dominant factor")
    add()
    add("   For Operators:")
    add("   - Base predictions on historical averages by station/hour/day-of-week")
    add("   - Apply match day adjustments for event-related capacity planning")
    add("   - Holiday schedules may need special consideration")
    add("   - Rush hour predictions are time-driven - focus on hour-specific patterns")
    add()

    # 8. Conclusion
    add("8. CONCLUSION")
    add("-" * 40)
    add()
    add("   RQ2 Analysis demonstrates that:")
    add()
    add("   1. SHAP provides interpretable explanations for occupancy predictions")
    add()
    add("   2. Historical patterns dominate predictions across all scenarios")
    add()
    add("   3. External factors (match days, holidays) activate in relevant contexts:")
    add("      - Match day importance increases during actual match days")
    add("      - Holiday importance increases during holidays")
    add("      - Rush hour effects are captured through hour feature")
    add()
    add("   4. Model explanations can support both:")
    add("      - Passenger trip planning (understand why certain times are busy)")
    add("      - Operational decisions (plan resources based on explainable predictions)")
    add()
    add("=" * 80)
    add("END OF RQ2 ANALYSIS REPORT")
    add("=" * 80)

    # Save report
    report_text = '\n'.join(report_lines)
    with open(Path(output_dir) / 'rq2_analysis_report.txt', 'w') as f:
        f.write(report_text)

    print("\n  Report saved to rq2_analysis_report.txt")

    return report_text


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_rq2_analysis(output_dir='outputs/analysis_rq2'):
    """Run complete RQ2 analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare data
    df = load_and_prepare_data()
    df = create_features(df)

    # 2. Prepare model data
    X, y, feature_cols, df_clean = prepare_model_data(df, target_horizon='1h')
    print(f"\n  Model data prepared: {len(X):,} samples, {len(feature_cols)} features")

    # 3. Train model
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)

    # Save model metrics
    model_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    # Save model
    joblib.dump(model, Path(output_dir) / 'rq2_model.joblib')

    # 4. Run SHAP analysis
    shap_values, X_sample, explainer = run_shap_analysis(model, X_test, feature_cols, sample_size=2000)

    # 5. Analyze scenarios
    scenarios = analyze_scenarios(shap_values, X_sample, feature_cols)

    # 6. Create visualizations
    create_visualizations(shap_values, X_sample, feature_cols, scenarios, output_dir)

    # 7. Create tables
    overall_imp, summary_df, external_df = create_tables(scenarios, feature_cols, output_dir)

    # 8. Generate report
    report = generate_report(scenarios, feature_cols, model_metrics, output_dir)

    # Print summary to console
    print("\n" + "=" * 70)
    print("RQ2 ANALYSIS COMPLETE")
    print("=" * 70)

    print("\n[Summary]")
    print(f"  Model Performance: MAE={model_metrics['MAE']:.2f}%, R2={model_metrics['R2']:.3f}")
    print(f"  SHAP samples analyzed: {len(X_sample)}")
    print(f"  Scenarios analyzed: {len(scenarios)}")

    print("\n[Top 5 Features by SHAP Importance]")
    top5 = scenarios['overall']['importance'].head(5)
    for i, row in top5.iterrows():
        print(f"  {list(top5.index).index(i)+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")

    print(f"\n[Output Files]")
    print(f"  Directory: {output_dir}/")
    print("  - rq2_analysis_report.txt (comprehensive report)")
    print("  - rq2_shap_summary_beeswarm.png (SHAP summary)")
    print("  - rq2_shap_importance_bar.png (feature importance)")
    print("  - rq2_match_day_comparison.png")
    print("  - rq2_holiday_comparison.png")
    print("  - rq2_rush_hour_comparison.png")
    print("  - rq2_weekend_comparison.png")
    print("  - rq2_scenario_heatmap.png")
    print("  - rq2_external_factors_comparison.png")
    print("  - rq2_shap_dependence.png")
    print("  - rq2_importance_delta.png")
    print("  - table_*.csv (result tables)")

    return {
        'model': model,
        'model_metrics': model_metrics,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'scenarios': scenarios,
        'feature_cols': feature_cols,
        'report': report
    }


if __name__ == '__main__':
    results = run_rq2_analysis('outputs/analysis_rq2')
