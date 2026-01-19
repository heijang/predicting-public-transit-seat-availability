import pandas as pd
import os

def load_raw(re5_path='data/ingested/RE5_2024_03.csv',
             re6_path='data/ingested/RE6_2024_03.csv'):
    re5 = pd.read_csv(re5_path)
    re6 = pd.read_csv(re6_path)
    return re5, re6


def standardize_columns(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def convert_datetime(df):
    """Convert datetime columns from HHMM format to proper datetime."""
    df['trip_start_date'] = pd.to_datetime(df['trip_start_date'], format='%Y-%m-%d', errors='coerce')

    # Convert HHMM format (e.g., "0431") to time
    df['trip_dep_time'] = df['trip_dep_time'].astype(str).str.zfill(4)
    df['trip_arr_time'] = df['trip_arr_time'].astype(str).str.zfill(4)

    # Combine date + time columns
    df['trip_dep_time'] = pd.to_datetime(
        df['trip_start_date'].astype(str) + ' ' +
        df['trip_dep_time'].str[:2] + ':' +
        df['trip_dep_time'].str[2:4],
        format='%Y-%m-%d %H:%M',
        errors='coerce'
    )

    df['trip_arr_time'] = pd.to_datetime(
        df['trip_start_date'].astype(str) + ' ' +
        df['trip_arr_time'].str[:2] + ':' +
        df['trip_arr_time'].str[2:4],
        format='%Y-%m-%d %H:%M',
        errors='coerce'
    )

    return df


def create_features(df):
    """Create derived features: hour, weekday, rush_hour, weekend, night, occupancy %."""

    # Extract hour from departure time
    df['hour'] = df['trip_dep_time'].dt.hour

    # Extract date (normalized)
    df['date'] = df['trip_start_date'].dt.date

    # Weekday (0=Monday, 6=Sunday)
    df['dow'] = df['trip_start_date'].dt.dayofweek
    df['dow_name'] = df['trip_start_date'].dt.day_name()

    # Rush hour (6-9, 16-19)
    df['is_rush_hour'] = df['hour'].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Weekend (Saturday=5, Sunday=6)
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    # Night (22-5)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    # Occupancy percentage
    df['occ_pct'] = (df['occupancy'] / df['capacity'] * 100).round(2)

    return df


def identify_match_days(df):
    # Calculate daily median occupancy
    daily_median = df.groupby('date')['occ_pct'].median()

    # Calculate IQR for threshold
    Q1 = daily_median.quantile(0.25)
    Q3 = daily_median.quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    # Mark high-occupancy days as match days
    match_day_dates = set(daily_median[daily_median > threshold].index)
    df['is_match_day'] = df['date'].isin(match_day_dates).astype(int)

    return df


def clean_re5_re6(re5_raw, re6_raw):
    re5_clean = re5_raw.copy()
    re5_clean = standardize_columns(re5_clean)
    re5_clean = convert_datetime(re5_clean)
    re5_clean = create_features(re5_clean)
    re5_clean = identify_match_days(re5_clean)
    
    re6_clean = re6_raw.copy()
    re6_clean = standardize_columns(re6_clean)
    re6_clean = convert_datetime(re6_clean)
    re6_clean = create_features(re6_clean)
    re6_clean = identify_match_days(re6_clean)
    
    return re5_clean, re6_clean


def save_processed(re5_clean, re6_clean, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)

    re5_path = os.path.join(output_dir, 're5_processed.csv')
    re6_path = os.path.join(output_dir, 're6_processed.csv')

    re5_clean.to_csv(re5_path, index=False)
    re6_clean.to_csv(re6_path, index=False)


def main():
    """Main entry point for module execution."""
    # Load
    re5_raw, re6_raw = load_raw()

    # Clean
    re5_clean, re6_clean = clean_re5_re6(re5_raw, re6_raw)

    # Save
    save_processed(re5_clean, re6_clean)


if __name__ == "__main__":
    main()