import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
RQ1 Analysis: Match Day vs Non-Match Day Occupancy Comparison

Statistical Methods:
- Mann-Whitney U test: Non-parametric test for comparing two distributions
- Welch's t-test: Robust t-test for unequal variances
- Cohen's d: Effect size measurement
- Bootstrap confidence intervals: Distribution-free uncertainty estimation
"""


def load_occupancy_data(filepath='data/ingested/RE5_2024_03.csv', min_capacity=100):
    """Load RE5 occupancy data with data quality filtering."""
    df = pd.read_csv(filepath)
    df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
    df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])

    # Filter out invalid capacity values (data quality)
    initial_count = len(df)
    df = df[df['capacity'] >= min_capacity].copy()
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"  Filtered {filtered_count:,} rows with capacity < {min_capacity}")

    # Calculate occupancy percentage (keep raw values, don't cap)
    df['occ_pct'] = (df['occupancy'] / df['capacity'] * 100).round(2)

    # Extract time features
    df['hour'] = df['trip_dep_time'].dt.hour
    df['dow'] = df['trip_start_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['dow_name'] = df['trip_start_date'].dt.day_name()
    df['date'] = df['trip_start_date'].dt.date

    return df


def load_match_schedule(filepath='data/raw/Schedule_Teams_2022_2024.xlsx'):
    """Load and parse match schedule from Excel."""
    # Read with header on row 1 (skip first empty row)
    df = pd.read_excel(filepath, header=1)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse date
    df['match_date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Parse time - extract hour from Time column
    if 'Time' in df.columns:
        # Time is in format HH:MM:SS
        time_str = df['Time'].astype(str)
        df['match_hour'] = time_str.str.extract(r'(\d{1,2})')[0].astype(float)

    # Keep team and stadium info
    df['team'] = df.get('Team', '')
    df['stadium'] = df.get('Stadium', '')

    return df


def identify_match_windows(match_df, hours_before=2, hours_after=2):
    """
    Create time windows around matches.

    Args:
        match_df: Match schedule dataframe
        hours_before: Hours before match to include
        hours_after: Hours after match end to include

    Returns:
        List of dicts with match date and affected hours
    """
    match_windows = []

    for _, row in match_df.iterrows():
        if pd.isna(row.get('match_date')):
            continue

        match_date = row['match_date'].date() if hasattr(row['match_date'], 'date') else row['match_date']
        match_hour = row.get('match_hour', 15)  # Default 15:00 if unknown

        if pd.isna(match_hour):
            match_hour = 15

        match_hour = int(match_hour)

        # Match typically lasts ~2 hours
        start_hour = max(0, match_hour - hours_before)
        end_hour = min(23, match_hour + 2 + hours_after)  # +2 for match duration

        match_windows.append({
            'date': match_date,
            'match_hour': match_hour,
            'affected_hours': list(range(start_hour, end_hour + 1)),
            'dow': pd.Timestamp(match_date).dayofweek
        })

    return match_windows


def classify_observations(occ_df, match_windows):
    """
    Classify each observation as match-day or control.

    Match-day: date is a match date AND hour is in affected window
    Control: same DOW, same hour range, but not a match date
    """
    occ_df = occ_df.copy()

    # Create lookup structures
    match_dates = {w['date'] for w in match_windows}
    match_hours_by_date = {w['date']: set(w['affected_hours']) for w in match_windows}

    # Classify each row
    def classify(row):
        row_date = row['date']
        row_hour = row['hour']

        if row_date in match_dates:
            if row_hour in match_hours_by_date.get(row_date, set()):
                return 'match_window'
            else:
                return 'match_day_outside'
        return 'non_match'

    occ_df['category'] = occ_df.apply(classify, axis=1)

    return occ_df


def get_matched_control_group(occ_df, match_windows, verbose=True):
    """
    Get control observations: same DOW and hour range, but non-match dates.

    For fair comparison, we match on:
    - Same day of week (e.g., Saturday vs Saturday)
    - Same hour range (e.g., 13-19시)
    """
    match_dates = {w['date'] for w in match_windows}
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Get match day observations
    match_obs = occ_df[occ_df['category'] == 'match_window'].copy()

    # For each match, find control observations
    control_groups = []

    if verbose:
        print("\n" + "=" * 60)
        print("MATCHING DETAILS: Same DOW + Same Hours")
        print("=" * 60)

    for window in match_windows:
        match_dow = window['dow']
        match_dow_name = dow_names[match_dow]
        affected_hours = sorted(window['affected_hours'])
        match_date = window['date']
        match_hour = window['match_hour']

        # Find control: same DOW, same hours, different (non-match) date
        control = occ_df[
            (occ_df['dow'] == match_dow) &
            (occ_df['hour'].isin(affected_hours)) &
            (~occ_df['date'].isin(match_dates))
        ].copy()

        # Count unique control dates
        control_dates = control['date'].unique()

        # Get match day observations for this specific match
        match_day_obs = occ_df[
            (occ_df['date'] == match_date) &
            (occ_df['hour'].isin(affected_hours))
        ]

        if verbose:
            print(f"\n🏟️  Match: {match_date} ({match_dow_name}) @ {int(match_hour):02d}:00")
            print(f"   Analysis window: {min(affected_hours):02d}:00 - {max(affected_hours):02d}:59")
            print(f"   Match day obs: {len(match_day_obs):,}")
            print(f"   Control {match_dow_name}s (non-match): {len(control_dates)} days, {len(control):,} obs")
            if len(control_dates) > 0:
                print(f"   Control dates: {sorted([str(d) for d in control_dates[:5]])}" +
                      (f" ... +{len(control_dates)-5} more" if len(control_dates) > 5 else ""))

        control['matched_to'] = str(match_date)
        control_groups.append(control)

    if control_groups:
        control_df = pd.concat(control_groups, ignore_index=True)
        # Remove duplicates (same observation might match multiple windows)
        control_df = control_df.drop_duplicates(subset=['trip_id', 'dep_id', 'trip_dep_time'])
    else:
        control_df = pd.DataFrame()

    return match_obs, control_df


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (group1.mean() - group2.mean()) / pooled_std


def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def bootstrap_ci(data1, data2, n_bootstrap=1000, ci=95, statistic='mean_diff'):
    """
    Calculate bootstrap confidence interval for mean difference.
    """
    np.random.seed(42)

    boot_stats = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)

        if statistic == 'mean_diff':
            boot_stats.append(sample1.mean() - sample2.mean())
        elif statistic == 'median_diff':
            boot_stats.append(np.median(sample1) - np.median(sample2))

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_stats, alpha)
    upper = np.percentile(boot_stats, 100 - alpha)

    return lower, upper, boot_stats


def run_statistical_tests(match_occ, control_occ):
    """
    Run comprehensive statistical comparison.

    Returns:
        dict: Test results with statistics and interpretations
    """
    results = {
        'n_match': len(match_occ),
        'n_control': len(control_occ),
        'mean_match': match_occ.mean(),
        'mean_control': control_occ.mean(),
        'median_match': match_occ.median(),
        'median_control': control_occ.median(),
        'std_match': match_occ.std(),
        'std_control': control_occ.std(),
    }

    # 1. Mann-Whitney U test (non-parametric)
    u_stat, mw_pvalue = stats.mannwhitneyu(
        match_occ, control_occ, alternative='greater'
    )
    results['mannwhitney_u'] = u_stat
    results['mannwhitney_p'] = mw_pvalue

    # 2. Welch's t-test (robust to unequal variances)
    t_stat, t_pvalue = stats.ttest_ind(
        match_occ, control_occ, equal_var=False, alternative='greater'
    )
    results['welch_t'] = t_stat
    results['welch_p'] = t_pvalue

    # 3. Cohen's d effect size
    d = cohens_d(match_occ, control_occ)
    results['cohens_d'] = d
    results['effect_interpretation'] = interpret_cohens_d(d)

    # 4. Bootstrap confidence interval
    ci_lower, ci_upper, _ = bootstrap_ci(
        match_occ.values, control_occ.values, n_bootstrap=1000
    )
    results['bootstrap_ci_lower'] = ci_lower
    results['bootstrap_ci_upper'] = ci_upper

    # 5. Interpretation
    alpha = 0.05
    results['significant_mw'] = mw_pvalue < alpha
    results['significant_welch'] = t_pvalue < alpha

    return results


def format_results(results):
    """Format results for display."""
    output = []
    output.append("=" * 60)
    output.append("MATCH DAY vs NON-MATCH DAY OCCUPANCY ANALYSIS")
    output.append("=" * 60)

    output.append(f"\n📊 Sample Sizes:")
    output.append(f"   Match window observations:     {results['n_match']:,}")
    output.append(f"   Control (non-match) observations: {results['n_control']:,}")

    output.append(f"\n📈 Descriptive Statistics (Occupancy %):")
    output.append(f"   {'':20} {'Match Day':>12} {'Control':>12} {'Diff':>10}")
    output.append(f"   {'Mean':20} {results['mean_match']:>12.2f} {results['mean_control']:>12.2f} {results['mean_match']-results['mean_control']:>+10.2f}")
    output.append(f"   {'Median':20} {results['median_match']:>12.2f} {results['median_control']:>12.2f} {results['median_match']-results['median_control']:>+10.2f}")
    output.append(f"   {'Std Dev':20} {results['std_match']:>12.2f} {results['std_control']:>12.2f}")

    output.append(f"\n🔬 Statistical Tests (H₁: Match day > Control):")
    output.append(f"   Mann-Whitney U test:")
    output.append(f"      U-statistic: {results['mannwhitney_u']:,.0f}")
    output.append(f"      p-value:     {results['mannwhitney_p']:.4e}")
    output.append(f"      Significant: {'✓ Yes' if results['significant_mw'] else '✗ No'} (α=0.05)")

    output.append(f"\n   Welch's t-test:")
    output.append(f"      t-statistic: {results['welch_t']:.4f}")
    output.append(f"      p-value:     {results['welch_p']:.4e}")
    output.append(f"      Significant: {'✓ Yes' if results['significant_welch'] else '✗ No'} (α=0.05)")

    output.append(f"\n📏 Effect Size:")
    output.append(f"   Cohen's d:      {results['cohens_d']:.4f} ({results['effect_interpretation']})")
    output.append(f"   95% Bootstrap CI for mean difference: [{results['bootstrap_ci_lower']:.2f}, {results['bootstrap_ci_upper']:.2f}]")

    output.append(f"\n💡 Interpretation:")
    if results['significant_mw'] and results['cohens_d'] > 0:
        output.append(f"   Match day periods show SIGNIFICANTLY HIGHER occupancy")
        output.append(f"   than comparable non-match day periods.")
        output.append(f"   Effect size is {results['effect_interpretation']} (d={results['cohens_d']:.2f}).")
    elif results['significant_mw']:
        output.append(f"   Statistically significant but effect size is {results['effect_interpretation']}.")
    else:
        output.append(f"   No statistically significant difference detected.")

    output.append("=" * 60)

    return "\n".join(output)


def plot_comparison(match_occ, control_occ, results, output_dir='outputs/rq1_analysis'):
    """Create visualization comparing match vs non-match occupancy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Box plot
    ax1 = axes[0]
    data = [control_occ, match_occ]
    bp = ax1.boxplot(data, labels=['Non-Match Day', 'Match Day'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('coral')
    ax1.set_ylabel('Occupancy (%)')
    ax1.set_title('Occupancy Distribution Comparison')
    ax1.grid(True, alpha=0.3)

    # Add mean markers
    means = [control_occ.mean(), match_occ.mean()]
    ax1.scatter([1, 2], means, color='red', marker='D', s=50, zorder=5, label='Mean')
    ax1.legend()

    # 2. Density plot
    ax2 = axes[1]
    sns.kdeplot(data=control_occ, ax=ax2, label='Non-Match Day', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(data=match_occ, ax=ax2, label='Match Day', color='red', fill=True, alpha=0.3)
    ax2.axvline(control_occ.mean(), color='blue', linestyle='--', alpha=0.7)
    ax2.axvline(match_occ.mean(), color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Occupancy (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Occupancy Density Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Summary statistics
    ax3 = axes[2]
    ax3.axis('off')

    summary_text = f"""
    Statistical Summary
    ───────────────────────

    Sample Sizes:
      Match Day:     {results['n_match']:,}
      Non-Match Day: {results['n_control']:,}

    Mean Occupancy:
      Match Day:     {results['mean_match']:.2f}%
      Non-Match Day: {results['mean_control']:.2f}%
      Difference:    {results['mean_match']-results['mean_control']:+.2f}%

    Mann-Whitney U Test:
      p-value: {results['mannwhitney_p']:.2e}
      Significant: {'Yes' if results['significant_mw'] else 'No'}

    Effect Size:
      Cohen's d: {results['cohens_d']:.3f}
      Magnitude: {results['effect_interpretation'].upper()}
    """

    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = Path(output_dir) / 'match_day_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def analyze_by_hour(occ_df, match_windows, output_dir='outputs/rq1_analysis'):
    """Analyze occupancy difference by hour relative to match time."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    match_dates = {w['date'] for w in match_windows}
    match_hours_by_date = {w['date']: w['match_hour'] for w in match_windows}

    # Calculate relative hour (hours from match start)
    hourly_data = []

    for window in match_windows:
        match_date = window['date']
        match_hour = window['match_hour']
        match_dow = window['dow']

        # Match day data
        match_day = occ_df[occ_df['date'] == match_date].copy()
        match_day['relative_hour'] = match_day['hour'] - match_hour
        match_day['is_match'] = True

        # Control data (same DOW, non-match dates)
        control_day = occ_df[
            (occ_df['dow'] == match_dow) &
            (~occ_df['date'].isin(match_dates))
        ].copy()
        control_day['relative_hour'] = control_day['hour'] - match_hour
        control_day['is_match'] = False

        hourly_data.append(match_day)
        hourly_data.append(control_day)

    if not hourly_data:
        return None

    combined = pd.concat(hourly_data, ignore_index=True)

    # Aggregate by relative hour
    hourly_stats = combined.groupby(['relative_hour', 'is_match'])['occ_pct'].agg(['mean', 'std', 'count']).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for is_match, group in hourly_stats.groupby('is_match'):
        label = 'Match Day' if is_match else 'Non-Match Day'
        color = 'red' if is_match else 'blue'
        ax.plot(group['relative_hour'], group['mean'], marker='o', label=label, color=color, linewidth=2)
        ax.fill_between(
            group['relative_hour'],
            group['mean'] - group['std'],
            group['mean'] + group['std'],
            alpha=0.2, color=color
        )

    ax.axvline(0, color='green', linestyle='--', label='Match Start', alpha=0.7)
    ax.axvline(2, color='orange', linestyle='--', label='Match End (~)', alpha=0.7)
    ax.axvspan(-2, 0, alpha=0.1, color='yellow', label='Pre-match window')
    ax.axvspan(2, 4, alpha=0.1, color='purple', label='Post-match window')

    ax.set_xlabel('Hours Relative to Match Start')
    ax.set_ylabel('Mean Occupancy (%)')
    ax.set_title('Occupancy Pattern Around Match Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'hourly_pattern.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return hourly_stats


def save_results(results, output_dir='outputs/rq1_analysis'):
    """Save results to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([results])
    output_path = Path(output_dir) / 'statistical_results.csv'
    results_df.to_csv(output_path, index=False)

    return output_path


def analyze_team_full_period(occ_df, match_df, team='Hertha BSC', hours_before=3, hours_after=3):
    """
    Analyze all matches for a specific team across the full data period.
    Compare same DOW + same hour windows.

    Args:
        hours_before: Hours before match to include
        hours_after: Hours after match END to include (match duration ~2h)
    """
    print("\n" + "=" * 70)
    print(f"FULL PERIOD ANALYSIS: {team}")
    print(f"Time window: {hours_before}h before ~ {hours_after}h after match end")
    print("Same Day-of-Week + Same Hour Range Comparison")
    print("=" * 70)

    # Get all dates with occupancy data
    occ_dates = set(occ_df['date'].unique())

    # Filter team matches
    team_matches = match_df[match_df['Team'] == team].copy()
    team_matches['date'] = team_matches['match_date'].dt.date

    # Filter to matches with occupancy data
    team_matches = team_matches[team_matches['date'].isin(occ_dates)].copy()
    team_matches['dow'] = team_matches['match_date'].dt.dayofweek

    match_dates = set(team_matches['date'].values)
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    print(f"\n{team} matches with occupancy data: {len(team_matches)}")

    all_results = []
    all_match_obs = []
    all_control_obs = []

    for _, match in team_matches.iterrows():
        match_date = match['date']
        match_hour = int(match['match_hour'])
        match_dow = match['dow']

        # Define time windows (before match, after match end)
        # Match duration ~2 hours
        pre_match_hours = list(range(max(0, match_hour - hours_before), match_hour))
        post_match_hours = list(range(match_hour + 2, min(24, match_hour + 2 + hours_after)))
        all_hours = pre_match_hours + post_match_hours

        if not all_hours:
            continue

        # Match day data
        match_day_data = occ_df[
            (occ_df['date'] == match_date) &
            (occ_df['hour'].isin(all_hours))
        ]['occ_pct']

        # Control: same DOW, same hours, non-match dates
        control_data = occ_df[
            (occ_df['dow'] == match_dow) &
            (occ_df['hour'].isin(all_hours)) &
            (~occ_df['date'].isin(match_dates))
        ]['occ_pct']

        if len(match_day_data) >= 5 and len(control_data) >= 5:
            all_match_obs.extend(match_day_data.values)
            all_control_obs.extend(control_data.values)

            all_results.append({
                'date': match_date,
                'dow': dow_names[match_dow],
                'match_hour': match_hour,
                'n_match': len(match_day_data),
                'n_control': len(control_data),
                'mean_match': match_day_data.mean(),
                'mean_control': control_data.mean(),
                'diff': match_day_data.mean() - control_data.mean(),
            })

    if not all_results:
        print("No valid comparisons found")
        return None

    results_df = pd.DataFrame(all_results)

    # Individual match results
    print(f"\nIndividual Match Results ({len(results_df)} matches):")
    print("-" * 70)
    print(f"{'Date':<12} {'DOW':<5} {'Hour':<6} {'Match%':>8} {'Control%':>10} {'Diff':>8}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{str(row['date']):<12} {row['dow']:<5} {row['match_hour']:02d}:00  {row['mean_match']:>8.1f} {row['mean_control']:>10.1f} {row['diff']:>+8.1f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print(f"POOLED ANALYSIS: All {team} Matches")
    print("=" * 70)

    all_match_obs = np.array(all_match_obs)
    all_control_obs = np.array(all_control_obs)

    print(f"\nTotal match window observations: {len(all_match_obs)}")
    print(f"Total control observations: {len(all_control_obs)}")

    print(f"\n📊 Descriptive Statistics:")
    print(f"   {'':20} {'Match Day':>12} {'Control':>12} {'Diff':>10}")
    print(f"   {'Mean':20} {all_match_obs.mean():>12.2f}% {all_control_obs.mean():>12.2f}% {all_match_obs.mean()-all_control_obs.mean():>+10.2f}%")
    print(f"   {'Median':20} {np.median(all_match_obs):>12.2f}% {np.median(all_control_obs):>12.2f}% {np.median(all_match_obs)-np.median(all_control_obs):>+10.2f}%")
    print(f"   {'Std':20} {all_match_obs.std():>12.2f}% {all_control_obs.std():>12.2f}%")

    # Statistical tests
    print(f"\n🔬 Statistical Tests (H₁: Match Day > Control):")

    # Mann-Whitney U
    u_stat, mw_p = stats.mannwhitneyu(all_match_obs, all_control_obs, alternative='greater')
    print(f"\n   Mann-Whitney U Test:")
    print(f"      U-statistic: {u_stat:,.0f}")
    print(f"      p-value: {mw_p:.4e}")
    print(f"      Significant (α=0.05): {'✓ YES' if mw_p < 0.05 else '✗ NO'}")

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(all_match_obs, all_control_obs, equal_var=False, alternative='greater')
    print(f"\n   Welch's t-test:")
    print(f"      t-statistic: {t_stat:.4f}")
    print(f"      p-value: {t_p:.4e}")
    print(f"      Significant (α=0.05): {'✓ YES' if t_p < 0.05 else '✗ NO'}")

    # Effect size
    d = cohens_d(pd.Series(all_match_obs), pd.Series(all_control_obs))
    print(f"\n📏 Effect Size:")
    print(f"   Cohen's d: {d:.4f} ({interpret_cohens_d(d)})")

    # Bootstrap CI
    ci_lower, ci_upper, _ = bootstrap_ci(all_match_obs, all_control_obs)
    print(f"   95% Bootstrap CI for mean diff: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

    # By DOW analysis
    print(f"\n📅 By Day-of-Week:")
    dow_summary = results_df.groupby('dow').agg({
        'n_match': 'sum',
        'mean_match': 'mean',
        'mean_control': 'mean',
        'diff': 'mean'
    }).round(2)
    print(dow_summary.to_string())

    return results_df, all_match_obs, all_control_obs


def analyze_march_2024(occ_df, match_df):
    """
    Focused analysis on March 2024 data only.
    Compare same DOW + same hour windows.
    """
    print("\n" + "=" * 70)
    print("FOCUSED ANALYSIS: March 2024 Only")
    print("Same Day-of-Week + Same Hour Range Comparison")
    print("=" * 70)

    # Filter to March 2024
    occ_march = occ_df[
        (occ_df['date'] >= pd.Timestamp('2024-03-01').date()) &
        (occ_df['date'] <= pd.Timestamp('2024-03-31').date())
    ].copy()

    # March 2024 matches
    matches_march = [
        {'date': pd.Timestamp('2024-03-01').date(), 'hour': 18, 'dow': 4, 'team': 'Hertha BSC'},
        {'date': pd.Timestamp('2024-03-16').date(), 'hour': 15, 'dow': 5, 'team': 'Union Berlin'},
        {'date': pd.Timestamp('2024-03-17').date(), 'hour': 13, 'dow': 6, 'team': 'Hertha BSC'},
        {'date': pd.Timestamp('2024-03-30').date(), 'hour': 20, 'dow': 5, 'team': 'Hertha BSC'},
    ]

    match_dates = {m['date'] for m in matches_march}
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    all_results = []

    for match in matches_march:
        match_date = match['date']
        match_hour = match['hour']
        match_dow = match['dow']
        team = match['team']

        # Check if data exists for this match date
        if match_date not in occ_march['date'].values:
            print(f"\n⚠️  {match_date} ({team}): No data available")
            continue

        # Define time windows
        pre_match_hours = list(range(max(0, match_hour - 2), match_hour))
        post_match_hours = list(range(match_hour + 2, min(24, match_hour + 4)))
        all_hours = pre_match_hours + post_match_hours

        print(f"\n{'='*60}")
        print(f"🏟️  {match_date} ({dow_names[match_dow]}) @ {match_hour:02d}:00 - {team}")
        print(f"   Pre-match: {min(pre_match_hours):02d}:00-{max(pre_match_hours):02d}:59" if pre_match_hours else "   Pre-match: N/A")
        print(f"   Post-match: {min(post_match_hours):02d}:00-{max(post_match_hours):02d}:59" if post_match_hours else "   Post-match: N/A")

        # Match day data
        match_day_data = occ_march[
            (occ_march['date'] == match_date) &
            (occ_march['hour'].isin(all_hours))
        ]['occ_pct']

        # Control: same DOW, same hours, non-match dates in March
        control_dates = occ_march[
            (occ_march['dow'] == match_dow) &
            (~occ_march['date'].isin(match_dates))
        ]['date'].unique()

        control_data = occ_march[
            (occ_march['dow'] == match_dow) &
            (occ_march['hour'].isin(all_hours)) &
            (~occ_march['date'].isin(match_dates))
        ]['occ_pct']

        print(f"\n   Match day observations: {len(match_day_data)}")
        print(f"   Control {dow_names[match_dow]}s: {sorted([str(d) for d in control_dates])}")
        print(f"   Control observations: {len(control_data)}")

        if len(match_day_data) < 5 or len(control_data) < 5:
            print(f"   ⚠️  Insufficient data for statistical test")
            continue

        # Statistics
        print(f"\n   📊 Occupancy Statistics:")
        print(f"      {'':15} {'Match Day':>12} {'Control':>12}")
        print(f"      {'Mean':15} {match_day_data.mean():>12.2f}% {control_data.mean():>12.2f}%")
        print(f"      {'Median':15} {match_day_data.median():>12.2f}% {control_data.median():>12.2f}%")
        print(f"      {'Std':15} {match_day_data.std():>12.2f}% {control_data.std():>12.2f}%")

        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(match_day_data, control_data, alternative='greater')
        d = cohens_d(match_day_data, control_data)

        print(f"\n   🔬 Mann-Whitney U Test (H₁: Match > Control):")
        print(f"      U-statistic: {u_stat:,.0f}")
        print(f"      p-value: {p_value:.4f}")
        print(f"      Significant (α=0.05): {'✓ YES' if p_value < 0.05 else '✗ NO'}")
        print(f"      Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")

        all_results.append({
            'date': match_date,
            'team': team,
            'dow': dow_names[match_dow],
            'match_hour': match_hour,
            'n_match': len(match_day_data),
            'n_control': len(control_data),
            'mean_match': match_day_data.mean(),
            'mean_control': control_data.mean(),
            'diff': match_day_data.mean() - control_data.mean(),
            'p_value': p_value,
            'cohens_d': d,
            'significant': p_value < 0.05
        })

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY: March 2024 Match Day Analysis")
        print("=" * 70)

        results_df = pd.DataFrame(all_results)
        print(f"\nMatches analyzed: {len(results_df)}")
        print(f"Significant results (p < 0.05): {results_df['significant'].sum()}")

        print("\n" + results_df[['date', 'team', 'mean_match', 'mean_control', 'diff', 'p_value', 'cohens_d']].to_string(index=False))

        # Overall pooled analysis
        print("\n" + "-" * 40)
        print("POOLED ANALYSIS (All March 2024 matches)")
        print("-" * 40)

        all_match_data = []
        all_control_data = []

        for match in matches_march:
            match_date = match['date']
            match_hour = match['hour']
            match_dow = match['dow']

            if match_date not in occ_march['date'].values:
                continue

            pre_match_hours = list(range(max(0, match_hour - 2), match_hour))
            post_match_hours = list(range(match_hour + 2, min(24, match_hour + 4)))
            all_hours = pre_match_hours + post_match_hours

            match_data = occ_march[
                (occ_march['date'] == match_date) &
                (occ_march['hour'].isin(all_hours))
            ]['occ_pct'].values

            control_data = occ_march[
                (occ_march['dow'] == match_dow) &
                (occ_march['hour'].isin(all_hours)) &
                (~occ_march['date'].isin(match_dates))
            ]['occ_pct'].values

            all_match_data.extend(match_data)
            all_control_data.extend(control_data)

        all_match_data = np.array(all_match_data)
        all_control_data = np.array(all_control_data)

        print(f"Total match observations: {len(all_match_data)}")
        print(f"Total control observations: {len(all_control_data)}")
        print(f"Match mean: {all_match_data.mean():.2f}%")
        print(f"Control mean: {all_control_data.mean():.2f}%")
        print(f"Difference: {all_match_data.mean() - all_control_data.mean():+.2f}%")

        u_stat, p_value = stats.mannwhitneyu(all_match_data, all_control_data, alternative='greater')
        d = cohens_d(pd.Series(all_match_data), pd.Series(all_control_data))

        print(f"\nMann-Whitney U p-value: {p_value:.4f}")
        print(f"Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")
        print(f"Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")

        return results_df

    return None


# =============================================================================
# RQ1 REVISED: Forecasting with Historical Patterns + External Factors
# =============================================================================

"""
Revised RQ1: Can we predict seat availability using only historical patterns
and external factors (match day, weather, holiday) WITHOUT real-time occupancy?

Key difference from original:
- Original: Uses current occupancy (occ_pct) as feature
- Revised: Uses only historical averages + external factors (available at prediction time)
"""


def create_historical_features(df, match_df=None):
    """
    Create historical pattern features that are available at prediction time.

    Features created:
    - hist_avg_stop_hour_dow: Historical avg for this stop/hour/day-of-week
    - hist_avg_stop_hour: Historical avg for this stop/hour (all days)
    - hist_avg_stop_dow: Historical avg for this stop/day-of-week
    - hist_std_stop_hour_dow: Historical std for variability
    """
    df = df.copy()

    # Ensure datetime columns
    if not pd.api.types.is_datetime64_any_dtype(df['trip_start_date']):
        df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
    if not pd.api.types.is_datetime64_any_dtype(df['trip_dep_time']):
        df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])

    # Extract time features if not present
    if 'hour' not in df.columns:
        df['hour'] = df['trip_dep_time'].dt.hour
    if 'dow' not in df.columns:
        df['dow'] = df['trip_start_date'].dt.dayofweek
    if 'date' not in df.columns:
        df['date'] = df['trip_start_date'].dt.date

    # Calculate occupancy if not present
    if 'occ_pct' not in df.columns:
        df['occ_pct'] = (df['occupancy'] / df['capacity'] * 100).round(2)

    print("  Creating historical average features...")

    # 1. Historical average by stop/hour/dow (most specific)
    hist_stop_hour_dow = df.groupby(['dep_id', 'hour', 'dow'])['occ_pct'].agg(['mean', 'std']).reset_index()
    hist_stop_hour_dow.columns = ['dep_id', 'hour', 'dow', 'hist_avg_stop_hour_dow', 'hist_std_stop_hour_dow']
    df = df.merge(hist_stop_hour_dow, on=['dep_id', 'hour', 'dow'], how='left')

    # 2. Historical average by stop/hour (across all days)
    hist_stop_hour = df.groupby(['dep_id', 'hour'])['occ_pct'].mean().reset_index()
    hist_stop_hour.columns = ['dep_id', 'hour', 'hist_avg_stop_hour']
    df = df.merge(hist_stop_hour, on=['dep_id', 'hour'], how='left')

    # 3. Historical average by stop/dow (across all hours)
    hist_stop_dow = df.groupby(['dep_id', 'dow'])['occ_pct'].mean().reset_index()
    hist_stop_dow.columns = ['dep_id', 'dow', 'hist_avg_stop_dow']
    df = df.merge(hist_stop_dow, on=['dep_id', 'dow'], how='left')

    # 4. Global historical average by stop
    hist_stop = df.groupby('dep_id')['occ_pct'].mean().reset_index()
    hist_stop.columns = ['dep_id', 'hist_avg_stop']
    df = df.merge(hist_stop, on=['dep_id'], how='left')

    # Fill NaN with global averages
    df['hist_std_stop_hour_dow'] = df['hist_std_stop_hour_dow'].fillna(df['occ_pct'].std())

    return df


def add_external_factors(df, match_df=None):
    """
    Add external factors that are known at prediction time.

    Factors:
    - is_match_day: Whether there's a match on this date
    - is_holiday: German public holiday
    - is_weekend: Saturday or Sunday
    - is_rush_hour: 6-9 or 16-19
    """
    df = df.copy()

    # Ensure date column
    if 'date' not in df.columns:
        df['date'] = df['trip_start_date'].dt.date

    # 1. Holiday flag (Germany, Brandenburg)
    print("  Adding holiday flags...")
    years = sorted(df['trip_start_date'].dt.year.unique())
    de_holidays = holidays.country_holidays('DE', subdiv='BB', years=years)
    df['is_holiday'] = df['trip_start_date'].dt.date.apply(lambda d: 1 if d in de_holidays else 0)

    # 2. Match day flag
    if match_df is not None:
        print("  Adding match day flags...")
        match_dates = set(match_df['match_date'].dt.date.dropna())
        df['is_match_day'] = df['date'].apply(lambda d: 1 if d in match_dates else 0)
    else:
        df['is_match_day'] = 0

    # 3. Weekend flag
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # 4. Rush hour flag
    if 'is_rush_hour' not in df.columns:
        df['is_rush_hour'] = df['hour'].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)

    # 5. Night flag
    if 'is_night' not in df.columns:
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    return df


def create_prediction_targets(df, horizons_minutes={'30min': 30, '1h': 60, '3h': 180, 'next_day': 1440}):
    """
    Create target variables for different prediction horizons.

    For each observation, find the occupancy at the same stop X minutes later.
    """
    df = df.copy()
    df = df.sort_values(['dep_id', 'trip_dep_time']).reset_index(drop=True)

    print("  Creating prediction targets...")

    for name, minutes in horizons_minutes.items():
        print(f"    - {name} ({minutes} minutes ahead)...")

        # For each stop, find future occupancy
        targets = []

        for dep_id in df['dep_id'].unique():
            stop_df = df[df['dep_id'] == dep_id].copy()
            stop_df = stop_df.sort_values('trip_dep_time')

            for idx, row in stop_df.iterrows():
                current_time = row['trip_dep_time']
                target_time = current_time + pd.Timedelta(minutes=minutes)

                # Find closest observation at or after target time (same stop)
                future_obs = stop_df[stop_df['trip_dep_time'] >= target_time]

                if len(future_obs) > 0:
                    # Get the first observation after target time
                    target_occ = future_obs.iloc[0]['occ_pct']
                else:
                    target_occ = np.nan

                targets.append({'idx': idx, f'target_{name}': target_occ})

        target_df = pd.DataFrame(targets).set_index('idx')
        df = df.join(target_df)

    return df


def prepare_forecast_data(df, horizon='30min', feature_set='base'):
    """
    Prepare X, y for forecasting model.

    Feature sets:
    - 'base': Only historical patterns + time features
    - 'extended': Base + external factors (match day, holiday)
    """
    base_features = [
        'hour', 'dow', 'is_weekend', 'is_rush_hour', 'is_night',
        'hist_avg_stop_hour_dow', 'hist_avg_stop_hour', 'hist_avg_stop_dow',
        'hist_std_stop_hour_dow'
    ]

    external_features = [
        'is_match_day', 'is_holiday'
    ]

    if feature_set == 'extended':
        features = base_features + external_features
    else:
        features = base_features

    target_col = f'target_{horizon}'

    # Filter valid rows
    required_cols = features + [target_col]
    df_clean = df.dropna(subset=required_cols)

    X = df_clean[features]
    y = df_clean[target_col]

    return X, y, features, df_clean


def train_forecast_model(X_train, y_train, random_state=42):
    """Train RandomForest forecasting model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_forecast_model(model, X_test, y_test):
    """Evaluate forecasting model."""
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'samples': len(y_test)
    }
    return metrics, y_pred


def run_rq1_forecast_analysis(output_dir='outputs/analysis'):
    """
    Run complete RQ1 forecast analysis.

    Compare base model (historical patterns only) vs extended model (+ external factors)
    across different prediction horizons.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("RQ1 REVISED: Forecasting with Historical Patterns + External Factors")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    occ_df = pd.read_csv('data/ingested/RE5_2024_03.csv')
    occ_df['trip_start_date'] = pd.to_datetime(occ_df['trip_start_date'])
    occ_df['trip_dep_time'] = pd.to_datetime(occ_df['trip_dep_time'])

    # Filter quality
    occ_df = occ_df[occ_df['capacity'] >= 100].copy()
    occ_df['occ_pct'] = (occ_df['occupancy'] / occ_df['capacity'] * 100).round(2)
    occ_df['hour'] = occ_df['trip_dep_time'].dt.hour
    occ_df['dow'] = occ_df['trip_start_date'].dt.dayofweek
    occ_df['date'] = occ_df['trip_start_date'].dt.date

    print(f"  Loaded {len(occ_df):,} observations")

    # Load match schedule
    match_df = pd.read_excel('data/raw/Schedule_Teams_2022_2024.xlsx', header=1)
    match_df['match_date'] = pd.to_datetime(match_df['Date'], errors='coerce')
    print(f"  Loaded {len(match_df):,} matches")

    # Create features
    print("\n[2/6] Creating historical features...")
    occ_df = create_historical_features(occ_df)

    print("\n[3/6] Adding external factors...")
    occ_df = add_external_factors(occ_df, match_df)

    # Create targets (simplified: use shift-based approach for efficiency)
    print("\n[4/6] Creating prediction targets...")
    horizons = {'30min': 1, '1h': 2, '3h': 6, 'next_day': 48}

    occ_df = occ_df.sort_values(['dep_id', 'trip_dep_time']).reset_index(drop=True)

    for name, shift in horizons.items():
        occ_df[f'target_{name}'] = occ_df.groupby('dep_id')['occ_pct'].shift(-shift)

    # Train and evaluate models
    print("\n[5/6] Training and evaluating models...")

    results = []
    models = {}

    for horizon in horizons.keys():
        print(f"\n  --- Horizon: {horizon} ---")

        for feature_set in ['base', 'extended']:
            try:
                X, y, features, df_clean = prepare_forecast_data(occ_df, horizon, feature_set)

                if len(X) < 100:
                    print(f"    {feature_set}: Insufficient data ({len(X)} samples)")
                    continue

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train model
                model = train_forecast_model(X_train, y_train)

                # Evaluate
                metrics, y_pred = evaluate_forecast_model(model, X_test, y_test)

                print(f"    {feature_set}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.3f}")

                results.append({
                    'horizon': horizon,
                    'feature_set': feature_set,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'samples': metrics['samples']
                })

                # Save model
                model_key = f"{horizon}_{feature_set}"
                models[model_key] = {
                    'model': model,
                    'features': features,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
                }

                # Save model file
                model_path = Path(output_dir) / f'model_{model_key}.joblib'
                joblib.dump(model, model_path)

            except Exception as e:
                print(f"    {feature_set}: Error - {e}")
                continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate improvement from external factors
    print("\n[6/6] Analyzing results...")

    improvement_data = []
    for horizon in horizons.keys():
        base_row = results_df[(results_df['horizon'] == horizon) & (results_df['feature_set'] == 'base')]
        ext_row = results_df[(results_df['horizon'] == horizon) & (results_df['feature_set'] == 'extended')]

        if len(base_row) > 0 and len(ext_row) > 0:
            base_mae = base_row['MAE'].values[0]
            ext_mae = ext_row['MAE'].values[0]
            base_r2 = base_row['R2'].values[0]
            ext_r2 = ext_row['R2'].values[0]

            mae_improvement = ((base_mae - ext_mae) / base_mae) * 100
            r2_improvement = ext_r2 - base_r2

            improvement_data.append({
                'horizon': horizon,
                'base_MAE': base_mae,
                'extended_MAE': ext_mae,
                'MAE_improvement_%': mae_improvement,
                'base_R2': base_r2,
                'extended_R2': ext_r2,
                'R2_improvement': r2_improvement
            })

    improvement_df = pd.DataFrame(improvement_data)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 Model Performance by Horizon:")
    print(results_df.to_string(index=False))

    print("\n📈 Improvement from External Factors (Match Day, Holiday):")
    print(improvement_df.to_string(index=False))

    # Interpretation
    print("\n💡 Interpretation:")
    avg_mae_improvement = improvement_df['MAE_improvement_%'].mean()
    avg_r2_improvement = improvement_df['R2_improvement'].mean()

    if avg_mae_improvement > 0:
        print(f"   ✓ External factors IMPROVE prediction accuracy")
        print(f"   ✓ Average MAE reduction: {avg_mae_improvement:.2f}%")
        print(f"   ✓ Average R² improvement: {avg_r2_improvement:.4f}")
    else:
        print(f"   ✗ External factors do not significantly improve predictions")
        print(f"   ✗ Average MAE change: {avg_mae_improvement:.2f}%")

    # Save results
    results_df.to_csv(Path(output_dir) / 'rq1_model_performance.csv', index=False)
    improvement_df.to_csv(Path(output_dir) / 'rq1_external_factor_improvement.csv', index=False)

    # Create visualization
    create_rq1_visualizations(results_df, improvement_df, models, output_dir)

    print(f"\n✓ Results saved to {output_dir}/")

    return results_df, improvement_df, models


def create_rq1_visualizations(results_df, improvement_df, models, output_dir):
    """Create RQ1 visualizations."""

    # 1. Model performance comparison (Bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE comparison
    ax1 = axes[0]
    horizons = results_df['horizon'].unique()
    x = np.arange(len(horizons))
    width = 0.35

    base_mae = results_df[results_df['feature_set'] == 'base']['MAE'].values
    ext_mae = results_df[results_df['feature_set'] == 'extended']['MAE'].values

    ax1.bar(x - width/2, base_mae, width, label='Base (Historical only)', color='steelblue')
    ax1.bar(x + width/2, ext_mae, width, label='Extended (+External factors)', color='coral')
    ax1.set_xlabel('Prediction Horizon')
    ax1.set_ylabel('MAE (%)')
    ax1.set_title('Prediction Error by Horizon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # R² comparison
    ax2 = axes[1]
    base_r2 = results_df[results_df['feature_set'] == 'base']['R2'].values
    ext_r2 = results_df[results_df['feature_set'] == 'extended']['R2'].values

    ax2.bar(x - width/2, base_r2, width, label='Base (Historical only)', color='steelblue')
    ax2.bar(x + width/2, ext_r2, width, label='Extended (+External factors)', color='coral')
    ax2.set_xlabel('Prediction Horizon')
    ax2.set_ylabel('R²')
    ax2.set_title('Prediction Accuracy (R²) by Horizon')
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature importance (for extended model, 1h horizon)
    if '1h_extended' in models:
        model_info = models['1h_extended']
        model = model_info['model']
        features = model_info['features']

        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['feature'], importance['importance'], color='steelblue')
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance for 1-Hour Ahead Prediction (Extended Model)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq1_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Scatter plot: Actual vs Predicted (for each horizon)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, horizon in enumerate(['30min', '1h', '3h', 'next_day']):
        ax = axes[i]
        model_key = f'{horizon}_extended'

        if model_key in models:
            y_test = models[model_key]['y_test']
            y_pred = models[model_key]['y_pred']

            # Sample for visualization
            sample_size = min(2000, len(y_test))
            idx = np.random.choice(len(y_test), sample_size, replace=False)

            ax.scatter(y_test.iloc[idx], y_pred[idx], alpha=0.3, s=10, c='steelblue')
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect prediction')

            # Add regression line
            z = np.polyfit(y_test.iloc[idx], y_pred[idx], 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 100, 100)
            ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, label=f'Fit (slope={z[0]:.2f})')

            ax.set_xlabel('Actual Occupancy (%)')
            ax.set_ylabel('Predicted Occupancy (%)')
            ax.set_title(f'{horizon} Horizon (R²={models[model_key]["y_test"].corr(pd.Series(y_pred)):.3f})')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq1_scatter_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Line chart: Performance trend by horizon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    horizon_order = ['30min', '1h', '3h', 'next_day']
    horizon_labels = ['30min', '1h', '3h', 'Next Day']

    # MAE trend
    ax1 = axes[0]
    base_mae = [results_df[(results_df['horizon'] == h) & (results_df['feature_set'] == 'base')]['MAE'].values[0] for h in horizon_order]
    ext_mae = [results_df[(results_df['horizon'] == h) & (results_df['feature_set'] == 'extended')]['MAE'].values[0] for h in horizon_order]

    ax1.plot(horizon_labels, base_mae, 'o-', linewidth=2, markersize=10, label='Base', color='steelblue')
    ax1.plot(horizon_labels, ext_mae, 's-', linewidth=2, markersize=10, label='Extended', color='coral')
    ax1.set_xlabel('Prediction Horizon')
    ax1.set_ylabel('MAE (%)')
    ax1.set_title('Prediction Error Trend by Horizon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # R² trend
    ax2 = axes[1]
    base_r2 = [results_df[(results_df['horizon'] == h) & (results_df['feature_set'] == 'base')]['R2'].values[0] for h in horizon_order]
    ext_r2 = [results_df[(results_df['horizon'] == h) & (results_df['feature_set'] == 'extended')]['R2'].values[0] for h in horizon_order]

    ax2.plot(horizon_labels, base_r2, 'o-', linewidth=2, markersize=10, label='Base', color='steelblue')
    ax2.plot(horizon_labels, ext_r2, 's-', linewidth=2, markersize=10, label='Extended', color='coral')
    ax2.set_xlabel('Prediction Horizon')
    ax2.set_ylabel('R²')
    ax2.set_title('Prediction Accuracy Trend by Horizon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq1_performance_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Residual distribution (histogram)
    if '1h_extended' in models:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        y_test = models['1h_extended']['y_test']
        y_pred = models['1h_extended']['y_pred']
        residuals = y_test.values - y_pred

        # Histogram
        ax1 = axes[0]
        ax1.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax1.axvline(residuals.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean={residuals.mean():.2f}')
        ax1.set_xlabel('Residual (Actual - Predicted)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Residual Distribution (1h Horizon)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual vs Predicted
        ax2 = axes[1]
        sample_size = min(2000, len(y_pred))
        idx = np.random.choice(len(y_pred), sample_size, replace=False)
        ax2.scatter(y_pred[idx], residuals[idx], alpha=0.3, s=10, c='steelblue')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Occupancy (%)')
        ax2.set_ylabel('Residual (Actual - Predicted)')
        ax2.set_title('Residuals vs Predicted Values')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq1_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Error by hour (heatmap-style)
    if '1h_extended' in models:
        X_test = models['1h_extended']['X_test']
        y_test = models['1h_extended']['y_test']
        y_pred = models['1h_extended']['y_pred']

        error_df = pd.DataFrame({
            'hour': X_test['hour'].values,
            'dow': X_test['dow'].values,
            'abs_error': np.abs(y_test.values - y_pred)
        })

        # Error by hour
        hourly_error = error_df.groupby('hour')['abs_error'].mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(hourly_error.index, hourly_error.values, color='steelblue', edgecolor='white')

        # Highlight rush hours
        for i, bar in enumerate(bars):
            if hourly_error.index[i] in [6, 7, 8, 9, 16, 17, 18, 19]:
                bar.set_color('coral')

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Absolute Error (%)')
        ax.set_title('Prediction Error by Hour (Orange = Rush Hour)')
        ax.set_xticks(range(24))
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq1_error_by_hour.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("  Visualizations saved.")


# =============================================================================
# RQ2: SHAP-based Explainability
# =============================================================================

"""
RQ2: How do external factors shape SHAP-based explanations of seat availability
predictions in special scenarios (match days, holidays, etc.)?
"""


def run_rq2_shap_analysis(models, occ_df, output_dir='outputs/analysis'):
    """
    Run SHAP analysis for model explainability.

    Analyzes how external factors influence predictions in different scenarios.
    """
    try:
        import shap
    except ImportError:
        print("\n⚠️  SHAP not installed. Install with: pip install shap")
        print("  Skipping RQ2 analysis.")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("RQ2: SHAP-based Explainability Analysis")
    print("=" * 70)

    # Use 1h extended model for analysis
    if '1h_extended' not in models:
        print("  No extended model available for SHAP analysis")
        return None

    model_info = models['1h_extended']
    model = model_info['model']
    features = model_info['features']
    X_test = model_info['X_test']

    print("\n[1/4] Computing SHAP values...")

    # Use a sample for efficiency
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"  Computed SHAP values for {sample_size} samples")

    # 1. Overall feature importance
    print("\n[2/4] Creating SHAP summary...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq2_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature importance bar
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rq2_shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Analyze special scenarios
    print("\n[3/4] Analyzing special scenarios...")

    scenarios = {}

    # Match day vs non-match day
    if 'is_match_day' in X_sample.columns:
        match_idx = X_sample['is_match_day'] == 1
        non_match_idx = X_sample['is_match_day'] == 0

        if match_idx.sum() > 0 and non_match_idx.sum() > 0:
            match_shap_mean = np.abs(shap_values[match_idx]).mean(axis=0)
            non_match_shap_mean = np.abs(shap_values[non_match_idx]).mean(axis=0)

            scenarios['match_day'] = {
                'n_samples': match_idx.sum(),
                'mean_abs_shap': dict(zip(features, match_shap_mean))
            }
            scenarios['non_match_day'] = {
                'n_samples': non_match_idx.sum(),
                'mean_abs_shap': dict(zip(features, non_match_shap_mean))
            }

            print(f"    Match day samples: {match_idx.sum()}")
            print(f"    Non-match day samples: {non_match_idx.sum()}")

    # Holiday vs non-holiday
    if 'is_holiday' in X_sample.columns:
        holiday_idx = X_sample['is_holiday'] == 1
        non_holiday_idx = X_sample['is_holiday'] == 0

        if holiday_idx.sum() > 0:
            scenarios['holiday'] = {
                'n_samples': holiday_idx.sum(),
                'mean_abs_shap': dict(zip(features, np.abs(shap_values[holiday_idx]).mean(axis=0)))
            }
            print(f"    Holiday samples: {holiday_idx.sum()}")

    # Rush hour vs non-rush hour
    if 'is_rush_hour' in X_sample.columns:
        rush_idx = X_sample['is_rush_hour'] == 1
        non_rush_idx = X_sample['is_rush_hour'] == 0

        if rush_idx.sum() > 0 and non_rush_idx.sum() > 0:
            scenarios['rush_hour'] = {
                'n_samples': rush_idx.sum(),
                'mean_abs_shap': dict(zip(features, np.abs(shap_values[rush_idx]).mean(axis=0)))
            }
            scenarios['non_rush_hour'] = {
                'n_samples': non_rush_idx.sum(),
                'mean_abs_shap': dict(zip(features, np.abs(shap_values[non_rush_idx]).mean(axis=0)))
            }
            print(f"    Rush hour samples: {rush_idx.sum()}")

    # 4. Create scenario comparison visualization
    print("\n[4/4] Creating scenario comparison...")

    if 'match_day' in scenarios and 'non_match_day' in scenarios:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(features))
        width = 0.35

        match_importance = [scenarios['match_day']['mean_abs_shap'][f] for f in features]
        non_match_importance = [scenarios['non_match_day']['mean_abs_shap'][f] for f in features]

        ax.bar(x - width/2, match_importance, width, label='Match Day', color='coral')
        ax.bar(x + width/2, non_match_importance, width, label='Non-Match Day', color='steelblue')

        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean |SHAP Value|')
        ax.set_title('Feature Importance: Match Day vs Non-Match Day')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'rq2_scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Save SHAP summary statistics
    shap_importance = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    shap_importance.to_csv(Path(output_dir) / 'rq2_shap_importance.csv', index=False)

    # Summary
    print("\n" + "=" * 70)
    print("RQ2 RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 Overall Feature Importance (by SHAP):")
    print(shap_importance.to_string(index=False))

    print("\n💡 Key Findings:")
    top_feature = shap_importance.iloc[0]['feature']
    print(f"   - Most important feature: {top_feature}")

    if 'is_match_day' in features:
        match_day_rank = shap_importance[shap_importance['feature'] == 'is_match_day'].index[0] + 1
        print(f"   - Match day importance rank: #{match_day_rank}")

    if 'is_holiday' in features:
        holiday_rank = shap_importance[shap_importance['feature'] == 'is_holiday'].index[0] + 1
        print(f"   - Holiday importance rank: #{holiday_rank}")

    print(f"\n✓ SHAP analysis saved to {output_dir}/")

    return shap_values, shap_importance, scenarios


def run_full_analysis(output_dir='outputs/analysis'):
    """
    Run complete RQ1 and RQ2 analysis.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("FULL ANALYSIS: RQ1 (Forecasting) + RQ2 (Explainability)")
    print("=" * 70)

    # Run RQ1
    results_df, improvement_df, models = run_rq1_forecast_analysis(output_dir)

    # Load data for RQ2
    occ_df = pd.read_csv('data/ingested/RE5_2024_03.csv')
    occ_df['trip_start_date'] = pd.to_datetime(occ_df['trip_start_date'])
    occ_df['trip_dep_time'] = pd.to_datetime(occ_df['trip_dep_time'])

    # Run RQ2
    shap_results = run_rq2_shap_analysis(models, occ_df, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}/")
    print("  - rq1_model_performance.csv")
    print("  - rq1_external_factor_improvement.csv")
    print("  - rq1_model_comparison.png")
    print("  - rq1_feature_importance.png")
    if shap_results:
        print("  - rq2_shap_summary.png")
        print("  - rq2_shap_importance.png")
        print("  - rq2_shap_importance.csv")
        print("  - rq2_scenario_comparison.png")

    return results_df, improvement_df, models, shap_results


def main():
    """Main analysis pipeline."""
    print("Loading data...")

    # Load occupancy data
    occ_df = load_occupancy_data('data/ingested/RE5_2024_03.csv')
    print(f"  Loaded {len(occ_df):,} occupancy observations")
    print(f"  Date range: {occ_df['date'].min()} to {occ_df['date'].max()}")

    # Load match schedule
    match_df = load_match_schedule('data/raw/Schedule_Teams_2022_2024.xlsx')
    print(f"  Loaded {len(match_df):,} matches")

    # Debug: print match_df columns
    print(f"  Match schedule columns: {list(match_df.columns)}")

    # Create match windows
    print("\nIdentifying match windows...")
    match_windows = identify_match_windows(match_df, hours_before=2, hours_after=2)

    # Filter to dates in occupancy data
    occ_dates = set(occ_df['date'].unique())
    match_windows = [w for w in match_windows if w['date'] in occ_dates]
    print(f"  Found {len(match_windows)} matches within occupancy data period")

    if not match_windows:
        print("\n⚠️  No matches found in the occupancy data period!")
        print("  Check if match dates overlap with occupancy dates.")
        return

    # Classify observations
    print("\nClassifying observations...")
    occ_df = classify_observations(occ_df, match_windows)

    # Get matched groups
    match_obs, control_obs = get_matched_control_group(occ_df, match_windows)
    print(f"  Match window observations: {len(match_obs):,}")
    print(f"  Matched control observations: {len(control_obs):,}")

    if len(match_obs) == 0 or len(control_obs) == 0:
        print("\n⚠️  Insufficient data for comparison!")
        return

    # Run statistical tests
    print("\nRunning statistical tests...")
    results = run_statistical_tests(match_obs['occ_pct'], control_obs['occ_pct'])

    # Display results
    print("\n" + format_results(results))

    # Create visualizations
    print("\nCreating visualizations...")
    plot_path = plot_comparison(match_obs['occ_pct'], control_obs['occ_pct'], results)
    print(f"  Saved: {plot_path}")

    hourly_stats = analyze_by_hour(occ_df, match_windows)
    if hourly_stats is not None:
        print(f"  Saved: outputs/rq1_analysis/hourly_pattern.png")

    # Save results
    results_path = save_results(results)
    print(f"  Saved: {results_path}")

    # Run full period analysis for Hertha BSC
    hertha_results = analyze_team_full_period(occ_df, match_df, team='Hertha BSC')

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
