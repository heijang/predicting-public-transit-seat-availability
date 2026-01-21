"""
Weather Impact Analysis on RE5 Occupancy

Analyzes how weather conditions (rain, temperature drops, snow) affect
seat availability compared to same day-of-week with normal weather.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path('outputs/analysis_weather')
DATA_DIR = Path('data')


# =============================================================================
# Data Loading
# =============================================================================

def load_occupancy_data(filepath='data/ingested/RE5_2024_03.csv'):
    """Load RE5 occupancy data."""
    df = pd.read_csv(filepath)
    df['trip_start_date'] = pd.to_datetime(df['trip_start_date'])
    df['trip_dep_time'] = pd.to_datetime(df['trip_dep_time'])
    df['date'] = df['trip_start_date'].dt.date
    df['hour'] = df['trip_dep_time'].dt.hour
    df['dow'] = df['trip_start_date'].dt.dayofweek

    # Filter: 2022-08 onwards
    df = df[df['trip_start_date'] >= '2022-08-01']

    # Filter: 100+ records per day
    date_counts = df.groupby('date').size()
    valid_dates = date_counts[date_counts >= 100].index
    df = df[df['date'].isin(valid_dates)]

    return df


def load_weather_data(filepath='data/raw/berlin_weather_2022_2024.csv'):
    """Load historical weather data."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def merge_data(occ_df, weather_df):
    """Merge occupancy and weather data."""
    return occ_df.merge(weather_df, on='date', how='left')


# =============================================================================
# Statistical Functions
# =============================================================================

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
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


def bootstrap_ci(data1, data2, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence interval for mean difference."""
    np.random.seed(42)
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        boot_diffs.append(sample1.mean() - sample2.mean())

    alpha = (100 - ci) / 2
    return np.percentile(boot_diffs, alpha), np.percentile(boot_diffs, 100 - alpha)


def run_comparison(treatment_data, control_data, treatment_name, control_name):
    """Run statistical comparison between two groups."""
    results = {
        'treatment_name': treatment_name,
        'control_name': control_name,
        'n_treatment': len(treatment_data),
        'n_control': len(control_data),
        'mean_treatment': treatment_data.mean(),
        'mean_control': control_data.mean(),
        'std_treatment': treatment_data.std(),
        'std_control': control_data.std(),
    }

    # Difference
    results['diff'] = results['mean_treatment'] - results['mean_control']
    results['pct_change'] = (results['diff'] / results['mean_control']) * 100 if results['mean_control'] > 0 else 0

    # Mann-Whitney U test
    u_stat, mw_p = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
    results['mannwhitney_u'] = u_stat
    results['mannwhitney_p'] = mw_p

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(treatment_data, control_data, equal_var=False)
    results['welch_t'] = t_stat
    results['welch_p'] = t_p

    # Cohen's d
    d = cohens_d(treatment_data, control_data)
    results['cohens_d'] = d
    results['effect_size'] = interpret_cohens_d(d)

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(treatment_data.values, control_data.values)
    results['ci_lower'] = ci_lower
    results['ci_upper'] = ci_upper

    # Significance
    results['significant'] = mw_p < 0.05

    return results


# =============================================================================
# Weather Analysis Functions
# =============================================================================

def analyze_weather_impact(df, weather_col, weather_name, same_dow=True):
    """
    Analyze impact of weather condition on occupancy.

    Args:
        df: Merged dataframe with occupancy and weather
        weather_col: Column name for weather condition (binary: 0/1)
        weather_name: Human readable name for the condition
        same_dow: If True, compare same day-of-week only

    Returns:
        dict with analysis results
    """
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Overall comparison
    treatment_dates = df[df[weather_col] == 1]['date'].unique()
    control_dates = df[df[weather_col] == 0]['date'].unique()

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {weather_name}")
    print(f"{'='*60}")
    print(f"  {weather_name} days: {len(treatment_dates)}")
    print(f"  Normal days: {len(control_dates)}")

    if same_dow:
        # Compare same day-of-week
        all_results = []

        for dow in range(7):
            dow_df = df[df['dow_x'] == dow] if 'dow_x' in df.columns else df[df['dow'] == dow]

            treatment = dow_df[dow_df[weather_col] == 1]['occupancy']
            control = dow_df[dow_df[weather_col] == 0]['occupancy']

            if len(treatment) >= 30 and len(control) >= 30:
                result = run_comparison(treatment, control, weather_name, 'Normal')
                result['dow'] = dow_names[dow]
                all_results.append(result)

                sig = '*' if result['significant'] else ''
                print(f"  {dow_names[dow]}: {weather_name} {result['mean_treatment']:.1f} vs Normal {result['mean_control']:.1f} "
                      f"→ {result['pct_change']:+.1f}% (p={result['mannwhitney_p']:.3f}{sig})")

        # Pooled analysis
        treatment_all = df[df[weather_col] == 1]['occupancy']
        control_all = df[df[weather_col] == 0]['occupancy']
        pooled_result = run_comparison(treatment_all, control_all, weather_name, 'Normal')

        print(f"\n  POOLED: {weather_name} {pooled_result['mean_treatment']:.1f} vs Normal {pooled_result['mean_control']:.1f}")
        print(f"          Diff: {pooled_result['diff']:+.1f} ({pooled_result['pct_change']:+.1f}%)")
        print(f"          p-value: {pooled_result['mannwhitney_p']:.4f}")
        print(f"          Cohen's d: {pooled_result['cohens_d']:.3f} ({pooled_result['effect_size']})")
        print(f"          95% CI: [{pooled_result['ci_lower']:.1f}, {pooled_result['ci_upper']:.1f}]")

        return {
            'by_dow': all_results,
            'pooled': pooled_result
        }

    return None


def analyze_temperature_impact(df):
    """Analyze impact of temperature on occupancy."""
    print(f"\n{'='*60}")
    print("ANALYSIS: Temperature vs Occupancy Correlation")
    print(f"{'='*60}")

    # Daily aggregation
    daily = df.groupby('date').agg({
        'occupancy': 'mean',
        'temp_mean': 'first',
        'temp_max': 'first',
        'temp_min': 'first',
        'dow_x': 'first' if 'dow_x' in df.columns else 'first'
    }).reset_index()

    # Correlation
    corr = daily['occupancy'].corr(daily['temp_mean'])
    print(f"  Correlation (occupancy vs temp_mean): {corr:.3f}")

    # By temperature bins
    daily['temp_bin'] = pd.cut(daily['temp_mean'],
                                bins=[-20, 0, 10, 20, 40],
                                labels=['<0°C', '0-10°C', '10-20°C', '>20°C'])

    print("\n  By Temperature Range:")
    for temp_bin in ['<0°C', '0-10°C', '10-20°C', '>20°C']:
        subset = daily[daily['temp_bin'] == temp_bin]
        if len(subset) > 0:
            print(f"    {temp_bin}: {subset['occupancy'].mean():.1f} avg occupancy ({len(subset)} days)")

    return {
        'correlation': corr,
        'daily': daily
    }


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(df, results, output_dir):
    """Create weather impact visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rain impact box plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rain vs No Rain
    ax1 = axes[0]
    df['rain_label'] = df['is_rainy'].map({0: 'No Rain', 1: 'Rainy'})
    sns.boxplot(data=df, x='rain_label', y='occupancy', ax=ax1, palette=['#3498db', '#e74c3c'])
    ax1.set_title('Occupancy: Rainy vs Non-Rainy Days')
    ax1.set_xlabel('')
    ax1.set_ylabel('Occupancy (passengers)')

    # Cold drop vs Normal
    ax2 = axes[1]
    df['cold_label'] = df['is_cold_drop'].map({0: 'Normal', 1: 'Cold Drop (>5°C)'})
    cold_data = df[df['is_cold_drop'].notna()]
    sns.boxplot(data=cold_data, x='cold_label', y='occupancy', ax=ax2, palette=['#3498db', '#9b59b6'])
    ax2.set_title('Occupancy: Cold Drop Days vs Normal')
    ax2.set_xlabel('')
    ax2.set_ylabel('Occupancy (passengers)')

    plt.tight_layout()
    plt.savefig(output_dir / 'weather_impact_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Temperature vs Occupancy scatter
    daily = df.groupby('date').agg({
        'occupancy': 'mean',
        'temp_mean': 'first',
        'precipitation': 'first'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(daily['temp_mean'], daily['occupancy'],
                         c=daily['precipitation'], cmap='Blues', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Precipitation (mm)')

    # Trend line
    z = np.polyfit(daily['temp_mean'].dropna(), daily['occupancy'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(daily['temp_mean'].min(), daily['temp_mean'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.1f})')

    ax.set_xlabel('Mean Temperature (°C)')
    ax.set_ylabel('Mean Daily Occupancy')
    ax.set_title('Temperature vs Occupancy (color = precipitation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_vs_occupancy.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Weather impact by day of week
    if 'rain' in results:
        rain_results = results['rain']['by_dow']
        if rain_results:
            fig, ax = plt.subplots(figsize=(10, 5))

            dows = [r['dow'] for r in rain_results]
            pct_changes = [r['pct_change'] for r in rain_results]
            colors = ['#e74c3c' if p > 0 else '#3498db' for p in pct_changes]

            bars = ax.bar(dows, pct_changes, color=colors, edgecolor='black')

            for bar, pct, r in zip(bars, pct_changes, rain_results):
                sig = '*' if r['significant'] else ''
                ax.annotate(f'{pct:+.1f}%{sig}',
                           xy=(bar.get_x() + bar.get_width()/2, pct),
                           xytext=(0, 5 if pct > 0 else -15),
                           textcoords='offset points',
                           ha='center', fontsize=9)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Occupancy Change (%)')
            ax.set_title('Rain Impact on Occupancy by Day of Week (* = p<0.05)')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / 'rain_impact_by_dow.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\n  Visualizations saved to {output_dir}/")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results, output_dir):
    """Generate analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 80)
    report.append("WEATHER IMPACT ON RE5 OCCUPANCY - ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("Analysis Period: 2022-08 ~ 2024-06")
    report.append("Comparison Method: Same day-of-week comparison")
    report.append("")

    # Rain impact
    if 'rain' in results:
        rain = results['rain']['pooled']
        report.append("-" * 80)
        report.append("1. RAIN IMPACT (>1mm precipitation)")
        report.append("-" * 80)
        report.append(f"  Rainy days mean occupancy:  {rain['mean_treatment']:.1f}")
        report.append(f"  Normal days mean occupancy: {rain['mean_control']:.1f}")
        report.append(f"  Difference: {rain['diff']:+.1f} ({rain['pct_change']:+.1f}%)")
        report.append(f"  p-value: {rain['mannwhitney_p']:.4f}")
        report.append(f"  Cohen's d: {rain['cohens_d']:.3f} ({rain['effect_size']})")
        report.append(f"  95% CI: [{rain['ci_lower']:.1f}, {rain['ci_upper']:.1f}]")
        report.append(f"  Significant: {'YES' if rain['significant'] else 'NO'}")
        report.append("")

    # Heavy rain impact
    if 'heavy_rain' in results:
        heavy = results['heavy_rain']['pooled']
        report.append("-" * 80)
        report.append("2. HEAVY RAIN IMPACT (>5mm precipitation)")
        report.append("-" * 80)
        report.append(f"  Heavy rain days mean occupancy: {heavy['mean_treatment']:.1f}")
        report.append(f"  Normal days mean occupancy:     {heavy['mean_control']:.1f}")
        report.append(f"  Difference: {heavy['diff']:+.1f} ({heavy['pct_change']:+.1f}%)")
        report.append(f"  p-value: {heavy['mannwhitney_p']:.4f}")
        report.append(f"  Cohen's d: {heavy['cohens_d']:.3f} ({heavy['effect_size']})")
        report.append(f"  Significant: {'YES' if heavy['significant'] else 'NO'}")
        report.append("")

    # Cold drop impact
    if 'cold_drop' in results:
        cold = results['cold_drop']['pooled']
        report.append("-" * 80)
        report.append("3. COLD DROP IMPACT (>5°C temperature drop from previous day)")
        report.append("-" * 80)
        report.append(f"  Cold drop days mean occupancy: {cold['mean_treatment']:.1f}")
        report.append(f"  Normal days mean occupancy:    {cold['mean_control']:.1f}")
        report.append(f"  Difference: {cold['diff']:+.1f} ({cold['pct_change']:+.1f}%)")
        report.append(f"  p-value: {cold['mannwhitney_p']:.4f}")
        report.append(f"  Cohen's d: {cold['cohens_d']:.3f} ({cold['effect_size']})")
        report.append(f"  Significant: {'YES' if cold['significant'] else 'NO'}")
        report.append("")

    # Snow impact
    if 'snow' in results:
        snow = results['snow']['pooled']
        report.append("-" * 80)
        report.append("4. SNOW IMPACT")
        report.append("-" * 80)
        report.append(f"  Snowy days mean occupancy:  {snow['mean_treatment']:.1f}")
        report.append(f"  Normal days mean occupancy: {snow['mean_control']:.1f}")
        report.append(f"  Difference: {snow['diff']:+.1f} ({snow['pct_change']:+.1f}%)")
        report.append(f"  p-value: {snow['mannwhitney_p']:.4f}")
        report.append(f"  Cohen's d: {snow['cohens_d']:.3f} ({snow['effect_size']})")
        report.append(f"  Significant: {'YES' if snow['significant'] else 'NO'}")
        report.append("")

    # Temperature correlation
    if 'temperature' in results:
        report.append("-" * 80)
        report.append("5. TEMPERATURE CORRELATION")
        report.append("-" * 80)
        report.append(f"  Correlation (occupancy vs temp): {results['temperature']['correlation']:.3f}")
        report.append("")

    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)

    significant_effects = []
    for key in ['rain', 'heavy_rain', 'cold_drop', 'snow']:
        if key in results and results[key]['pooled']['significant']:
            r = results[key]['pooled']
            significant_effects.append(f"  - {key}: {r['pct_change']:+.1f}% (p={r['mannwhitney_p']:.4f})")

    if significant_effects:
        report.append("Significant weather effects found:")
        report.extend(significant_effects)
    else:
        report.append("No significant weather effects found at p<0.05 level.")

    report.append("")
    report.append("=" * 80)

    # Save report
    report_text = '\n'.join(report)
    with open(output_dir / 'weather_impact_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)

    return report_text


# =============================================================================
# Main Analysis
# =============================================================================

def run_weather_analysis():
    """Run complete weather impact analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WEATHER IMPACT ANALYSIS ON RE5 OCCUPANCY")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    occ_df = load_occupancy_data()
    weather_df = load_weather_data()

    print(f"  Occupancy records: {len(occ_df):,}")
    print(f"  Weather days: {len(weather_df)}")

    # Merge
    print("\n[2/5] Merging data...")
    df = merge_data(occ_df, weather_df)
    df = df.dropna(subset=['temp_mean'])  # Remove days without weather data
    print(f"  Merged records: {len(df):,}")

    # Analyze weather impacts
    print("\n[3/5] Analyzing weather impacts...")
    results = {}

    # Rain impact
    results['rain'] = analyze_weather_impact(df, 'is_rainy', 'Rainy (>1mm)')

    # Heavy rain impact
    results['heavy_rain'] = analyze_weather_impact(df, 'is_heavy_rain', 'Heavy Rain (>5mm)')

    # Cold drop impact
    results['cold_drop'] = analyze_weather_impact(df, 'is_cold_drop', 'Cold Drop (>5°C)')

    # Snow impact
    results['snow'] = analyze_weather_impact(df, 'is_snowy', 'Snowy')

    # Temperature correlation
    results['temperature'] = analyze_temperature_impact(df)

    # Visualizations
    print("\n[4/5] Creating visualizations...")
    create_visualizations(df, results, OUTPUT_DIR)

    # Generate report
    print("\n[5/5] Generating report...")
    generate_report(results, OUTPUT_DIR)

    # Save results to CSV
    summary_data = []
    for key in ['rain', 'heavy_rain', 'cold_drop', 'snow']:
        if key in results and results[key]:
            r = results[key]['pooled']
            summary_data.append({
                'condition': key,
                'treatment_mean': r['mean_treatment'],
                'control_mean': r['mean_control'],
                'diff': r['diff'],
                'pct_change': r['pct_change'],
                'p_value': r['mannwhitney_p'],
                'cohens_d': r['cohens_d'],
                'effect_size': r['effect_size'],
                'significant': r['significant']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'weather_impact_summary.csv', index=False)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}/")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    run_weather_analysis()
