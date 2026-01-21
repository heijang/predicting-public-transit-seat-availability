"""
Match Day Effect Statistical Analysis
- t-test / Mann-Whitney U test
- OLS Regression (controlling for other variables)
- Effect Size (Cohen's d)
- Visualization (Box plot, Violin plot)
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

def load_and_filter_data():
    """Load and filter data with specified conditions"""
    df = pd.read_csv('data/processed/re5_with_targets.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Filter: August 2022 onwards
    df = df[df['date'] >= '2022-08-01']

    # Load holiday info
    import holidays
    de_holidays = holidays.country_holidays('DE', subdiv='BB', years=range(2022, 2025))
    df['is_holiday'] = df['date'].dt.date.apply(lambda x: x in de_holidays).astype(int)

    # Exclude holidays
    df = df[df['is_holiday'] == 0]

    # Add month column if not exists
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month

    # Add historical average columns if not exist
    if 'hist_avg_stop_dow' not in df.columns:
        df['hist_avg_stop_dow'] = df.groupby(['dep_id', 'dow'])['occupancy'].transform('mean')
    if 'hist_avg_stop_hour' not in df.columns:
        df['hist_avg_stop_hour'] = df.groupby(['dep_id', 'hour'])['occupancy'].transform('mean')

    print(f"Total records after filtering: {len(df):,}")
    print(f"Match day records: {df['is_match_day'].sum():,}")
    print(f"Non-match day records: {(df['is_match_day'] == 0).sum():,}")

    return df


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_cohens_d(d):
    """Interpret Cohen's d value"""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def statistical_tests(df):
    """Perform statistical tests"""
    match_day = df[df['is_match_day'] == 1]['occupancy']
    non_match_day = df[df['is_match_day'] == 0]['occupancy']

    results = {}

    # Basic statistics
    results['match_day_mean'] = match_day.mean()
    results['match_day_std'] = match_day.std()
    results['match_day_n'] = len(match_day)
    results['non_match_day_mean'] = non_match_day.mean()
    results['non_match_day_std'] = non_match_day.std()
    results['non_match_day_n'] = len(non_match_day)

    # Difference
    results['mean_diff'] = results['match_day_mean'] - results['non_match_day_mean']
    results['pct_diff'] = (results['mean_diff'] / results['non_match_day_mean']) * 100

    # Independent samples t-test
    t_stat, t_pvalue = stats.ttest_ind(match_day, non_match_day)
    results['t_statistic'] = t_stat
    results['t_pvalue'] = t_pvalue

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(match_day, non_match_day, alternative='two-sided')
    results['u_statistic'] = u_stat
    results['u_pvalue'] = u_pvalue

    # Cohen's d
    results['cohens_d'] = cohens_d(match_day, non_match_day)
    results['cohens_d_interpretation'] = interpret_cohens_d(results['cohens_d'])

    # 95% Confidence Interval for mean difference (using bootstrap)
    np.random.seed(42)
    n_bootstrap = 10000
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_match = np.random.choice(match_day, size=len(match_day), replace=True)
        boot_non_match = np.random.choice(non_match_day, size=len(non_match_day), replace=True)
        boot_diffs.append(boot_match.mean() - boot_non_match.mean())

    results['ci_lower'] = np.percentile(boot_diffs, 2.5)
    results['ci_upper'] = np.percentile(boot_diffs, 97.5)

    return results


def ols_regression(df):
    """OLS Regression controlling for other variables"""
    # Prepare data
    df_reg = df.copy()

    # Model 1: Simple regression (match day only)
    model1 = smf.ols('occupancy ~ is_match_day', data=df_reg).fit()

    # Model 2: Controlled regression (with time/day variables)
    model2 = smf.ols('''occupancy ~ is_match_day + hour + C(dow) + C(month) +
                        is_weekend + is_rush_hour + is_night''', data=df_reg).fit()

    # Model 3: Full model with historical averages
    model3 = smf.ols('''occupancy ~ is_match_day + hour + C(dow) + C(month) +
                        is_weekend + is_rush_hour + is_night +
                        hist_avg_stop_dow + hist_avg_stop_hour''', data=df_reg).fit()

    return model1, model2, model3


def time_specific_analysis(df):
    """Analyze match day effect by time period"""
    results = []

    # Define time periods
    periods = [
        ('Pre-match (2h before)', lambda x: True),  # Will be filtered by match time
        ('Rush Hour (7-9, 17-19)', df['is_rush_hour'] == 1),
        ('Daytime (10-16)', (df['hour'] >= 10) & (df['hour'] <= 16)),
        ('Evening (17-22)', (df['hour'] >= 17) & (df['hour'] <= 22)),
        ('Weekend', df['is_weekend'] == 1),
        ('Weekday', df['is_weekend'] == 0),
    ]

    for period_name, condition in periods[1:]:  # Skip first one (special handling)
        subset = df[condition]
        match = subset[subset['is_match_day'] == 1]['occupancy']
        non_match = subset[subset['is_match_day'] == 0]['occupancy']

        if len(match) > 30 and len(non_match) > 30:
            t_stat, p_val = stats.ttest_ind(match, non_match)
            d = cohens_d(match, non_match)

            results.append({
                'Period': period_name,
                'Match Day Mean': match.mean(),
                'Non-Match Mean': non_match.mean(),
                'Difference': match.mean() - non_match.mean(),
                'Pct Change': ((match.mean() - non_match.mean()) / non_match.mean()) * 100,
                't-statistic': t_stat,
                'p-value': p_val,
                "Cohen's d": d,
                'Effect Size': interpret_cohens_d(d),
                'N (Match)': len(match),
                'N (Non-Match)': len(non_match)
            })

    return pd.DataFrame(results)


def create_visualizations(df, output_dir):
    """Create visualizations"""

    # 1. Box Plot - Overall comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    df['Match Day'] = df['is_match_day'].map({0: 'Non-Match Day', 1: 'Match Day'})
    sns.boxplot(data=df, x='Match Day', y='occupancy', ax=axes[0], palette=['#3498db', '#e74c3c'])
    axes[0].set_title('Occupancy Distribution: Match Day vs Non-Match Day', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Occupancy (passengers)')

    # Add mean markers
    means = df.groupby('Match Day')['occupancy'].mean()
    for i, (label, mean) in enumerate(means.items()):
        axes[0].scatter(i, mean, color='white', s=100, zorder=5, edgecolor='black', linewidth=2)
        axes[0].annotate(f'Mean: {mean:.1f}', (i, mean), textcoords="offset points",
                        xytext=(0, 15), ha='center', fontsize=10, fontweight='bold')

    # Violin plot
    sns.violinplot(data=df, x='Match Day', y='occupancy', ax=axes[1], palette=['#3498db', '#e74c3c'])
    axes[1].set_title('Occupancy Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Occupancy (passengers)')

    plt.tight_layout()
    plt.savefig(output_dir / 'match_day_boxplot_violin.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Hourly comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    hourly = df.groupby(['hour', 'is_match_day'])['occupancy'].agg(['mean', 'std', 'count']).reset_index()
    hourly_match = hourly[hourly['is_match_day'] == 1]
    hourly_non = hourly[hourly['is_match_day'] == 0]

    # Calculate 95% CI
    hourly_match['ci'] = 1.96 * hourly_match['std'] / np.sqrt(hourly_match['count'])
    hourly_non['ci'] = 1.96 * hourly_non['std'] / np.sqrt(hourly_non['count'])

    ax.errorbar(hourly_non['hour'], hourly_non['mean'], yerr=hourly_non['ci'],
                label='Non-Match Day', marker='o', capsize=3, color='#3498db', linewidth=2)
    ax.errorbar(hourly_match['hour'], hourly_match['mean'], yerr=hourly_match['ci'],
                label='Match Day', marker='s', capsize=3, color='#e74c3c', linewidth=2)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Mean Occupancy (passengers)', fontsize=12)
    ax.set_title('Hourly Occupancy: Match Day vs Non-Match Day (with 95% CI)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'match_day_hourly_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Effect size by day of week
    fig, ax = plt.subplots(figsize=(12, 6))

    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_effects = []

    for dow in range(7):
        subset = df[df['dow'] == dow]
        match = subset[subset['is_match_day'] == 1]['occupancy']
        non_match = subset[subset['is_match_day'] == 0]['occupancy']

        if len(match) > 10:
            pct_diff = ((match.mean() - non_match.mean()) / non_match.mean()) * 100
            dow_effects.append({'dow': dow_names[dow], 'pct_diff': pct_diff, 'n_match': len(match)})

    if dow_effects:
        dow_df = pd.DataFrame(dow_effects)
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in dow_df['pct_diff']]
        bars = ax.bar(dow_df['dow'], dow_df['pct_diff'], color=colors, edgecolor='black')

        # Add value labels
        for bar, n in zip(bars, dow_df['n_match']):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%\n(n={n})',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=10)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Occupancy Change (%)', fontsize=12)
        ax.set_title('Match Day Effect by Day of Week', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'match_day_effect_by_dow.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Regression coefficient plot
    return True


def generate_report(test_results, models, time_analysis, output_dir):
    """Generate comprehensive report"""

    report = []
    report.append("=" * 80)
    report.append("MATCH DAY EFFECT: STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # 1. Overview
    report.append("1. DATA OVERVIEW")
    report.append("-" * 80)
    report.append(f"  Match Day samples:     {test_results['match_day_n']:,}")
    report.append(f"  Non-Match Day samples: {test_results['non_match_day_n']:,}")
    report.append(f"  Filters applied:")
    report.append(f"    - Period: August 2022 onwards (exclude COVID/off-season)")
    report.append(f"    - Team: Hertha BSC only (Union Berlin excluded)")
    report.append(f"    - Holidays: Excluded")
    report.append("")

    # 2. Descriptive Statistics
    report.append("2. DESCRIPTIVE STATISTICS")
    report.append("-" * 80)
    report.append("")
    report.append("  ┌─────────────────────────────────────────────────────────────┐")
    report.append("  │                    │  Match Day    │  Non-Match Day         │")
    report.append("  ├─────────────────────────────────────────────────────────────┤")
    report.append(f"  │  Mean Occupancy   │  {test_results['match_day_mean']:>10.2f}  │  {test_results['non_match_day_mean']:>10.2f}           │")
    report.append(f"  │  Std Deviation    │  {test_results['match_day_std']:>10.2f}  │  {test_results['non_match_day_std']:>10.2f}           │")
    report.append(f"  │  Sample Size      │  {test_results['match_day_n']:>10,}  │  {test_results['non_match_day_n']:>10,}           │")
    report.append("  └─────────────────────────────────────────────────────────────┘")
    report.append("")
    report.append(f"  Mean Difference: {test_results['mean_diff']:+.2f} passengers ({test_results['pct_diff']:+.2f}%)")
    report.append(f"  95% CI: [{test_results['ci_lower']:.2f}, {test_results['ci_upper']:.2f}]")
    report.append("")

    # 3. Statistical Tests
    report.append("3. STATISTICAL SIGNIFICANCE TESTS")
    report.append("-" * 80)
    report.append("")
    report.append("  3.1 Independent Samples t-test")
    report.append(f"      t-statistic: {test_results['t_statistic']:.4f}")
    report.append(f"      p-value:     {test_results['t_pvalue']:.2e}")
    sig_t = "YES (p < 0.05)" if test_results['t_pvalue'] < 0.05 else "NO (p >= 0.05)"
    report.append(f"      Significant: {sig_t}")
    report.append("")
    report.append("  3.2 Mann-Whitney U test (non-parametric)")
    report.append(f"      U-statistic: {test_results['u_statistic']:.4f}")
    report.append(f"      p-value:     {test_results['u_pvalue']:.2e}")
    sig_u = "YES (p < 0.05)" if test_results['u_pvalue'] < 0.05 else "NO (p >= 0.05)"
    report.append(f"      Significant: {sig_u}")
    report.append("")

    # 4. Effect Size
    report.append("4. EFFECT SIZE")
    report.append("-" * 80)
    report.append("")
    report.append(f"  Cohen's d: {test_results['cohens_d']:.4f}")
    report.append(f"  Interpretation: {test_results['cohens_d_interpretation']}")
    report.append("")
    report.append("  Reference:")
    report.append("    |d| < 0.2  : Negligible")
    report.append("    |d| 0.2-0.5: Small")
    report.append("    |d| 0.5-0.8: Medium")
    report.append("    |d| > 0.8  : Large")
    report.append("")

    # 5. OLS Regression
    report.append("5. OLS REGRESSION ANALYSIS")
    report.append("-" * 80)
    report.append("")
    model1, model2, model3 = models

    report.append("  5.1 Simple Regression (Match Day only)")
    report.append(f"      Coefficient: {model1.params['is_match_day']:.4f}")
    report.append(f"      Std Error:   {model1.bse['is_match_day']:.4f}")
    report.append(f"      t-value:     {model1.tvalues['is_match_day']:.4f}")
    report.append(f"      p-value:     {model1.pvalues['is_match_day']:.2e}")
    report.append(f"      R-squared:   {model1.rsquared:.4f}")
    report.append("")
    report.append(f"      Interpretation: On match days, occupancy increases by")
    report.append(f"                      {model1.params['is_match_day']:.2f} passengers on average")
    report.append("")

    report.append("  5.2 Controlled Regression (with time/day variables)")
    report.append(f"      Coefficient: {model2.params['is_match_day']:.4f}")
    report.append(f"      Std Error:   {model2.bse['is_match_day']:.4f}")
    report.append(f"      t-value:     {model2.tvalues['is_match_day']:.4f}")
    report.append(f"      p-value:     {model2.pvalues['is_match_day']:.2e}")
    report.append(f"      R-squared:   {model2.rsquared:.4f}")
    report.append("")
    report.append(f"      Interpretation: Controlling for hour, day of week, month,")
    report.append(f"                      weekend, rush hour, and night, match days")
    report.append(f"                      increase occupancy by {model2.params['is_match_day']:.2f} passengers")
    report.append("")

    report.append("  5.3 Full Model (with historical averages)")
    report.append(f"      Coefficient: {model3.params['is_match_day']:.4f}")
    report.append(f"      Std Error:   {model3.bse['is_match_day']:.4f}")
    report.append(f"      t-value:     {model3.tvalues['is_match_day']:.4f}")
    report.append(f"      p-value:     {model3.pvalues['is_match_day']:.2e}")
    report.append(f"      R-squared:   {model3.rsquared:.4f}")
    report.append("")
    report.append(f"      Interpretation: Even after controlling for historical patterns,")
    report.append(f"                      match days increase occupancy by {model3.params['is_match_day']:.2f} passengers")
    report.append("")

    # 6. Time-specific analysis
    report.append("6. TIME-SPECIFIC ANALYSIS")
    report.append("-" * 80)
    report.append("")

    if len(time_analysis) > 0:
        report.append("  ┌────────────────────────────────────────────────────────────────────────────────┐")
        report.append("  │  Period              │ Match │ Non-M │ Diff  │  %Chg │ p-value │ Effect Size │")
        report.append("  ├────────────────────────────────────────────────────────────────────────────────┤")
        for _, row in time_analysis.iterrows():
            sig = "*" if row['p-value'] < 0.05 else " "
            report.append(f"  │  {row['Period']:<20}│{row['Match Day Mean']:>6.1f} │{row['Non-Match Mean']:>6.1f} │{row['Difference']:>+6.1f} │{row['Pct Change']:>+6.1f}%│ {row['p-value']:.2e}{sig}│ {row['Effect Size']:<11} │")
        report.append("  └────────────────────────────────────────────────────────────────────────────────┘")
        report.append("  * p < 0.05")
    report.append("")

    # 7. Conclusion
    report.append("7. CONCLUSION")
    report.append("-" * 80)
    report.append("")

    if test_results['t_pvalue'] < 0.05:
        report.append("  Statistical Significance: YES")
        report.append(f"    - The difference in occupancy between match days and non-match days")
        report.append(f"      is statistically significant (p < 0.05)")
        report.append("")

    report.append(f"  Effect Size: {test_results['cohens_d_interpretation']} (Cohen's d = {test_results['cohens_d']:.3f})")
    report.append("")
    report.append(f"  Practical Impact:")
    report.append(f"    - Match days increase occupancy by {test_results['mean_diff']:+.1f} passengers")
    report.append(f"    - This represents a {test_results['pct_diff']:+.1f}% change from non-match days")
    report.append(f"    - 95% CI: [{test_results['ci_lower']:.1f}, {test_results['ci_upper']:.1f}] passengers")
    report.append("")
    report.append(f"  Regression Analysis:")
    report.append(f"    - Simple model: +{model1.params['is_match_day']:.1f} passengers")
    report.append(f"    - Controlled model: +{model2.params['is_match_day']:.1f} passengers")
    report.append(f"    - Full model: +{model3.params['is_match_day']:.1f} passengers")
    report.append("")

    report.append("=" * 80)
    report.append("Generated by: Match Day Statistical Analysis")
    report.append("=" * 80)

    # Save report
    with open(output_dir / 'match_day_statistical_report.txt', 'w') as f:
        f.write('\n'.join(report))

    return '\n'.join(report)


def main():
    # Setup
    output_dir = Path('outputs/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and filtering data...")
    df = load_and_filter_data()

    print("\nPerforming statistical tests...")
    test_results = statistical_tests(df)

    print("\nRunning OLS regression...")
    models = ols_regression(df)

    print("\nAnalyzing time-specific effects...")
    time_analysis = time_specific_analysis(df)

    print("\nCreating visualizations...")
    create_visualizations(df, output_dir)

    print("\nGenerating report...")
    report = generate_report(test_results, models, time_analysis, output_dir)

    # Save time analysis
    time_analysis.to_csv(output_dir / 'match_day_time_analysis.csv', index=False)

    # Save regression summaries
    model1, model2, model3 = models
    with open(output_dir / 'match_day_regression_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL 1: Simple Regression\n")
        f.write("=" * 80 + "\n")
        f.write(model1.summary().as_text())
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("MODEL 2: Controlled Regression\n")
        f.write("=" * 80 + "\n")
        f.write(model2.summary().as_text())
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("MODEL 3: Full Model\n")
        f.write("=" * 80 + "\n")
        f.write(model3.summary().as_text())

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - match_day_statistical_report.txt")
    print("  - match_day_time_analysis.csv")
    print("  - match_day_regression_summary.txt")
    print("  - match_day_boxplot_violin.png")
    print("  - match_day_hourly_comparison.png")
    print("  - match_day_effect_by_dow.png")

    # Print key results
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print(f"\nMean Difference: {test_results['mean_diff']:+.2f} passengers ({test_results['pct_diff']:+.2f}%)")
    print(f"95% CI: [{test_results['ci_lower']:.2f}, {test_results['ci_upper']:.2f}]")
    print(f"t-test p-value: {test_results['t_pvalue']:.2e}")
    print(f"Cohen's d: {test_results['cohens_d']:.4f} ({test_results['cohens_d_interpretation']})")
    print(f"\nControlled Regression Coefficient: {models[1].params['is_match_day']:.2f} passengers")


if __name__ == "__main__":
    main()
