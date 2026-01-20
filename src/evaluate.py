import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


######################################
# Evaluation & Visualization for RQ1 #
######################################

def load_results(results_path='outputs/rq1_results/training_results.csv'):
    """Load training results."""
    return pd.read_csv(results_path)


def create_comparison_table(results_df):
    """
    Create RQ1 comparison table.
    
    Shows MAE, RMSE, R² for each horizon and feature set.
    """
    # Pivot table
    pivot = results_df.pivot_table(
        index='horizon',
        columns='feature_set',
        values=['MAE', 'RMSE', 'R²']
    )
    
    # Calculate benefit (when extended features available)
    if 'extended' in results_df['feature_set'].values:
        benefit = results_df.pivot_table(
            index='horizon',
            columns='feature_set',
            values='MAE'
        )
        
        if 'base' in benefit.columns and 'extended' in benefit.columns:
            benefit['Δ MAE'] = benefit['base'] - benefit['extended']
            benefit['% improvement'] = (benefit['Δ MAE'] / benefit['base'] * 100).round(2)
            
    return pivot


def plot_horizon_comparison(results_df, output_dir='outputs/rq1_results'):
    """
    Create visualization comparing horizons.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    
    # Plot MAE across horizons
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    horizons_order = ['30min', '1h', '3h', 'next_day']
    
    for idx, metric in enumerate(['MAE', 'RMSE', 'R²']):
        ax = axes[idx]
        
        for feature_set in results_df['feature_set'].unique():
            subset = results_df[results_df['feature_set'] == feature_set]
            subset = subset.set_index('horizon').reindex(horizons_order)
            
            ax.plot(subset.index, subset[metric], marker='o', label=feature_set, linewidth=2)
        
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'horizon_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_rq1_summary(results_df, output_path='outputs/rq1_results/rq1_summary.csv'):
    """
    Create final RQ1 summary table for thesis/report.
    """
    # Reshape for readability
    summary = results_df.copy()
    summary = summary.round(3)
    
    # Add interpretation column
    def interpret_r2(r2):
        if r2 > 0.7:
            return 'Good'
        elif r2 > 0.5:
            return 'Moderate'
        else:
            return 'Weak'
    
    summary['Performance'] = summary['R²'].apply(interpret_r2)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def main():
    """Main execution."""
    
    # Load results
    results_df = load_results()
    
    # Create comparison table
    pivot = create_comparison_table(results_df)
    
    # Create plots
    plot_horizon_comparison(results_df)
    
    # Create summary
    summary = create_rq1_summary(results_df)
    
    print("RQ1 EVALUATION COMPLETED")

if __name__ == '__main__':
    main()
