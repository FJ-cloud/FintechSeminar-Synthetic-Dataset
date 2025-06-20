#!/usr/bin/env python3
"""
Simplified Marginal Fidelity Assessment Script

This script provides a quick and direct comparison between seed datasets and synthetic datasets
using mean/std gaps and KS statistics for easy evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def quick_marginal_report(seed_df, synth_df, label, cols):
    """Print mean & std gaps + median KS statistic for a synthetic dataset."""
    gaps = []
    for col in cols:
        if col not in seed_df.columns or col not in synth_df.columns:
            continue
            
        seed_mean, seed_std = seed_df[col].mean(), seed_df[col].std()
        synth_mean, synth_std = synth_df[col].mean(), synth_df[col].std()
        mean_gap = abs(synth_mean - seed_mean)
        std_gap = abs(synth_std - seed_std)
        ks_stat = ks_2samp(seed_df[col], synth_df[col]).statistic
        gaps.append((col, mean_gap, std_gap, ks_stat))

    if not gaps:
        print(f"\n=== {label} ===")
        print("No matching columns found for analysis.")
        return
        
    rpt = (pd.DataFrame(gaps, columns=["feature", "mean_gap", "std_gap", "ks_stat"])
             .set_index("feature")
             .sort_values("ks_stat"))
    
    print(f"\n=== {label} ===")
    print(rpt.head(10))  # show 10 best-matching variables
    print(f"Median KS statistic: {rpt['ks_stat'].median():.4f}")
    print(f"Mean absolute gap: {rpt['mean_gap'].mean():.4f}")
    print(f"Std absolute gap: {rpt['std_gap'].mean():.4f}")

def load_datasets():
    """Load all available datasets for comparison"""
    data_dir = Path("../data")
    datasets = {}
    
    # Training datasets (seed data)
    files = {
        'seed_scorable': 'synthetic_digital_footprint_with_target.csv',
        'seed_unscorable': 'synthetic_digital_footprint_with_target_unscorable.csv',
        'copula_scorable': 'synthetic_digital_footprint_copula.csv',
        'copula_unscorable': 'synthetic_digital_footprint_copula_unscorable.csv',
        'ctgan_scorable': 'synthetic_digital_footprint_ctgan.csv',
        'ctgan_unscorable': 'synthetic_digital_footprint_ctgan_unscorable.csv'
    }
    
    for key, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                datasets[key] = pd.read_csv(filepath)
                print(f"Loaded {key}: {datasets[key].shape}")
            except Exception as e:
                print(f"Error loading {key}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    return datasets

def main():
    """Main analysis function"""
    print("="*80)
    print("SIMPLIFIED MARGINAL FIDELITY ASSESSMENT")
    print("="*80)
    
    # Load datasets
    datasets = load_datasets()
    
    if not datasets:
        print("No datasets found. Please ensure the data files exist in ../data/")
        return
    
    # Define continuous columns to analyze
    continuous_cols = ['age', 'order_amount']
    
    # Get all available columns from seed data
    if 'seed_scorable' in datasets:
        all_cols = [col for col in datasets['seed_scorable'].columns 
                   if datasets['seed_scorable'][col].dtype in ['int64', 'float64']]
        print(f"\nAnalyzing {len(all_cols)} numeric columns: {all_cols}")
    else:
        all_cols = continuous_cols
        print(f"\nAnalyzing default columns: {all_cols}")
    
    # SCORABLE POPULATION ANALYSIS
    if 'seed_scorable' in datasets:
        print(f"\n{'='*60}")
        print("SCORABLE POPULATION ANALYSIS")
        print(f"{'='*60}")
        
        seed_df = datasets['seed_scorable']
        
        # Compare Copula vs Seed
        if 'copula_scorable' in datasets:
            quick_marginal_report(seed_df, datasets['copula_scorable'], 
                                "Copula vs Seed (Scorable)", all_cols)
        
        # Compare CTGAN vs Seed
        if 'ctgan_scorable' in datasets:
            quick_marginal_report(seed_df, datasets['ctgan_scorable'], 
                                "CTGAN vs Seed (Scorable)", all_cols)
    
    # UNSCORABLE POPULATION ANALYSIS
    if 'seed_unscorable' in datasets:
        print(f"\n{'='*60}")
        print("UNSCORABLE POPULATION ANALYSIS")
        print(f"{'='*60}")
        
        seed_df = datasets['seed_unscorable']
        
        # Compare Copula vs Seed
        if 'copula_unscorable' in datasets:
            quick_marginal_report(seed_df, datasets['copula_unscorable'], 
                                "Copula vs Seed (Unscorable)", all_cols)
        
        # Compare CTGAN vs Seed
        if 'ctgan_unscorable' in datasets:
            quick_marginal_report(seed_df, datasets['ctgan_unscorable'], 
                                "CTGAN vs Seed (Unscorable)", all_cols)
    
    # CROSS-METHOD COMPARISON
    if 'copula_scorable' in datasets and 'ctgan_scorable' in datasets:
        print(f"\n{'='*60}")
        print("CROSS-METHOD COMPARISON (SCORABLE)")
        print(f"{'='*60}")
        quick_marginal_report(datasets['copula_scorable'], datasets['ctgan_scorable'], 
                            "Copula vs CTGAN (Scorable)", all_cols)
    
    if 'copula_unscorable' in datasets and 'ctgan_unscorable' in datasets:
        print(f"\n{'='*60}")
        print("CROSS-METHOD COMPARISON (UNSCORABLE)")
        print(f"{'='*60}")
        quick_marginal_report(datasets['copula_unscorable'], datasets['ctgan_unscorable'], 
                            "Copula vs CTGAN (Unscorable)", all_cols)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("INTERPRETATION:")
    print("- Lower mean_gap & std_gap = better match in central tendency & spread")
    print("- Lower KS statistic = better match in full distribution shape")
    print("- Expected: Copula ≪ CTGAN (Copula should match seed much better)")
    print("="*80)

if __name__ == "__main__":
    main() 