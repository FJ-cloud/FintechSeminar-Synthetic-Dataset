#!/usr/bin/env python3
"""
Marginal Fidelity Assessment Script

This script implements comprehensive quality assessment for synthetic datasets using three methods:
1. Absolute-gap table (fast descriptive check)
2. Chi-square goodness-of-fit test (formal categorical test)
3. KS test for continuous/binned variables

Author: Credit Scoring Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare, kstest, norm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_target_marginals():
    """Define target marginal distributions for all variables based on Berg et al. (2020)"""
    
    # Target marginals for scorable population
    scorable_marginals = {
        "credit_score_quintile": {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        "device_type": {"Desktop": 0.57, "Tablet": 0.18, "Mobile": 0.11, "Do-not-track": 0.14},
        "os": {"Windows": 0.49, "iOS": 0.16, "Android": 0.11, "Macintosh": 0.08, "Other": 0.01, "Do-not-track": 0.14},
        "email_host": {"Gmx": 0.23, "Web": 0.22, "T-Online": 0.12, "Gmail": 0.11, "Yahoo": 0.05, "Hotmail": 0.04, "Other": 0.24},
        "channel": {"Paid": 0.44, "Direct": 0.18, "Affiliate": 0.10, "Organic": 0.07, "Other": 0.07, "Do-not-track": 0.14},
        "checkout_time": {"Evening": 0.43, "Night": 0.03, "Morning": 0.18, "Afternoon": 0.36},
        "name_in_email": {"No": 0.28, "Yes": 0.72},
        "number_in_email": {"No": 0.84, "Yes": 0.16},
        "is_lowercase": {"No": 0.99, "Yes": 0.01},
        "email_error": {"No": 0.92, "Yes": 0.08},
        "age_quintile": {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        "order_amount_quintile": {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        "gender": {"Female": 0.66, "Male": 0.34},
        "item_category": {f"Cat{i}": 1/16 for i in range(1, 17)},
        "month": {month: 1/15 for month in ["Oct15", "Nov15", "Dec15", "Jan16", "Feb16", "Mar16", "Apr16", 
                                          "May16", "Jun16", "Jul16", "Aug16", "Sep16", "Oct16", "Nov16", "Dec16"]}
    }
    
    # Target marginals for unscorable population
    unscorable_marginals = {
        "credit_score_quintile": {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        "device_type": {"Desktop": 0.59, "Tablet": 0.17, "Mobile": 0.10, "Do-not-track": 0.14},
        "os": {"Windows": 0.50, "iOS": 0.16, "Android": 0.11, "Macintosh": 0.09, "Other": 0.01, "Do-not-track": 0.14},
        "email_host": {"Gmx": 0.24, "Web": 0.21, "T-Online": 0.11, "Gmail": 0.11, "Yahoo": 0.05, "Hotmail": 0.04, "Other": 0.25},
        "channel": {"Paid": 0.41, "Direct": 0.21, "Affiliate": 0.09, "Organic": 0.08, "Other": 0.07, "Do-not-track": 0.14},
        "checkout_time": {"Evening": 0.41, "Night": 0.02, "Morning": 0.19, "Afternoon": 0.38},
        "name_in_email": {"No": 0.28, "Yes": 0.72},
        "number_in_email": {"No": 0.83, "Yes": 0.17},
        "is_lowercase": {"No": 0.93, "Yes": 0.07},
        "email_error": {"No": 0.98, "Yes": 0.02},
        "age_quintile": {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        "order_amount_quintile": {"Q1": 0.20, "Q2": 0.20, "Q3": 0.20, "Q4": 0.20, "Q5": 0.20},
        "gender": {"Female": 0.66, "Male": 0.34},
        "item_category": {f"Cat{i}": 1/16 for i in range(1, 17)},
        "month": {month: 1/15 for month in ["Oct15", "Nov15", "Dec15", "Jan16", "Feb16", "Mar16", "Apr16", 
                                          "May16", "Jun16", "Jul16", "Aug16", "Sep16", "Oct16", "Nov16", "Dec16"]}
    }
    
    return scorable_marginals, unscorable_marginals

def get_continuous_targets():
    """Define target parameters for continuous variables"""
    return {
        "scorable": {
            "age": {"mean": 45.06, "std": 13.31},
            "order_amount": {"mean": 261.27, "std": 179.32}
        },
        "unscorable": {
            "age": {"mean": 38.2, "std": 10.46},
            "order_amount": {"mean": 261.27, "std": 179.32}
        }
    }

def absolute_gap_analysis(synthetic_df, target_marginals, dataset_name):
    """1. Absolute-gap table (fast descriptive check)"""
    print(f"\n{'='*60}")
    print(f"ABSOLUTE GAP ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    rows = []
    total_gap = 0
    total_categories = 0
    
    for var, target_dist in target_marginals.items():
        if var not in synthetic_df.columns:
            continue
            
        counts = synthetic_df[var].value_counts(normalize=True)
        
        for cat, target_p in target_dist.items():
            synth_p = counts.get(cat, 0)
            gap = abs(synth_p - target_p) * 100
            rows.append([var, cat, target_p*100, synth_p*100, gap])
            total_gap += gap
            total_categories += 1
    
    fidelity_df = pd.DataFrame(
        rows, columns=["Variable", "Category", "Target (%)", "Synthetic (%)", "|Gap| (pp)"]
    ).sort_values(["Variable", "Category"])
    
    avg_gap = total_gap / total_categories if total_categories > 0 else 0
    max_gap = fidelity_df["|Gap| (pp)"].max() if len(fidelity_df) > 0 else 0
    
    print(f"Average absolute gap: {avg_gap:.3f} percentage points")
    print(f"Maximum absolute gap: {max_gap:.3f} percentage points")
    print(f"Categories with gap > 1.0 pp: {len(fidelity_df[fidelity_df['|Gap| (pp)'] > 1.0])}")
    
    if len(fidelity_df) > 0:
        print(f"\nWorst offenders (Top 10):")
        worst = fidelity_df.nlargest(10, "|Gap| (pp)")
        print(worst.to_string(index=False))
    
    return fidelity_df, {"avg_gap": avg_gap, "max_gap": max_gap}

def chi_square_analysis(synthetic_df, target_marginals, dataset_name):
    """2. Chi-square goodness-of-fit test"""
    print(f"\n{'='*60}")
    print(f"CHI-SQUARE ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    results = []
    significant_vars = []
    
    for var, target_dist in target_marginals.items():
        if var not in synthetic_df.columns:
            continue
        
        n_samples = len(synthetic_df)
        target_counts = np.array([p * n_samples for p in target_dist.values()])
        observed_counts = synthetic_df[var].value_counts().reindex(
            target_dist.keys(), fill_value=0
        ).values
        
        try:
            chi2, p_val = chisquare(observed_counts, f_exp=target_counts)
            results.append({
                'Variable': var,
                'Chi2_Statistic': chi2,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })
            
            if p_val < 0.05:
                significant_vars.append(var)
            
            print(f"{var:25s}  chi2 = {chi2:8.2f}   p = {p_val:0.3g}   {'*' if p_val < 0.05 else ' '}")
            
        except Exception as e:
            print(f"{var:25s}  ERROR: {str(e)}")
    
    print(f"\nSignificant deviations (p < 0.05): {len(significant_vars)}")
    if significant_vars:
        print(f"Variables requiring investigation: {significant_vars}")
    
    return pd.DataFrame(results)

def ks_test_analysis(synthetic_df, continuous_targets, population_type, dataset_name):
    """3. KS test for continuous variables"""
    print(f"\n{'='*60}")
    print(f"KS TEST ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    if population_type not in continuous_targets:
        print(f"No continuous targets for population: {population_type}")
        return pd.DataFrame()
    
    targets = continuous_targets[population_type]
    results = []
    
    for var, params in targets.items():
        if var not in synthetic_df.columns:
            continue
        
        data = synthetic_df[var].dropna()
        if len(data) == 0:
            continue
        
        mu, sigma = params['mean'], params['std']
        try:
            stat, p_val = kstest(data, cdf=lambda x: norm.cdf(x, loc=mu, scale=sigma))
            
            results.append({
                'Variable': var,
                'KS_Statistic': stat,
                'P_Value': p_val,
                'Significant': p_val < 0.05,
                'Target_Mean': mu,
                'Target_Std': sigma,
                'Actual_Mean': data.mean(),
                'Actual_Std': data.std()
            })
            
            print(f"{var:15s}  KS = {stat:0.4f}   p = {p_val:0.3g}   {'*' if p_val < 0.05 else ' '}")
            print(f"{'':15s}  Target: mu={mu:.2f}, sigma={sigma:.2f}")
            print(f"{'':15s}  Actual: mu={data.mean():.2f}, sigma={data.std():.2f}")
            
        except Exception as e:
            print(f"{var:15s}  ERROR: {str(e)}")
    
    return pd.DataFrame(results)

def comprehensive_assessment(dataset_path, population_type, method_name):
    """Run comprehensive fidelity assessment"""
    print(f"\n{'#'*80}")
    print(f"COMPREHENSIVE FIDELITY ASSESSMENT")
    print(f"Dataset: {dataset_path}")
    print(f"Population: {population_type}, Method: {method_name}")
    print(f"{'#'*80}")
    
    # Load dataset
    try:
        synthetic_df = pd.read_csv(dataset_path)
        print(f"Loaded: {synthetic_df.shape[0]:,} rows, {synthetic_df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Get targets
    scorable_marginals, unscorable_marginals = get_target_marginals()
    target_marginals = scorable_marginals if population_type == 'scorable' else unscorable_marginals
    continuous_targets = get_continuous_targets()
    
    dataset_name = f"{method_name}_{population_type}"
    
    # Run analyses
    gap_df, gap_summary = absolute_gap_analysis(synthetic_df, target_marginals, dataset_name)
    chi2_df = chi_square_analysis(synthetic_df, target_marginals, dataset_name)
    ks_df = ks_test_analysis(synthetic_df, continuous_targets, population_type, dataset_name)
    
    # Quality score
    quality_score = 100
    avg_gap = gap_summary['avg_gap']
    
    if avg_gap > 2.0:
        quality_score -= 30
    elif avg_gap > 1.0:
        quality_score -= 20
    elif avg_gap > 0.5:
        quality_score -= 10
    
    if len(chi2_df) > 0:
        chi2_failures = chi2_df['Significant'].sum()
        failure_rate = chi2_failures / len(chi2_df)
        if failure_rate > 0.5:
            quality_score -= 30
        elif failure_rate > 0.25:
            quality_score -= 20
        elif failure_rate > 0.1:
            quality_score -= 10
    
    quality_score = max(0, quality_score)
    
    print(f"\n{'='*60}")
    print(f"OVERALL ASSESSMENT: {dataset_name}")
    print(f"{'='*60}")
    print(f"Marginal Fidelity Score: {quality_score}/100")
    
    if quality_score >= 90:
        print("Assessment: EXCELLENT")
    elif quality_score >= 75:
        print("Assessment: GOOD")
    elif quality_score >= 60:
        print("Assessment: FAIR")
    else:
        print("Assessment: POOR")
    
    return {
        'absolute_gap': gap_df,
        'chi_square': chi2_df,
        'ks_test': ks_df,
        'quality_score': quality_score,
        'avg_gap': avg_gap
    }

def main():
    """Main function to run assessments"""
    data_dir = Path("FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/data")
    
    # Define datasets
    datasets = [
        ("synthetic_digital_footprint_copula.csv", "scorable", "copula"),
        ("synthetic_digital_footprint_copula_unscorable.csv", "unscorable", "copula"),
        ("synthetic_digital_footprint_ctgan.csv", "scorable", "ctgan"),
        ("synthetic_digital_footprint_ctgan_unscorable.csv", "unscorable", "ctgan"),
    ]
    
    all_results = {}
    summary_rows = []
    
    for filename, population_type, method_name in datasets:
        dataset_path = data_dir / filename
        
        if dataset_path.exists():
            try:
                results = comprehensive_assessment(dataset_path, population_type, method_name)
                if results:
                    key = f"{method_name}_{population_type}"
                    all_results[key] = results
                    summary_rows.append([
                        key,
                        f"{results['quality_score']}/100",
                        f"{results['avg_gap']:.3f} pp"
                    ])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Dataset not found: {dataset_path}")
    
    # Summary
    if summary_rows:
        print(f"\n{'#'*80}")
        print("SUMMARY REPORT")
        print(f"{'#'*80}")
        
        summary_df = pd.DataFrame(
            summary_rows, 
            columns=["Dataset", "Quality Score", "Avg Gap"]
        )
        print(summary_df.to_string(index=False))
        
        # Save results
        summary_df.to_csv("../results/marginal_fidelity_summary.csv", index=False)
        print(f"\nResults saved to: ../results/marginal_fidelity_summary.csv")

if __name__ == "__main__":
    main() 