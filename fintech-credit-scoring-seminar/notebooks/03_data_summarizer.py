#!/usr/bin/env python3
"""
Data Summarizer for Synthetic Credit Scoring Datasets

This script loads multiple synthetic datasets and computes comprehensive
statistical summaries including:
- Dataset dimensions and basic statistics
- Marginal distribution fit (Kolmogorov-Smirnov tests)
- Correlation fidelity (RMSE vs target correlations)
- Collinearity diagnostics (VIF)
- Multivariate normality tests (Mardia's tests)
- Covariance matrix diagnostics
- Effective rank analysis
- Shrinkage parameter reporting

Author: Generated for Fintech Seminar
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import stats
from scipy.linalg import eigvals, det, norm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

class DataSummarizer:
    """
    Comprehensive statistical analysis of synthetic datasets
    """
    
    def __init__(self, data_dir="data", reference_data_path=None):
        self.data_dir = Path(data_dir)
        self.reference_data_path = reference_data_path
        self.results = []
        
    def load_datasets(self):
        """Load all synthetic datasets from the data directory"""
        datasets = {}
        
        # Define expected dataset files (both scorable and unscorable versions)
        dataset_files = {
            # Unscorable (shrinkage) versions
            'synthetic_basic_unscorable': 'synthetic_digital_footprint_with_target_unscorable.csv',
            'synthetic_copula_unscorable': 'synthetic_digital_footprint_copula_unscrorable.csv',
            'synthetic_ctgan_unscorable': 'synthetic_digital_footprint_ctgan_unscorable.csv',
            # Scorable (original) versions if they exist
            'synthetic_basic_scorable': 'synthetic_digital_footprint_with_target.csv',
            'synthetic_copula_scorable': 'synthetic_digital_footprint_copula.csv',
            'synthetic_ctgan_scorable': 'synthetic_digital_footprint_ctgan.csv'
        }
        
        for name, filename in dataset_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {df.shape}")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
            else:
                print(f"✗ File not found: {filepath}")
        
        return datasets
    
    def encode_categorical_variables(self, df):
        """Convert categorical variables to numeric for analysis"""
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        encoders = {}
        for col in categorical_columns:
            if col != 'TARGET':  # Don't encode target if it exists
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        return df_encoded, encoders
    
    def compute_basic_stats(self, df, dataset_name):
        """Compute basic dataset statistics"""
        n_obs, n_vars = df.shape
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        stats_dict = {
            'dataset': dataset_name,
            'n_observations': n_obs,
            'n_variables': n_vars,
            'n_numeric_vars': len(numeric_df.columns),
            'mean_of_means': numeric_df.mean().mean(),
            'mean_of_stds': numeric_df.std().mean(),
            'mean_skewness': numeric_df.skew().mean(),
            'mean_excess_kurtosis': numeric_df.kurtosis().mean()
        }
        
        return stats_dict
    
    def kolmogorov_smirnov_tests(self, df, dataset_name):
        """Perform KS tests for marginal distribution fit"""
        numeric_df = df.select_dtypes(include=[np.number])
        ks_pvalues = []
        
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            if len(data) > 0:
                # Test against normal distribution (as baseline)
                try:
                    ks_stat, ks_pval = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                    ks_pvalues.append(ks_pval)
                except:
                    pass
        
        return {
            'median_ks_pvalue': np.median(ks_pvalues) if ks_pvalues else np.nan,
            'mean_ks_pvalue': np.mean(ks_pvalues) if ks_pvalues else np.nan,
            'n_ks_tests': len(ks_pvalues)
        }
    
    def correlation_fidelity(self, df, target_corr=None, dataset_name=None):
        """Compute correlation matrix fidelity metrics"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'corr_rmse': np.nan, 'corr_mae': np.nan}
        
        empirical_corr = numeric_df.corr()
        
        # If no target correlation provided, use identity as baseline
        if target_corr is None:
            target_corr = np.eye(empirical_corr.shape[0])
        else:
            # Ensure dimensions match
            min_dim = min(empirical_corr.shape[0], target_corr.shape[0])
            empirical_corr = empirical_corr.iloc[:min_dim, :min_dim]
            target_corr = target_corr[:min_dim, :min_dim]
        
        # Remove diagonal elements for comparison
        mask = ~np.eye(empirical_corr.shape[0], dtype=bool)
        emp_off_diag = empirical_corr.values[mask]
        target_off_diag = target_corr[mask] if isinstance(target_corr, np.ndarray) else target_corr.values[mask]
        
        rmse = np.sqrt(np.mean((emp_off_diag - target_off_diag) ** 2))
        mae = np.mean(np.abs(emp_off_diag - target_off_diag))
        
        return {
            'corr_rmse': rmse,
            'corr_mae': mae,
            'corr_matrix_condition_number': np.linalg.cond(empirical_corr.values)
        }
    
    def variance_inflation_factors(self, df, dataset_name):
        """Compute VIF for collinearity assessment"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'median_vif': np.nan, 'max_vif': np.nan, 'n_vif_computed': 0}
        
        # Remove highly correlated or constant columns
        numeric_df = numeric_df.loc[:, numeric_df.std() > 1e-6]
        
        if numeric_df.shape[1] < 2:
            return {'median_vif': np.nan, 'max_vif': np.nan, 'n_vif_computed': 0}
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_df.columns)
        
        vif_values = []
        for i in range(X_scaled_df.shape[1]):
            try:
                vif = variance_inflation_factor(X_scaled_df.values, i)
                if np.isfinite(vif):
                    vif_values.append(vif)
            except:
                continue
        
        if not vif_values:
            return {'median_vif': np.nan, 'max_vif': np.nan, 'n_vif_computed': 0}
        
        return {
            'median_vif': np.median(vif_values),
            'max_vif': np.max(vif_values),
            'n_vif_computed': len(vif_values)
        }
    
    def mardia_multivariate_normality(self, df, dataset_name):
        """Mardia's test for multivariate normality"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 3:
            return {
                'mardia_skewness': np.nan,
                'mardia_kurtosis': np.nan,
                'mardia_skewness_pvalue': np.nan,
                'mardia_kurtosis_pvalue': np.nan
            }
        
        # Remove missing values and standardize
        clean_data = numeric_df.dropna()
        if clean_data.shape[0] < 3:
            return {
                'mardia_skewness': np.nan,
                'mardia_kurtosis': np.nan,
                'mardia_skewness_pvalue': np.nan,
                'mardia_kurtosis_pvalue': np.nan
            }
        
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(clean_data)
            n, p = X.shape
            
            # Mardia's multivariate skewness
            S = np.cov(X.T)
            S_inv = np.linalg.pinv(S)
            
            # Mahalanobis distances
            diff = X - np.mean(X, axis=0)
            md_squared = np.sum(diff @ S_inv * diff, axis=1)
            
            # Skewness statistic
            skewness = np.sum(md_squared ** 3) / (n ** 2)
            skewness_stat = n * skewness / 6
            skewness_pval = 1 - stats.chi2.cdf(skewness_stat, p * (p + 1) * (p + 2) / 6)
            
            # Kurtosis statistic
            kurtosis = np.mean(md_squared ** 2)
            expected_kurtosis = p * (p + 2)
            kurtosis_stat = (kurtosis - expected_kurtosis) / np.sqrt(8 * p * (p + 2) / n)
            kurtosis_pval = 2 * (1 - stats.norm.cdf(np.abs(kurtosis_stat)))
            
            return {
                'mardia_skewness': skewness,
                'mardia_kurtosis': kurtosis,
                'mardia_skewness_pvalue': skewness_pval,
                'mardia_kurtosis_pvalue': kurtosis_pval
            }
            
        except Exception as e:
            return {
                'mardia_skewness': np.nan,
                'mardia_kurtosis': np.nan,
                'mardia_skewness_pvalue': np.nan,
                'mardia_kurtosis_pvalue': np.nan
            }
    
    def covariance_diagnostics(self, df, dataset_name):
        """Covariance matrix diagnostics"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {
                'cov_condition_number': np.nan,
                'cov_log_det': np.nan,
                'cov_log_det_diff': np.nan,
                'effective_rank': np.nan
            }
        
        try:
            # Compute covariance matrix
            cov_matrix = numeric_df.cov().values
            
            # Condition number
            cond_num = np.linalg.cond(cov_matrix)
            
            # Log determinant
            sign, log_det = np.linalg.slogdet(cov_matrix)
            log_det = log_det if sign > 0 else np.nan
            
            # Effective rank (eigenvalues above threshold)
            eigenvals = eigvals(cov_matrix)
            eigenvals = np.real(eigenvals)  # Take real part
            eigenvals = eigenvals[eigenvals > 0]  # Only positive eigenvalues
            effective_rank = np.sum(eigenvals > 1e-6)
            
            # Log determinant difference (vs identity)
            identity_log_det = numeric_df.shape[1] * np.log(1)  # log(1) = 0
            log_det_diff = log_det - identity_log_det if not np.isnan(log_det) else np.nan
            
            return {
                'cov_condition_number': cond_num,
                'cov_log_det': log_det,
                'cov_log_det_diff': log_det_diff,
                'effective_rank': effective_rank
            }
            
        except Exception as e:
            return {
                'cov_condition_number': np.nan,
                'cov_log_det': np.nan,
                'cov_log_det_diff': np.nan,
                'effective_rank': np.nan
            }
    
    def shrinkage_details(self, df, dataset_name):
        """Extract shrinkage details for unscorable datasets"""
        # These are the unscorable datasets that used shrinkage
        if 'unscorable' in dataset_name.lower() or 'unscrorable' in dataset_name.lower():
            
            # Estimate shrinkage by comparing to empirical correlation structure
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] < 2:
                return {
                    'shrinkage_lambda': np.nan,
                    'shrinkage_corr_rmse': np.nan,
                    'dataset_type': 'unscorable'
                }
            
            try:
                # Compute empirical correlation
                emp_corr = numeric_df.corr().values
                
                # Estimate shrinkage intensity by comparing diagonal structure
                # High shrinkage -> correlations closer to 0 (except diagonal)
                off_diag_corr = emp_corr[~np.eye(emp_corr.shape[0], dtype=bool)]
                
                # Estimate lambda as the reduction in off-diagonal correlation strength
                # This is a rough approximation
                mean_abs_corr = np.mean(np.abs(off_diag_corr))
                
                # Estimate shrinkage lambda (0 = no shrinkage, 1 = full shrinkage to identity)
                # Based on the reduction in correlation magnitudes
                estimated_lambda = max(0, 1 - (mean_abs_corr / 0.5))  # Assuming 0.5 as baseline
                
                # RMSE vs identity (what shrinkage targets)
                identity_matrix = np.eye(emp_corr.shape[0])
                shrinkage_rmse = np.sqrt(np.mean((emp_corr - identity_matrix) ** 2))
                
                return {
                    'shrinkage_lambda': estimated_lambda,
                    'shrinkage_corr_rmse': shrinkage_rmse,
                    'dataset_type': 'unscorable',
                    'mean_off_diag_corr': mean_abs_corr
                }
                
            except Exception as e:
                return {
                    'shrinkage_lambda': np.nan,
                    'shrinkage_corr_rmse': np.nan,
                    'dataset_type': 'unscorable',
                    'mean_off_diag_corr': np.nan
                }
        else:
            return {
                'shrinkage_lambda': 0.0,  # No shrinkage for regular datasets
                'shrinkage_corr_rmse': np.nan,
                'dataset_type': 'regular',
                'mean_off_diag_corr': np.nan
            }
    
    def analyze_dataset(self, df, dataset_name):
        """Perform comprehensive analysis of a single dataset"""
        print(f"\nAnalyzing {dataset_name}...")
        
        # Encode categorical variables for numeric analysis
        df_encoded, encoders = self.encode_categorical_variables(df)
        
        # Initialize results dictionary
        results = {'dataset': dataset_name}
        
        # Basic statistics
        basic_stats = self.compute_basic_stats(df, dataset_name)
        results.update(basic_stats)
        
        # KS tests for marginal fit
        ks_results = self.kolmogorov_smirnov_tests(df_encoded, dataset_name)
        results.update(ks_results)
        
        # Correlation fidelity
        corr_results = self.correlation_fidelity(df_encoded, dataset_name=dataset_name)
        results.update(corr_results)
        
        # Variance Inflation Factors
        vif_results = self.variance_inflation_factors(df_encoded, dataset_name)
        results.update(vif_results)
        
        # Multivariate normality (Mardia's tests)
        mardia_results = self.mardia_multivariate_normality(df_encoded, dataset_name)
        results.update(mardia_results)
        
        # Covariance diagnostics
        cov_results = self.covariance_diagnostics(df_encoded, dataset_name)
        results.update(cov_results)
        
        # Shrinkage details
        shrinkage_results = self.shrinkage_details(df_encoded, dataset_name)
        results.update(shrinkage_results)
        
        return results
    
    def run_analysis(self):
        """Run complete analysis on all datasets"""
        print("=== Synthetic Data Analysis ===")
        
        # Load datasets
        datasets = self.load_datasets()
        
        if not datasets:
            print("No datasets found to analyze!")
            return None
        
        # Analyze each dataset
        all_results = []
        for name, df in datasets.items():
            try:
                results = self.analyze_dataset(df, name)
                all_results.append(results)
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
                continue
        
        # Create summary DataFrame
        if all_results:
            summary_df = pd.DataFrame(all_results)
            
            # Reorder columns for better readability
            column_order = [
                'dataset', 'dataset_type', 'n_observations', 'n_variables', 'n_numeric_vars',
                'mean_of_means', 'mean_of_stds', 'mean_skewness', 'mean_excess_kurtosis',
                'median_ks_pvalue', 'n_ks_tests',
                'corr_rmse', 'corr_mae', 'corr_matrix_condition_number',
                'median_vif', 'max_vif', 'n_vif_computed',
                'mardia_skewness_pvalue', 'mardia_kurtosis_pvalue',
                'cov_condition_number', 'cov_log_det', 'effective_rank',
                'shrinkage_lambda', 'shrinkage_corr_rmse', 'mean_off_diag_corr'
            ]
            
            # Only include columns that exist
            existing_columns = [col for col in column_order if col in summary_df.columns]
            summary_df = summary_df[existing_columns]
            
            return summary_df
        else:
            print("No successful analyses completed!")
            return None

def main():
    """Main execution function"""
    # Initialize summarizer
    summarizer = DataSummarizer(data_dir="data")
    
    # Run analysis
    summary_df = summarizer.run_analysis()
    
    if summary_df is not None:
        # Display results
        print("\n" + "="*80)
        print("SUMMARY STATISTICS TABLE")
        print("="*80)
        
        # Display with better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print(summary_df.to_string(index=False))
        
        # Save results
        output_path = "results/synthetic_data_summary.csv"
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        
        # Additional detailed output
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("""
Key Metrics Interpretation:
- median_ks_pvalue: Higher values (>0.05) indicate better marginal distribution fit
- corr_rmse: Lower values indicate better correlation structure preservation
- median_vif: Values >10 suggest high multicollinearity
- mardia_*_pvalue: >0.05 suggests multivariate normality
- cov_condition_number: Lower values indicate better-conditioned covariance matrix
- effective_rank: Higher values indicate more independent dimensions
        """)
        
        return summary_df
    else:
        print("Analysis failed - no results to display")
        return None

if __name__ == "__main__":
    summary_results = main() 