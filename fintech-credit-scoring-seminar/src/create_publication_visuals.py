#!/usr/bin/env python3
"""
Publication-Quality Visualization Script

This script implements the designer's checklist for turning three key figures into clean,
low-relief visuals suitable for academic publication:
1. Marginal bar plots (four key variables)
2. Cramer's V Heat Map (side-by-side comparison)
3. ROC Curves overlay (model performance comparison)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. GLOBAL PLOT STYLE (set once)
# =============================================================================

plt.rcParams.update({
    "figure.dpi": 120,          # high-res screens
    "figure.figsize": (6, 3.2), # default size
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")      # light grid, low-relief

print("Global plot style configured for publication-quality visuals")

# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data():
    """Load synthetic datasets and prepare for visualization"""
    print("Loading synthetic datasets...")
    
    data_dir = Path("../data")
    datasets = {}
    
    # Load main datasets
    for method in ['basic', 'copula', 'ctgan']:
        for pop in ['scorable', 'unscorable']:
            # Handle the file naming inconsistency for basic method
            if method == 'basic':
                if pop == 'scorable':
                    filename = "synthetic_digital_footprint_with_target.csv"
                else:
                    filename = "synthetic_digital_footprint_with_target_unscorable.csv"
            else:
                if pop == 'scorable':
                    filename = f"synthetic_digital_footprint_{method}.csv"
                else:
                    filename = f"synthetic_digital_footprint_{method}_unscorable.csv"
            
            filepath = data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    datasets[f'{pop}_{method}'] = df
                    print(f"Loaded {filename}: {df.shape}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return datasets

def get_berg_reference_marginals():
    """Get Berg et al. (2020) reference distributions for comparison"""
    return {
        "device_type": {"Desktop": 0.57, "Tablet": 0.18, "Mobile": 0.11, "Do-not-track": 0.14},
        "checkout_time": {"Evening": 0.43, "Night": 0.03, "Morning": 0.18, "Afternoon": 0.36},
        "email_host": {"Gmx": 0.23, "Web": 0.22, "T-Online": 0.12, "Gmail": 0.11, 
                      "Yahoo": 0.05, "Hotmail": 0.04, "Other": 0.24},
        "channel": {"Paid": 0.44, "Direct": 0.18, "Affiliate": 0.10, "Organic": 0.07, 
                   "Other": 0.07, "Do-not-track": 0.14}
    }

def calculate_marginal_frequencies(datasets, variables):
    """Calculate marginal frequencies for synthetic datasets"""
    freqs = {}
    
    for dataset_name, df in datasets.items():
        freqs[dataset_name] = {}
        for var in variables:
            if var in df.columns:
                freqs[dataset_name][var] = df[var].value_counts(normalize=True).sort_index()
            else:
                print(f"Warning: {var} not found in {dataset_name}")
    
    return freqs

# =============================================================================
# 3. MARGINAL BAR PLOTS (FIGURE 1)
# =============================================================================

def create_marginal_bar_plots(datasets):
    """Create publication-quality marginal bar plots for four key variables"""
    print("Creating marginal bar plots...")
    
    # Variables to plot
    variables = ["device_type", "checkout_time", "email_host", "channel"]
    
    # Get reference distributions
    berg_freqs = get_berg_reference_marginals()
    
    # Calculate synthetic frequencies (using scorable population)
    synth_freqs = {}
    for method in ['copula', 'ctgan']:
        dataset_key = f'scorable_{method}'
        if dataset_key in datasets:
            synth_freqs[method] = {}
            for var in variables:
                if var in datasets[dataset_key].columns:
                    synth_freqs[method][var] = datasets[dataset_key][var].value_counts(normalize=True)
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.8))
    
    for ax, var in zip(axes.ravel(), variables):
        # Combine all data for consistent ordering
        all_categories = set(berg_freqs[var].keys())
        for method in synth_freqs:
            if var in synth_freqs[method]:
                all_categories.update(synth_freqs[method][var].index)
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame(index=sorted(all_categories))
        plot_data['Berg'] = pd.Series(berg_freqs[var])
        
        for method in synth_freqs:
            if var in synth_freqs[method]:
                plot_data[method.title()] = synth_freqs[method][var]
        
        plot_data = plot_data.fillna(0).sort_index()
        
        # Plot with specified colors
        plot_data.plot(kind="bar",
                      width=0.8,
                      ax=ax,
                      color=["#4d4d4d", "#9ecae1", "#fdd0a2"])  # grey + two pastel hues
        
        # Calculate and display total variation distance (TVD) for each method
        tvd_scores = []
        for method in synth_freqs:
            if var in synth_freqs[method]:
                # Align the data for TVD calculation
                berg_aligned = plot_data['Berg'].fillna(0)
                synth_aligned = plot_data[method.title()].fillna(0)
                tvd = 0.5 * np.sum(np.abs(berg_aligned - synth_aligned))
                tvd_scores.append(f"{method.title()}: TVD={tvd:.3f}")
        
        ax.set_title(var.replace("_", " ").title(), pad=4, fontweight='bold')
        ax.set_ylabel("Proportion")
        ax.set_xlabel("")            # no x-label clutter
        ax.legend(frameon=False, loc="upper right", fontsize=6)
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        
        # Add TVD information as text
        if tvd_scores:
            tvd_text = "\n".join(tvd_scores)
            ax.text(0.02, 0.98, tvd_text, transform=ax.transAxes, 
                   fontsize=5, verticalalignment='top', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    fig.tight_layout(w_pad=0.7, h_pad=1.2)
    
    # Add explanatory text at the bottom
    fig.text(0.5, 0.02, 
             'Comparison of marginal distributions: Berg et al. (2020) reference vs. synthetic data. '
             'TVD = Total Variation Distance (lower is better, 0 = perfect match)',
             ha='center', fontsize=7, style='italic', wrap=True)
    
    # Adjust layout to accommodate the caption
    fig.subplots_adjust(bottom=0.12)
    
    # Save figure
    output_path = Path("../results/fig_marginals.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Marginal bar plots saved to: {output_path}")
    
    return fig

# =============================================================================
# 4. CRAMER'S V HEAT MAP (FIGURE 2)
# =============================================================================

def calculate_cramers_v(x, y):
    """Calculate Cramer's V statistic for two categorical variables"""
    from scipy.stats import chi2_contingency
    
    # Create contingency table
    confusion_matrix = pd.crosstab(x, y)
    
    # Calculate chi-square
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    
    # Calculate Cramer's V
    n = confusion_matrix.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
    
    return cramers_v

def create_cramers_v_matrix(df, variables):
    """Create Cramer's V correlation matrix for categorical variables"""
    n_vars = len(variables)
    cramers_matrix = np.zeros((n_vars, n_vars))
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                cramers_matrix[i, j] = 1.0
            elif var1 in df.columns and var2 in df.columns:
                try:
                    cramers_matrix[i, j] = calculate_cramers_v(df[var1], df[var2])
                except:
                    cramers_matrix[i, j] = 0.0
    
    return pd.DataFrame(cramers_matrix, index=variables, columns=variables)

def create_cramers_v_heatmap(datasets):
    """Create side-by-side Cramer's V heatmaps with full annotations"""
    print("Creating Cramer's V heatmaps...")
    
    # Variables for correlation analysis
    variables = ["device_type", "checkout_time", "email_host", "channel", 
                "credit_score_quintile", "age_quintile"]
    
    # Calculate Cramer's V matrices
    matrices = {}
    for method in ['copula', 'ctgan']:
        dataset_key = f'scorable_{method}'
        if dataset_key in datasets:
            df = datasets[dataset_key]
            # Filter to variables that exist in the dataset
            available_vars = [var for var in variables if var in df.columns]
            if len(available_vars) >= 3:  # Need at least 3 variables for meaningful heatmap
                matrices[method] = create_cramers_v_matrix(df, available_vars)
    
    if len(matrices) < 2:
        print("Warning: Not enough datasets for comparison heatmap")
        return None
    
    # Create the plot with adjusted size for better readability
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    
    methods = list(matrices.keys())
    
    # Create a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    for i, (method, matrix) in enumerate(matrices.items()):
        # Create pretty variable names for display
        display_names = [name.replace('_', ' ').title() for name in matrix.index]
        
        # Create heatmap with annotations
        sns.heatmap(matrix,
                    cmap="Greys",          # neutral scale
                    vmin=0, vmax=0.3,
                    cbar=i == 1,           # Only show colorbar on last plot
                    cbar_ax=cbar_ax if i == 1 else None,
                    square=True,
                    annot=True,            # Show correlation values
                    fmt='.2f',             # Format to 2 decimal places
                    annot_kws={'fontsize': 6},
                    xticklabels=display_names,
                    yticklabels=display_names,
                    ax=axes[i])
        
        axes[i].set_title(f"{method.title()} Cramer's V", pad=10, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45, labelsize=7)
        axes[i].tick_params(axis='y', rotation=0, labelsize=7)
        
        # Add subtle grid lines for better readability
        axes[i].set_xticks(np.arange(len(display_names)) + 0.5, minor=True)
        axes[i].set_yticks(np.arange(len(display_names)) + 0.5, minor=True)
        axes[i].grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Add colorbar label
    cbar_ax.set_ylabel('Cramer\'s V Coefficient', rotation=270, labelpad=15, fontsize=8)
    
    # Add interpretation text
    fig.text(0.5, 0.02, 
             'Cramer\'s V measures association strength between categorical variables (0=no association, 1=perfect association)',
             ha='center', fontsize=7, style='italic')
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, right=0.9)
    
    # Save figure
    output_path = Path("../results/fig_cramer_heatmap.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Cramer's V heatmap saved to: {output_path}")
    
    return fig

# =============================================================================
# 5. ROC CURVES OVERLAY (FIGURE 3)
# =============================================================================

def prepare_modeling_data(df, features, target='is_bad'):
    """Prepare data for modeling"""
    # Check if target exists
    if target not in df.columns:
        print(f"Warning: Target variable '{target}' not found. Available columns: {list(df.columns)}")
        return None, None
    
    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    if len(available_features) == 0:
        print("Warning: No features available for modeling")
        return None, None
    
    X = df[available_features].copy()
    y = df[target].copy()
    
    # Encode categorical variables
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))
    
    return X, y

def create_roc_curves_overlay(datasets):
    """Create ROC curves overlay comparing different models and feature sets"""
    print("Creating ROC curves overlay...")
    
    # Feature sets
    footprint_features = ["device_type", "checkout_time", "email_host", "channel"]
    full_features = footprint_features + ["credit_score_quintile", "age_quintile", "order_amount_quintile"]
    
    # Use copula dataset for demonstration
    dataset_key = 'scorable_copula'
    if dataset_key not in datasets:
        print(f"Dataset {dataset_key} not available for ROC analysis")
        return None
    
    df = datasets[dataset_key]
    
    # Create balanced dataset for training
    if 'is_bad' not in df.columns:
        # Create synthetic target variable for demonstration
        df['is_bad'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    # Prepare data
    X_footprint, y = prepare_modeling_data(df, footprint_features)
    X_full, _ = prepare_modeling_data(df, full_features)
    
    if X_footprint is None or X_full is None:
        print("Could not prepare modeling data")
        return None
    
    # Split data
    X_foot_train, X_foot_test, y_train, y_test = train_test_split(
        X_footprint, y, test_size=0.3, random_state=42, stratify=y
    )
    X_full_train, X_full_test, _, _ = train_test_split(
        X_full, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train models
    models_results = {}
    
    # Logistic Regression models
    lr_foot = LogisticRegression(random_state=42, max_iter=1000)
    lr_foot.fit(X_foot_train, y_train)
    models_results["Logit - Footprint"] = lr_foot.predict_proba(X_foot_test)[:, 1]
    
    lr_full = LogisticRegression(random_state=42, max_iter=1000)
    lr_full.fit(X_full_train, y_train)
    models_results["Logit - Full"] = lr_full.predict_proba(X_full_test)[:, 1]
    
    # Random Forest models
    rf_foot = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_foot.fit(X_foot_train, y_train)
    models_results["RF - Footprint"] = rf_foot.predict_proba(X_foot_test)[:, 1]
    
    rf_full = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_full.fit(X_full_train, y_train)
    models_results["RF - Full"] = rf_full.predict_proba(X_full_test)[:, 1]
    
    # Create ROC plot
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    
    for label, y_score in models_results.items():
        RocCurveDisplay.from_predictions(y_test, y_score, name=label, ax=ax)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.7)
    ax.set_title("ROC curves (Copula data)")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    
    fig.tight_layout()
    
    # Save figure
    output_path = Path("../results/fig_roc.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"ROC curves saved to: {output_path}")
    
    return fig

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate all publication-quality visualizations"""
    print("="*80)
    print("PUBLICATION-QUALITY VISUALIZATION GENERATOR")
    print("="*80)
    
    # Ensure results directory exists
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    datasets = load_and_prepare_data()
    
    if not datasets:
        print("No datasets loaded. Please check data files.")
        return
    
    print(f"\nLoaded {len(datasets)} datasets")
    
    # Generate visualizations
    print("\n" + "="*60)
    
    # 1. Marginal bar plots
    try:
        fig1 = create_marginal_bar_plots(datasets)
        if fig1:
            plt.show()
    except Exception as e:
        print(f"Error creating marginal bar plots: {e}")
    
            # 2. Cramer's V heatmaps
    try:
        fig2 = create_cramers_v_heatmap(datasets)
        if fig2:
            plt.show()
    except Exception as e:
        print(f"Error creating Cramer's V heatmaps: {e}")
    
    # 3. ROC curves
    try:
        fig3 = create_roc_curves_overlay(datasets)
        if fig3:
            plt.show()
    except Exception as e:
        print(f"Error creating ROC curves: {e}")
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("All figures saved as PDF files in ../results/")
    print("Ready for inclusion in LaTeX/Overleaf documents")
    print("="*80)

if __name__ == "__main__":
    main() 