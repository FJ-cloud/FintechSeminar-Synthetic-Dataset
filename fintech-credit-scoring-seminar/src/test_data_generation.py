#!/usr/bin/env python3
"""
Test Dataset Generation Script

This script generates dedicated test datasets using the same logic as the training datasets
but with different random seeds to ensure complete independence and no data leakage.

Test datasets will be smaller (10,000 samples each) and maintain the correct default rates:
- Scorable: 1.0% default rate
- Unscorable: 2.5% default rate

"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Use different random seed for test data generation
TEST_RANDOM_SEED = 12345
np.random.seed(TEST_RANDOM_SEED)

print("GENERATING DEDICATED TEST DATASETS")
print(f"Using random seed: {TEST_RANDOM_SEED}")

N_TEST = 10000

# Variables and categories
variables = [
    "credit_score_quintile", "device_type", "os", "email_host", "channel", "checkout_time",
    "name_in_email", "number_in_email", "is_lowercase", "email_error",
    "age_quintile", "order_amount_quintile", "item_category", "month"
]

categories = {
    "credit_score_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "device_type": ["Desktop", "Tablet", "Mobile", "Do-not-track"],
    "os": ["Windows", "iOS", "Android", "Macintosh", "Other", "Do-not-track"],
    "email_host": ["Gmx", "Web", "T-Online", "Gmail", "Yahoo", "Hotmail", "Other"],
    "channel": ["Paid", "Direct", "Affiliate", "Organic", "Other", "Do-not-track"],
    "checkout_time": ["Evening", "Night", "Morning", "Afternoon"],
    "name_in_email": ["No", "Yes"],
    "number_in_email": ["No", "Yes"],
    "is_lowercase": ["No", "Yes"],
    "email_error": ["No", "Yes"],
    "age_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "order_amount_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "item_category": [f"Cat{i}" for i in range(1, 17)],
    "month": ["Oct15", "Nov15", "Dec15", "Jan16", "Feb16", "Mar16", "Apr16", 
              "May16", "Jun16", "Jul16", "Aug16", "Sep16", "Oct16", "Nov16", "Dec16"]
}

marginals_list = [
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.57, 0.18, 0.11, 0.14],
    [0.49, 0.16, 0.11, 0.08, 0.01, 0.14],
    [0.23, 0.22, 0.12, 0.11, 0.05, 0.04, 0.24],
    [0.44, 0.18, 0.10, 0.07, 0.07, 0.14],
    [0.43, 0.03, 0.18, 0.36],
    [0.28, 0.72],
    [0.84, 0.16],
    [0.99, 0.01],
    [0.92, 0.08],
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [1/16]*16,
    [1/15]*15
]

cramers_v_array = np.array([
    [1.00, 0.07, 0.05, 0.07, 0.03, 0.03, 0.01, 0.07, 0.02, 0.00, 0.2, 0.01, 0.05, 0.01],
    [0.07, 1.00, 0.71, 0.07, 0.06, 0.04, 0.05, 0.06, 0.07, 0.01, 0.12, 0.03, 0.05, 0.06],
    [0.05, 0.71, 1.00, 0.08, 0.06, 0.04, 0.06, 0.08, 0.06, 0.01, 0.1, 0.02, 0.04, 0.03],
    [0.07, 0.07, 0.08, 1.00, 0.03, 0.03, 0.08, 0.18, 0.04, 0.06, 0.16, 0.02, 0.02, 0.01],
    [0.03, 0.06, 0.06, 0.03, 1.00, 0.02, 0.01, 0.02, 0.04, 0.02, 0.09, 0.04, 0.06, 0.13],
    [0.03, 0.04, 0.04, 0.03, 0.02, 1.00, 0.01, 0.01, 0.01, 0.01, 0.06, 0.01, 0.03, 0.02],
    [0.01, 0.05, 0.06, 0.08, 0.01, 0.01, 1.00, 0.22, 0.01, 0.02, 0.04, 0.01, 0.03, 0.01],
    [0.07, 0.06, 0.08, 0.18, 0.02, 0.01, 0.22, 1.00, 0.02, 0.00, 0.06, 0.01, 0.04, 0.01],
    [0.02, 0.07, 0.06, 0.04, 0.04, 0.01, 0.01, 0.02, 1.00, 0.03, 0.03, 0.02, 0.02, 0.02],
    [0.00, 0.01, 0.01, 0.06, 0.02, 0.01, 0.02, 0.00, 0.03, 1.00, 0.03, 0.01, 0.01, 0.01],
    [0.2, 0.12, 0.1, 0.16, 0.09, 0.06, 0.04, 0.06, 0.03, 0.03, 1.00, 0.05, 0.11, 0.03],
    [0.01, 0.03, 0.02, 0.02, 0.04, 0.01, 0.01, 0.01, 0.02, 0.01, 0.05, 1.00, 0.27, 0.02],
    [0.05, 0.05, 0.04, 0.02, 0.06, 0.03, 0.03, 0.04, 0.02, 0.01, 0.11, 0.27, 1.00, 0.11],
    [0.01, 0.06, 0.03, 0.01, 0.13, 0.02, 0.01, 0.01, 0.02, 0.01, 0.03, 0.02, 0.11, 1.00]
])

def generate_continuous_variables(n_samples):
    # Age
    ages = np.random.normal(loc=45.06, scale=13.31, size=n_samples)
    ages = np.clip(np.round(ages), 18, 80).astype(int)
    
    # Order Amount
    mu = np.log(219)
    sigma = np.sqrt(2 * (np.log(318) - mu))
    order_amounts = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
    order_amounts = np.clip(order_amounts, 10, 1500)
    order_amounts = np.round(order_amounts, 2)
    
    return ages, order_amounts

def generate_synthetic_data(n_samples, target_default_rate, population_type, method):
    print(f"Generating {method} test data ({population_type})...")
    
    # Generate continuous variables
    ages, order_amounts = generate_continuous_variables(n_samples)
    
    # Initialize DataFrame
    synthetic = pd.DataFrame({
        "age": ages,
        "order_amount": order_amounts
    })
    
    # Bin into quintiles
    synthetic['age_quintile'] = pd.qcut(synthetic['age'], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    synthetic['order_amount_quintile'] = pd.qcut(synthetic['order_amount'], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    
    # Check Cramer's V matrix
    eigvals = np.linalg.eigvalsh(cramers_v_array)
    if np.any(eigvals < 0):
        cramers_v_matrix = np.eye(14)
    else:
        cramers_v_matrix = cramers_v_array
    
    # Compute thresholds
    thresholds_list = []
    for p in marginals_list:
        cumprob = np.cumsum(p)[:-1]
        thresholds = norm.ppf(cumprob)
        thresholds_list.append(thresholds)
    
    # Generate multivariate normal data
    Z = np.random.multivariate_normal(mean=np.zeros(14), cov=cramers_v_matrix, size=n_samples)
    
    # Map to categorical variables
    for i, var in enumerate(variables):
        thresholds = thresholds_list[i]
        z = Z[:, i]
        cat_indices = np.searchsorted(thresholds, z, side='right')
        synthetic[var] = [categories[var][idx] for idx in cat_indices]
    
    # Add gender
    synthetic['gender'] = np.random.choice(['Female', 'Male'], size=n_samples, p=[0.66, 0.34])
    
    # Assign TARGET with correct default rates
    if population_type == 'scorable':
        default_rates = {
            "credit_score_quintile": {"Q1": 0.0212, "Q2": 0.0102, "Q3": 0.0068, "Q4": 0.0047, "Q5": 0.0039},
            "device_type": {"Desktop": 0.0074, "Tablet": 0.0091, "Mobile": 0.0214, "Do-not-track": 0.0088},
            "os": {"Windows": 0.0074, "iOS": 0.0107, "Android": 0.0179, "Macintosh": 0.0069, "Other": 0.0109, "Do-not-track": 0.0088},
            "email_host": {"Gmx": 0.0082, "Web": 0.0086, "T-Online": 0.0051, "Gmail": 0.0125, "Yahoo": 0.0196, "Hotmail": 0.0145, "Other": 0.0090},
            "channel": {"Paid": 0.0111, "Direct": 0.0084, "Affiliate": 0.0064, "Organic": 0.0086, "Other": 0.0069, "Do-not-track": 0.0088},
            "checkout_time": {"Evening": 0.0085, "Night": 0.0197, "Morning": 0.0109, "Afternoon": 0.0089},
            "name_in_email": {"No": 0.0124, "Yes": 0.0082},
            "number_in_email": {"No": 0.0084, "Yes": 0.0141},
            "is_lowercase": {"No": 0.0084, "Yes": 0.0214},
            "email_error": {"No": 0.0088, "Yes": 0.0509},
        }
    else:  # unscorable
        default_rates = {
            "credit_score_quintile": {"Q1": 0.0212, "Q2": 0.0102, "Q3": 0.0068, "Q4": 0.0047, "Q5": 0.0039},
            "device_type": {"Desktop": 0.0216, "Tablet": 0.0164, "Mobile": 0.0621, "Do-not-track": 0.0228},
            "os": {"Windows": 0.0219, "iOS": 0.0235, "Android": 0.0480, "Macintosh": 0.0169, "Other": 0.0745, "Do-not-track": 0.0228},
            "email_host": {"Gmx": 0.0242, "Web": 0.0263, "T-Online": 0.0152, "Gmail": 0.0361, "Yahoo": 0.0315, "Hotmail": 0.0275, "Other": 0.0222},
            "channel": {"Paid": 0.0289, "Direct": 0.0187, "Affiliate": 0.0265, "Organic": 0.0255, "Other": 0.0215, "Do-not-track": 0.0228},
            "checkout_time": {"Evening": 0.0205, "Night": 0.0352, "Morning": 0.0274, "Afternoon": 0.0278},
            "name_in_email": {"No": 0.0124, "Yes": 0.0082},
            "number_in_email": {"No": 0.0084, "Yes": 0.0141},
            "is_lowercase": {"No": 0.0084, "Yes": 0.0214},
            "email_error": {"No": 0.0088, "Yes": 0.0509},
        }
    
    # Calculate default probabilities
    cat_vars = [var for var in default_rates if var in synthetic.columns]
    default_probs = np.zeros(n_samples)
    for var in cat_vars:
        default_probs += synthetic[var].map(default_rates[var]).values
    default_probs /= len(cat_vars)
    
    # Apply precise calibration to target default rate
    current_mean = default_probs.mean()
    scaling_factor = target_default_rate / current_mean
    adjusted_probs = np.clip(default_probs * scaling_factor, 0, 1)
    
    # Sort and select exact number of defaults
    num_defaults_needed = int(n_samples * target_default_rate)
    sorted_indices = np.argsort(adjusted_probs)[::-1]
    synthetic['TARGET'] = 0
    synthetic.loc[sorted_indices[:num_defaults_needed], 'TARGET'] = 1
    
    print(f"Generated {n_samples} samples with {synthetic['TARGET'].sum()} defaults ({synthetic['TARGET'].mean():.3%})")
    
    return synthetic

# Main generation process
data_dir = Path("../data/test/")
data_dir.mkdir(parents=True, exist_ok=True)

# Generate test datasets
datasets_to_generate = [
    ('scorable', 'basic', 0.01, 'test_synthetic_digital_footprint_with_target.csv'),
    ('scorable', 'copula', 0.01, 'test_synthetic_digital_footprint_copula.csv'),
    ('unscorable', 'basic', 0.025, 'test_synthetic_digital_footprint_with_target_unscorable.csv'),
    ('unscorable', 'copula', 0.025, 'test_synthetic_digital_footprint_copula_unscorable.csv'),
]

generated_files = []

for population_type, method, target_rate, filename in datasets_to_generate:
    print(f"\nGENERATING: {filename}")
    print(f"Population: {population_type}, Method: {method}, Target rate: {target_rate:.1%}")
    
    try:
        test_data = generate_synthetic_data(N_TEST, target_rate, population_type, method)
        
        # Save dataset
        filepath = data_dir / filename
        test_data.to_csv(filepath, index=False)
        generated_files.append(filename)
        
        print(f"Saved: {filepath}")
        print(f"Shape: {test_data.shape}")
        print(f"Actual default rate: {test_data['TARGET'].mean():.3%}")
        
    except Exception as e:
        print(f"Error generating {filename}: {str(e)}")
        continue

# Summary
print(f"\nTEST DATASET GENERATION COMPLETE")
print(f"Generated {len(generated_files)} test datasets:")
for filename in generated_files:
    print(f"  - {filename}")

print(f"\nTest datasets are completely independent from training datasets.")
print(f"Random seed used: {TEST_RANDOM_SEED}")
print(f"Dataset size: {N_TEST:,} samples each")

if __name__ == "__main__":
    generate_test_datasets() 