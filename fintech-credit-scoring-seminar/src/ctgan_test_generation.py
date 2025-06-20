#!/usr/bin/env python3
"""
CTGAN Test Dataset Generation Script

This script generates CTGAN test datasets using the same logic as the training datasets
but with different random seeds to ensure complete independence.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import CTGAN
try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
    print("CTGAN library available")
except ImportError:
    CTGAN_AVAILABLE = False
    print("Warning: CTGAN library not available. Install with: pip install ctgan")

# Use different random seed for test data generation
TEST_RANDOM_SEED = 12345
np.random.seed(TEST_RANDOM_SEED)

def generate_ctgan_test_datasets():
    """Generate CTGAN test datasets"""
    
    if not CTGAN_AVAILABLE:
        print("CTGAN not available. Skipping CTGAN test dataset generation.")
        return
    
    print("="*80)
    print("GENERATING CTGAN TEST DATASETS")
    print("="*80)
    print(f"Using random seed: {TEST_RANDOM_SEED}")
    
    data_dir = Path("../data")
    test_dir = Path("../data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    N_TEST = 10000  # Test dataset size
    
    # Load training datasets to train CTGAN models
    training_files = {
        'scorable': 'synthetic_digital_footprint_with_target.csv',
        'unscorable': 'synthetic_digital_footprint_with_target_unscorable.csv'
    }
    
    # Target default rates
    target_rates = {
        'scorable': 0.01,    # 1.0%
        'unscorable': 0.025  # 2.5%
    }
    
    # Output files
    output_files = {
        'scorable': 'test_synthetic_digital_footprint_ctgan.csv',
        'unscorable': 'test_synthetic_digital_footprint_ctgan_unscorable.csv'
    }
    
    for population_type in ['scorable', 'unscorable']:
        print(f"\n{'='*60}")
        print(f"GENERATING CTGAN TEST DATA - {population_type.upper()}")
        print(f"{'='*60}")
        
        # Load training data
        training_file = data_dir / training_files[population_type]
        if not training_file.exists():
            print(f"Warning: Training file {training_file} not found. Skipping {population_type}.")
            continue
        
        print(f"Loading training data: {training_file}")
        train_df = pd.read_csv(training_file)
        print(f"Training data shape: {train_df.shape}")
        print(f"Training default rate: {train_df['TARGET'].mean():.3%}")
        
        # Prepare data for CTGAN
        print("Preparing data for CTGAN...")
        
        # Define discrete columns (categorical variables)
        discrete_columns = [
            'credit_score_quintile', 'device_type', 'os', 'email_host', 'channel', 
            'checkout_time', 'name_in_email', 'number_in_email', 'is_lowercase', 
            'email_error', 'age_quintile', 'order_amount_quintile', 'item_category', 
            'month', 'gender', 'TARGET'
        ]
        
        # Filter to only existing columns
        existing_discrete_columns = [col for col in discrete_columns if col in train_df.columns]
        print(f"Discrete columns: {existing_discrete_columns}")
        
        # Initialize and train CTGAN
        print("Initializing CTGAN model...")
        ctgan = CTGAN(
            epochs=100,  # Reduced for faster generation, increase if needed
            batch_size=500,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            verbose=True
        )
        
        print("Training CTGAN model... (this may take several minutes)")
        ctgan.fit(train_df, discrete_columns=existing_discrete_columns)
        
        # Generate synthetic data
        print(f"Generating {N_TEST} synthetic samples...")
        synthetic_data = ctgan.sample(N_TEST)
        
        print(f"Generated data shape: {synthetic_data.shape}")
        print(f"Generated default rate: {synthetic_data['TARGET'].mean():.3%}")
        
        # Apply precise calibration to target default rate
        target_rate = target_rates[population_type]
        print(f"Calibrating to target default rate: {target_rate:.1%}")
        
        # Separate defaults and non-defaults
        defaults = synthetic_data[synthetic_data['TARGET'] == 1]
        non_defaults = synthetic_data[synthetic_data['TARGET'] == 0]
        
        print(f"Generated defaults: {len(defaults)}")
        print(f"Generated non-defaults: {len(non_defaults)}")
        
        # Calculate exact number of defaults needed
        num_defaults_needed = int(N_TEST * target_rate)
        num_non_defaults_needed = N_TEST - num_defaults_needed
        
        print(f"Target defaults: {num_defaults_needed}")
        print(f"Target non-defaults: {num_non_defaults_needed}")
        
        # Sample to get exact target distribution
        if len(defaults) >= num_defaults_needed:
            defaults_sample = defaults.sample(n=num_defaults_needed, random_state=TEST_RANDOM_SEED)
        else:
            print(f"Warning: Not enough defaults generated ({len(defaults)}). Using all available.")
            defaults_sample = defaults
            num_defaults_needed = len(defaults)
            num_non_defaults_needed = N_TEST - num_defaults_needed
        
        if len(non_defaults) >= num_non_defaults_needed:
            non_defaults_sample = non_defaults.sample(n=num_non_defaults_needed, random_state=TEST_RANDOM_SEED)
        else:
            print(f"Warning: Not enough non-defaults generated ({len(non_defaults)}). Using all available.")
            non_defaults_sample = non_defaults
        
        # Combine and shuffle
        calibrated_data = pd.concat([defaults_sample, non_defaults_sample], ignore_index=True)
        calibrated_data = calibrated_data.sample(frac=1, random_state=TEST_RANDOM_SEED).reset_index(drop=True)
        
        final_default_rate = calibrated_data['TARGET'].mean()
        
        # Save the test dataset
        output_file = test_dir / output_files[population_type]
        calibrated_data.to_csv(output_file, index=False)
        
        print(f"\nCTGAN test dataset saved: {output_file}")
        print(f"Final shape: {calibrated_data.shape}")
        print(f"Final default rate: {final_default_rate:.3%}")
        print(f"Target achieved: {abs(final_default_rate - target_rate) < 0.001}")
    
    print(f"\n{'='*80}")
    print("CTGAN TEST DATASET GENERATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    generate_ctgan_test_datasets() 