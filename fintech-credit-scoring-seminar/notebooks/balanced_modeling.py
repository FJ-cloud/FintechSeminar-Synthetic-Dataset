#!/usr/bin/env python3
"""
Balanced Training with Realistic Testing for Credit Scoring Models

This script implements a more realistic modeling approach:
1. Training sets: Balanced samples (higher default rate for better learning)
2. Test sets: Real-world distribution samples (preserving actual default rates)
3. No overlap between training and testing sets
4. Proper handling of scorable vs unscorable populations

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_datasets():
    """Load all synthetic datasets"""
    data_dir = Path("FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/data")
    
    datasets = {}
    
    # Scorable population datasets
    scorable_files = {
        'basic': 'synthetic_digital_footprint_with_target.csv',
        'copula': 'synthetic_digital_footprint_copula.csv',
        'ctgan': 'synthetic_digital_footprint_ctgan.csv'
    }
    
    # Unscorable population datasets  
    unscorable_files = {
        'basic': 'synthetic_digital_footprint_with_target_unscorable.csv',
        'copula': 'synthetic_digital_footprint_copula_unscorable.csv',
        'ctgan': 'synthetic_digital_footprint_ctgan_unscorable.csv'
    }
    
    print("Loading datasets...")
    
    # Load scorable datasets
    for method, filename in scorable_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            datasets[f'scorable_{method}'] = df
            print(f"Loaded {filename}: {df.shape}, Default rate: {df['TARGET'].mean():.3%}")
        else:
            print(f"Warning: {filename} not found")
    
    # Load unscorable datasets
    for method, filename in unscorable_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            datasets[f'unscorable_{method}'] = df
            print(f"Loaded {filename}: {df.shape}, Default rate: {df['TARGET'].mean():.3%}")
        else:
            print(f"Warning: {filename} not found")
    
    return datasets

def create_balanced_training_set(df, population_type='scorable', train_size=5000):
    """
    Create balanced training sets with higher default rates for better model learning
    
    Parameters:
    - df: Full dataset
    - population_type: 'scorable' or 'unscorable'
    - train_size: Total size of training set
    
    Returns:
    - Training set with balanced class distribution
    """
    defaults = df[df['TARGET'] == 1]
    non_defaults = df[df['TARGET'] == 0]
    
    print(f"\nCreating balanced training set for {population_type} population:")
    print(f"Available defaults: {len(defaults)}")
    print(f"Available non-defaults: {len(non_defaults)}")
    
    if population_type == 'unscorable':
        # Unscorable: Use 1000 defaults (should have ~2500 total)
        if len(defaults) >= 1000:
            selected_defaults = defaults.sample(n=1000, random_state=42)
            selected_non_defaults = non_defaults.sample(n=train_size-1000, random_state=42)
            balance_ratio = 1000 / train_size
        else:
            print(f"Warning: Only {len(defaults)} defaults available, using all")
            selected_defaults = defaults
            selected_non_defaults = non_defaults.sample(n=train_size-len(defaults), random_state=42)
            balance_ratio = len(defaults) / train_size
    else:
        # Scorable: Reserve enough defaults for test set to maintain 1% default rate
        # For 10,000 test samples at 1% = 100 defaults needed for test
        # Use remaining defaults for training (should be ~900 for training)
        defaults_for_test = 100  # Reserve for test set
        defaults_for_training = min(500, len(defaults) - defaults_for_test)  # Use up to 500 for training
        
        if len(defaults) >= (defaults_for_test + defaults_for_training):
            selected_defaults = defaults.sample(n=defaults_for_training, random_state=42)
            selected_non_defaults = non_defaults.sample(n=train_size-defaults_for_training, random_state=42)
            balance_ratio = defaults_for_training / train_size
        else:
            print(f"Warning: Only {len(defaults)} defaults available, using conservative split")
            # Use half for training, half for test
            defaults_for_training = len(defaults) // 2
            selected_defaults = defaults.sample(n=defaults_for_training, random_state=42)
            selected_non_defaults = non_defaults.sample(n=train_size-defaults_for_training, random_state=42)
            balance_ratio = defaults_for_training / train_size
    
    # Combine and shuffle
    train_set = pd.concat([selected_defaults, selected_non_defaults], ignore_index=True)
    train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training set created: {len(train_set)} samples")
    print(f"Training default rate: {train_set['TARGET'].mean():.1%} (balance ratio: {balance_ratio:.1%})")
    
    return train_set, selected_defaults.index, selected_non_defaults.index

def create_realistic_test_set(df, used_indices, population_type='scorable', test_size=10000):
    """
    Create test set that preserves real-world distribution and doesn't overlap with training
    
    Parameters:
    - df: Full dataset
    - used_indices: Indices already used in training
    - population_type: 'scorable' or 'unscorable'
    - test_size: Desired test set size
    
    Returns:
    - Test set with realistic distribution
    """
    # Remove training samples
    available_df = df.drop(used_indices).reset_index(drop=True)
    
    print(f"\nCreating realistic test set:")
    print(f"Available samples after removing training: {len(available_df)}")
    print(f"Available default rate: {available_df['TARGET'].mean():.3%}")
    
    # For scorable populations, ensure we maintain at least 1% default rate
    if population_type == 'scorable':
        available_defaults = available_df[available_df['TARGET'] == 1]
        available_non_defaults = available_df[available_df['TARGET'] == 0]
        
        # Target: 1% default rate in test set
        target_defaults = max(100, int(test_size * 0.01))  # At least 100 defaults
        target_non_defaults = test_size - target_defaults
        
        if len(available_defaults) >= target_defaults and len(available_non_defaults) >= target_non_defaults:
            # Sample to achieve exact 1% default rate
            test_defaults = available_defaults.sample(n=target_defaults, random_state=42)
            test_non_defaults = available_non_defaults.sample(n=target_non_defaults, random_state=42)
            test_set = pd.concat([test_defaults, test_non_defaults], ignore_index=True)
            test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            print(f"Warning: Not enough samples for target distribution. Using available samples.")
            test_set = available_df.sample(n=min(test_size, len(available_df)), random_state=42)
    else:
        # For unscorable populations, sample preserving original distribution
        if len(available_df) >= test_size:
            test_set = available_df.sample(n=test_size, random_state=42)
        else:
            print(f"Warning: Only {len(available_df)} samples available, using all")
            test_set = available_df
    
    print(f"Test set created: {len(test_set)} samples")
    print(f"Test default rate: {test_set['TARGET'].mean():.3%}")
    
    return test_set

def define_feature_blocks():
    """
    Define feature blocks as per the study methodology
    
    Returns:
    - Dictionary of feature blocks for scorable and unscorable populations
    """
    feature_blocks = {
        'scorable': {
            'Bureau only': ['credit_score_quintile'],
            'Footprint only': [
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
            ],
            'Bureau + Footprint': [
                'credit_score_quintile',
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
            ],
            'Full': [
                'credit_score_quintile',
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error',
                'age', 'age_quintile', 'order_amount', 'order_amount_quintile', 
                'item_category', 'month', 'gender'
            ]
        },
        'unscorable': {
            'Footprint only': [
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
            ],
            'Full': [
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error',
                'age', 'age_quintile', 'order_amount', 'order_amount_quintile', 
                'item_category', 'month', 'gender'
            ]
        }
    }
    
    return feature_blocks

def preprocess_data_for_block(train_df, test_df, feature_list):
    """
    Preprocess training and test data for a specific feature block
    
    Returns:
    - X_train, y_train, X_test, y_test, feature_names, encoders
    """
    # Prepare target variables
    y_train = train_df['TARGET'].copy()
    y_test = test_df['TARGET'].copy()
    
    # Select only features available in the dataset
    available_features = []
    for feature in feature_list:
        if feature in train_df.columns:
            available_features.append(feature)
        else:
            print(f"Warning: Feature '{feature}' not found in dataset, skipping")
    
    if not available_features:
        raise ValueError("No features available for this block")
    
    print(f"Using features: {available_features}")
    
    # Prepare feature data
    X_train = train_df[available_features].copy()
    X_test = test_df[available_features].copy()
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        # Fit on combined data to handle unseen categories
        combined_data = pd.concat([X_train[col], X_test[col]], ignore_index=True)
        encoder.fit(combined_data.astype(str))
        
        X_train[col] = encoder.transform(X_train[col].astype(str))
        X_test[col] = encoder.transform(X_test[col].astype(str))
        encoders[col] = encoder
    
    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols:
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        encoders['scaler'] = scaler
    
    feature_names = X_train.columns.tolist()
    
    return X_train, y_train, X_test, y_test, feature_names, encoders

def train_models(X_train, y_train):
    """
    Train multiple models and return them
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    trained_models = {}
    
    print(f"\nTraining models on {len(X_train)} samples...")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"{name} - CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return trained_models

def evaluate_models(models, X_test, y_test, dataset_name, population_type, feature_block):
    """
    Evaluate models on test set and return results
    """
    results = []
    
    print(f"\nEvaluating models on {dataset_name} ({population_type}) - {feature_block}:")
    print(f"Test set size: {len(X_test)}")
    print(f"Test default rate: {y_test.mean():.3%}")
    print(f"Number of features: {X_test.shape[1]}")
    
    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        result = {
            'dataset': dataset_name,
            'population': population_type,
            'feature_block': feature_block,
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'test_size': len(X_test),
            'test_default_rate': y_test.mean(),
            'num_features': X_test.shape[1]
        }
        
        results.append(result)
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC: {auc:.3f}")
    
    return results

def run_modeling_pipeline():
    """
    Main modeling pipeline with feature blocks
    """
    print("="*80)
    print("BALANCED TRAINING WITH REALISTIC TESTING - FEATURE BLOCKS")
    print("="*80)
    
    # Load datasets and feature blocks
    datasets = load_datasets()
    feature_blocks = define_feature_blocks()
    
    if not datasets:
        print("No datasets found. Please run data generation scripts first.")
        return
    
    all_results = []
    
    # Process each dataset
    for dataset_name, df in datasets.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Determine population type
        population_type = 'scorable' if 'scorable' in dataset_name else 'unscorable'
        method = dataset_name.split('_')[-1]  # basic, copula, or ctgan
        
        try:
            # Create balanced training set (once per dataset)
            train_set, used_default_idx, used_non_default_idx = create_balanced_training_set(
                df, population_type=population_type, train_size=5000
            )
            
            # Create realistic test set (once per dataset)
            used_indices = list(used_default_idx) + list(used_non_default_idx)
            test_set = create_realistic_test_set(df, used_indices, population_type=population_type, test_size=10000)
            
            # Process each feature block for this dataset
            blocks_to_process = feature_blocks[population_type]
            
            for block_name, feature_list in blocks_to_process.items():
                print(f"\n{'-'*60}")
                print(f"FEATURE BLOCK: {block_name}")
                print(f"{'-'*60}")
                
                try:
                    # Preprocess data for this specific feature block
                    X_train, y_train, X_test, y_test, feature_names, encoders = preprocess_data_for_block(
                        train_set, test_set, feature_list
                    )
                    
                    # Train models for this feature block
                    trained_models = train_models(X_train, y_train)
                    
                    # Evaluate models for this feature block
                    results = evaluate_models(
                        trained_models, X_test, y_test, 
                        f"{population_type}_{method}", population_type, block_name
                    )
                    
                    all_results.extend(results)
                    
                except Exception as e:
                    print(f"Error processing feature block '{block_name}' for {dataset_name}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    # Save and display results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_dir = Path("FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/results")
        output_dir.mkdir(exist_ok=True)
        results_df.to_csv(output_dir / "balanced_modeling_results_feature_blocks.csv", index=False)
        
        # Display summary
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Pivot table for easy comparison by feature block
        pivot_auc = results_df.pivot_table(
            values='auc', 
            index=['population', 'feature_block', 'model'], 
            columns='dataset', 
            aggfunc='mean'
        )
        
        print("\nAUC Scores by Population, Feature Block, and Model:")
        print(pivot_auc.round(3))
        
        # Best performing models by population and feature block
        print("\nBest Performing Models by Population and Feature Block:")
        for pop in results_df['population'].unique():
            print(f"\n{pop.capitalize()} Population:")
            pop_results = results_df[results_df['population'] == pop]
            for block in pop_results['feature_block'].unique():
                block_results = pop_results[pop_results['feature_block'] == block]
                if not block_results.empty:
                    best_model = block_results.loc[block_results['auc'].idxmax()]
                    print(f"  {block}: {best_model['model']} on {best_model['dataset']} (AUC: {best_model['auc']:.3f})")
        
        # Summary by feature block across all datasets
        print(f"\nFeature Block Performance Summary:")
        block_summary = results_df.groupby(['population', 'feature_block', 'model'])['auc'].agg(['mean', 'std']).round(3)
        print(block_summary)
        
        print(f"\nDetailed results saved to: {output_dir / 'balanced_modeling_results_feature_blocks.csv'}")
        
    else:
        print("No results generated. Check for errors above.")

if __name__ == "__main__":
    run_modeling_pipeline() 