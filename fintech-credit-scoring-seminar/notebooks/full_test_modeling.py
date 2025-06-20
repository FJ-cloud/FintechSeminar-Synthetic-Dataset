#!/usr/bin/env python3
"""
Full Test Dataset Modeling Script

This script implements the balanced training/realistic testing methodology with dedicated test datasets:
- Training: Use ALL defaults from each dataset + fill up to 8,000 total samples
- Testing: Use complete test datasets (10,000 samples each)
- No data leakage between training and testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load training and test datasets separately"""
    data_dir = Path("../data")
    test_dir = Path("../data/test")
    
    training_datasets = {}
    test_datasets = {}
    
    # Training datasets
    training_files = {
        'scorable_basic': 'synthetic_digital_footprint_with_target.csv',
        'scorable_copula': 'synthetic_digital_footprint_copula.csv',
        'scorable_ctgan': 'synthetic_digital_footprint_ctgan.csv',
        'unscorable_basic': 'synthetic_digital_footprint_with_target_unscorable.csv',
        'unscorable_copula': 'synthetic_digital_footprint_copula_unscorable.csv',
        'unscorable_ctgan': 'synthetic_digital_footprint_ctgan_unscorable.csv'
    }
    
    # Test datasets
    test_files = {
        'scorable_basic': 'test_synthetic_digital_footprint_with_target.csv',
        'scorable_copula': 'test_synthetic_digital_footprint_copula.csv',
        'scorable_ctgan': 'test_synthetic_digital_footprint_ctgan.csv',
        'unscorable_basic': 'test_synthetic_digital_footprint_with_target_unscorable.csv',
        'unscorable_copula': 'test_synthetic_digital_footprint_copula_unscorable.csv',
        'unscorable_ctgan': 'test_synthetic_digital_footprint_ctgan_unscorable.csv'
    }
    
    print("Loading training datasets...")
    
    # Load training datasets
    for dataset_key, filename in training_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            training_datasets[dataset_key] = df
            print(f"Loaded training {filename}: {df.shape}, Default rate: {df['TARGET'].mean():.3%}")
        else:
            print(f"Warning: Training {filename} not found")
    
    print("\nLoading test datasets...")
    
    # Load test datasets
    for dataset_key, filename in test_files.items():
        filepath = test_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            test_datasets[dataset_key] = df
            print(f"Loaded test {filename}: {df.shape}, Default rate: {df['TARGET'].mean():.3%}")
        else:
            print(f"Warning: Test {filename} not found")
    
    return training_datasets, test_datasets

def create_balanced_training_set(df, train_size=5000):
    """
    Create balanced training set using ALL defaults + fill up to train_size total samples
    """
    defaults = df[df['TARGET'] == 1]
    non_defaults = df[df['TARGET'] == 0]
    
    print(f"Available defaults: {len(defaults)}")
    print(f"Available non-defaults: {len(non_defaults)}")
    
    # Use ALL defaults
    num_defaults = len(defaults)
    
    # Calculate how many non-defaults needed to reach train_size
    num_non_defaults_needed = train_size - num_defaults
    
    if num_non_defaults_needed > len(non_defaults):
        print(f"Warning: Not enough non-defaults. Using all {len(non_defaults)} available.")
        num_non_defaults_needed = len(non_defaults)
    
    # Sample non-defaults
    non_defaults_sample = non_defaults.sample(n=num_non_defaults_needed, random_state=42)
    
    # Combine
    training_set = pd.concat([defaults, non_defaults_sample], ignore_index=True)
    training_set = training_set.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    actual_default_rate = training_set['TARGET'].mean()
    
    print(f"Training set created:")
    print(f"  - Total samples: {len(training_set)}")
    print(f"  - Defaults: {training_set['TARGET'].sum()} ({actual_default_rate:.1%})")
    print(f"  - Non-defaults: {(training_set['TARGET'] == 0).sum()}")
    
    return training_set

def define_feature_blocks():
    """Define feature blocks for each population type"""
    
    # Bureau features (only available for scorable population)
    bureau_features = ['credit_score_quintile']
    
    # Digital footprint features
    footprint_features = [
        'device_type', 'os', 'email_host', 'channel', 'checkout_time',
        'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
    ]
    
    # Control features
    control_features = ['age', 'age_quintile', 'order_amount', 'order_amount_quintile', 
                       'item_category', 'month', 'gender']
    
    # Feature blocks for scorable population
    scorable_blocks = {
        'Bureau only': bureau_features,
        'Footprint only': footprint_features,
        'Bureau + Footprint': bureau_features + footprint_features,
        'Full': bureau_features + footprint_features + control_features
    }
    
    # Feature blocks for unscorable population
    unscorable_blocks = {
        'Footprint only': footprint_features,
        'Full': footprint_features + control_features
    }
    
    return scorable_blocks, unscorable_blocks

def preprocess_features(df, features):
    """Preprocess features for modeling"""
    df_processed = df[features + ['TARGET']].copy()
    
    # Handle categorical variables
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'TARGET' in categorical_features:
        categorical_features.remove('TARGET')
    
    # Label encode categorical features
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
        label_encoders[feature] = le
    
    # Separate features and target
    X = df_processed.drop('TARGET', axis=1)
    y = df_processed['TARGET']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled, y, label_encoders, scaler

def apply_preprocessing(df, features, label_encoders, scaler):
    """Apply existing preprocessing to new data"""
    df_processed = df[features + ['TARGET']].copy()
    
    # Apply label encoders
    for feature, le in label_encoders.items():
        if feature in df_processed.columns:
            # Handle unseen categories
            df_processed[feature] = df_processed[feature].astype(str)
            mask = df_processed[feature].isin(le.classes_)
            df_processed.loc[~mask, feature] = le.classes_[0]  # Replace unseen with first class
            df_processed[feature] = le.transform(df_processed[feature])
    
    # Separate features and target
    X = df_processed.drop('TARGET', axis=1)
    y = df_processed['TARGET']
    
    # Apply scaling
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled, y

def train_and_evaluate_models(X_train, y_train, X_test, y_test, dataset_name, block_name):
    """Train models and evaluate on test set"""
    
    results = {}
    
    # Models to train
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    print(f"\n--- {dataset_name} - {block_name} ---")
    print(f"Training set: {X_train.shape[0]} samples, {y_train.mean():.1%} default rate")
    print(f"Test set: {X_test.shape[0]} samples, {y_test.mean():.1%} default rate")
    
    for model_name, model in models.items():
        try:
            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            results[model_name] = {
                'cv_auc_mean': cv_mean,
                'cv_auc_std': cv_std,
                'test_auc': test_auc,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_default_rate': y_train.mean(),
                'test_default_rate': y_test.mean()
            }
            
            print(f"{model_name}:")
            print(f"  CV AUC: {cv_mean:.4f} (+/-{cv_std:.4f})")
            print(f"  Test AUC: {test_auc:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results

def run_modeling_analysis():
    """Run the complete modeling analysis"""
    
    print("="*80)
    print("FULL TEST DATASET MODELING ANALYSIS")
    print("="*80)
    
    # Load datasets
    training_datasets, test_datasets = load_datasets()
    
    # Define feature blocks
    scorable_blocks, unscorable_blocks = define_feature_blocks()
    
    # Store all results
    all_results = {}
    
    # Process each generation method
    generation_methods = ['basic', 'copula', 'ctgan']
    
    print(f"Comparing: {', '.join(generation_methods)} methods with proper train/test split.")
    
    for method in generation_methods:
        print(f"\n{'='*60}")
        print(f"PROCESSING {method.upper()} METHOD")
        print(f"{'='*60}")
        
        # Process scorable population
        if f'scorable_{method}' in training_datasets and f'scorable_{method}' in test_datasets:
            print(f"\n--- SCORABLE POPULATION ({method}) ---")
            
            train_df = training_datasets[f'scorable_{method}']
            test_df = test_datasets[f'scorable_{method}']
            
            # Create balanced training set
            train_balanced = create_balanced_training_set(train_df, train_size=8000)
            
            # Process each feature block
            for block_name, features in scorable_blocks.items():
                try:
                    # Check if all features exist
                    missing_features = [f for f in features if f not in train_balanced.columns]
                    if missing_features:
                        print(f"Skipping {block_name}: Missing features {missing_features}")
                        continue
                    
                    # Preprocess training data
                    X_train, y_train, label_encoders, scaler = preprocess_features(train_balanced, features)
                    
                    # Preprocess test data
                    X_test, y_test = apply_preprocessing(test_df, features, label_encoders, scaler)
                    
                    # Train and evaluate
                    results = train_and_evaluate_models(
                        X_train, y_train, X_test, y_test, 
                        f'Scorable {method}', block_name
                    )
                    
                    # Store results
                    key = f'scorable_{method}_{block_name.replace(" ", "_").replace("+", "plus").lower()}'
                    all_results[key] = results
                    
                except Exception as e:
                    print(f"Error processing scorable {method} - {block_name}: {str(e)}")
        
        # Process unscorable population
        if f'unscorable_{method}' in training_datasets and f'unscorable_{method}' in test_datasets:
            print(f"\n--- UNSCORABLE POPULATION ({method}) ---")
            
            train_df = training_datasets[f'unscorable_{method}']
            test_df = test_datasets[f'unscorable_{method}']
            
            # Create balanced training set
            train_balanced = create_balanced_training_set(train_df, train_size=8000)
            
            # Process each feature block
            for block_name, features in unscorable_blocks.items():
                try:
                    # Check if all features exist
                    missing_features = [f for f in features if f not in train_balanced.columns]
                    if missing_features:
                        print(f"Skipping {block_name}: Missing features {missing_features}")
                        continue
                    
                    # Preprocess training data
                    X_train, y_train, label_encoders, scaler = preprocess_features(train_balanced, features)
                    
                    # Preprocess test data
                    X_test, y_test = apply_preprocessing(test_df, features, label_encoders, scaler)
                    
                    # Train and evaluate
                    results = train_and_evaluate_models(
                        X_train, y_train, X_test, y_test, 
                        f'Unscorable {method}', block_name
                    )
                    
                    # Store results
                    key = f'unscorable_{method}_{block_name.replace(" ", "_").lower()}'
                    all_results[key] = results
                    
                except Exception as e:
                    print(f"Error processing unscorable {method} - {block_name}: {str(e)}")
    
    # Create summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    summary_data = []
    for key, methods in all_results.items():
        for model_name, metrics in methods.items():
            if 'error' not in metrics:
                summary_data.append({
                    'Dataset': key,
                    'Model': model_name,
                    'CV_AUC': metrics['cv_auc_mean'],
                    'CV_AUC_Std': metrics['cv_auc_std'],
                    'Test_AUC': metrics['test_auc'],
                    'Train_Samples': metrics['train_samples'],
                    'Test_Samples': metrics['test_samples'],
                    'Train_Default_Rate': metrics['train_default_rate'],
                    'Test_Default_Rate': metrics['test_default_rate']
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save results
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        summary_df.to_csv(results_dir / "full_test_modeling_results.csv", index=False)
        print(f"Results saved to: {results_dir / 'full_test_modeling_results.csv'}")
        
        # Display summary
        print("\nTop performing models by Test AUC:")
        top_models = summary_df.nlargest(10, 'Test_AUC')[['Dataset', 'Model', 'Test_AUC', 'CV_AUC']]
        print(top_models.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("MODELING ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_modeling_analysis() 