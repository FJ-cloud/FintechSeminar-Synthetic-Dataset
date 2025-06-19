#!/usr/bin/env python3
"""
Comprehensive Modeling Pipeline for Synthetic Credit Scoring Data

This script performs the complete modeling workflow:
1. Load synthetic datasets
2. Define feature combinations (digital footprint, credit score, controls, all)
3. Preprocess data (encoding, scaling)  
4. Fit multiple models (Logistic Regression, Random Forest)
5. Evaluate performance (AUC, pseudo-R², OOB scores)
6. Generate visualizations and summary tables

Author: Generated for Fintech Seminar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score,
    classification_report, precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Imbalanced learning imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CreditScoringModeler:
    """
    Comprehensive modeling pipeline for credit scoring analysis
    """
    
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Define feature sets
        self.feature_sets = {
            'digital_footprint': [
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
            ],
            'credit_score': [
                'credit_score_quintile'
            ],
            'controls': [
                'age', 'gender', 'order_amount', 'item_category', 'month',
                'age_quintile', 'order_amount_quintile'
            ],
            'all_predictors': []  # Will be populated with all features
        }
        
        # Results storage
        self.results = []
        self.roc_data = {}
        self.feature_importance = {}
        self.class_balance_info = {}
        
    def load_datasets(self):
        """Load all available synthetic datasets"""
        datasets = {}
        
        # Define expected dataset files
        dataset_files = {
            'synthetic_basic_unscorable': 'synthetic_digital_footprint_with_target_unscorable.csv',
            'synthetic_copula_unscorable': 'synthetic_digital_footprint_copula_unscrorable.csv',
            'synthetic_ctgan_unscorable': 'synthetic_digital_footprint_ctgan_unscorable.csv',
            'synthetic_basic_scorable': 'synthetic_digital_footprint_with_target.csv',
            'synthetic_copula_scorable': 'synthetic_digital_footprint_copula.csv',
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
    
    def define_feature_sets(self, df):
        """Define and validate feature sets based on available columns"""
        available_columns = set(df.columns) - {'TARGET'}
        
        # Update feature sets based on available columns
        updated_sets = {}
        for set_name, features in self.feature_sets.items():
            if set_name == 'all_predictors':
                # All available features except target
                updated_sets[set_name] = list(available_columns)
            else:
                # Only include features that exist in the dataset
                updated_sets[set_name] = [f for f in features if f in available_columns]
        
        # Remove empty feature sets
        updated_sets = {k: v for k, v in updated_sets.items() if v}
        
        print(f"\nFeature sets defined:")
        for name, features in updated_sets.items():
            print(f"  {name}: {len(features)} features")
            if len(features) <= 10:
                print(f"    {features}")
        
        return updated_sets
    
    def analyze_class_balance(self, y, dataset_name):
        """Analyze class imbalance in the target variable"""
        class_counts = Counter(y)
        total = len(y)
        
        balance_info = {
            'dataset': dataset_name,
            'n_total': total,
            'n_defaults': class_counts[1],
            'n_non_defaults': class_counts[0],
            'default_rate': class_counts[1] / total,
            'imbalance_ratio': class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
        }
        
        print(f"  Class balance analysis:")
        print(f"    Total samples: {total:,}")
        print(f"    Defaults (1): {class_counts[1]:,} ({balance_info['default_rate']:.1%})")
        print(f"    Non-defaults (0): {class_counts[0]:,} ({(1-balance_info['default_rate']):.1%})")
        print(f"    Imbalance ratio (majority:minority): {balance_info['imbalance_ratio']:.1f}:1")
        
        self.class_balance_info[dataset_name] = balance_info
        return balance_info
    
    def create_preprocessor(self, X, feature_set_name):
        """Create preprocessing pipeline for the given features"""
        
        # Identify categorical and numerical columns
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                # Check if numerical feature needs scaling (range > 400)
                if X[col].max() - X[col].min() > 400:
                    numerical_features.append(col)
        
        # Create preprocessing steps
        preprocessor_steps = []
        
        # One-hot encode categorical features
        if categorical_features:
            preprocessor_steps.append(
                ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            )
        
        # Standard scale numerical features with large ranges
        if numerical_features:
            preprocessor_steps.append(
                ('numerical', StandardScaler(), numerical_features)
            )
        
        # Create column transformer
        if preprocessor_steps:
            preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='passthrough'  # Keep other features as-is
            )
        else:
            # No preprocessing needed
            preprocessor = 'passthrough'
        
        return preprocessor
    
    def compute_pseudo_r2(self, y_true, y_prob):
        """Compute McFadden's Pseudo R-squared"""
        # Likelihood of the fitted model
        ll_model = np.sum(y_true * np.log(np.clip(y_prob, 1e-15, 1-1e-15)) + 
                         (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-15, 1-1e-15)))
        
        # Likelihood of the null model (intercept only)
        p0 = np.mean(y_true)
        ll_null = np.sum(y_true * np.log(p0) + (1 - y_true) * np.log(1 - p0))
        
        # McFadden's R²
        pseudo_r2 = 1 - (ll_model / ll_null)
        return pseudo_r2
    
    def fit_logistic_regression(self, X_train, X_test, y_train, y_test, preprocessor, balance_method='none'):
        """Fit Logistic Regression model with various balancing strategies"""
        
        results = {}
        
        if balance_method == 'balanced_weights':
            # Use balanced class weights
            lr_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    penalty='l2', random_state=42, max_iter=1000, class_weight='balanced'
                ))
            ])
            
        elif balance_method == 'smote':
            # Use SMOTE oversampling
            lr_pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(penalty='l2', random_state=42, max_iter=1000))
            ])
            
        else:  # 'none'
            # Standard logistic regression
            lr_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(penalty='l2', random_state=42, max_iter=1000))
            ])
        
        # Fit model
        lr_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr_pipeline.predict(X_test)
        y_prob = lr_pipeline.predict_proba(X_test)[:, 1]
        
        # Comprehensive metrics
        auc_score = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pseudo_r2 = self.compute_pseudo_r2(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': lr_pipeline,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'auc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pseudo_r2': pseudo_r2,
            'confusion_matrix': cm,
            'roc_data': (fpr, tpr),
            'model_type': 'Logistic Regression',
            'balance_method': balance_method
        }
    
    def fit_random_forest(self, X_train, X_test, y_train, y_test, preprocessor, balance_method='none'):
        """Fit Random Forest with hyperparameter tuning and balancing strategies"""
        
        if balance_method == 'balanced_weights':
            # Use balanced class weights
            param_grid = {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [None, 5, 10],
                'classifier__max_features': ['sqrt', 'log2']
            }
            
            rf_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    random_state=42, oob_score=True, class_weight='balanced'
                ))
            ])
            
        elif balance_method == 'smote':
            # Use SMOTE oversampling
            param_grid = {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [None, 5, 10],
                'classifier__max_features': ['sqrt', 'log2']
            }
            
            rf_pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(random_state=42, oob_score=True))
            ])
            
        else:  # 'none'
            # Standard Random Forest
            param_grid = {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [None, 5, 10],
                'classifier__max_features': ['sqrt', 'log2']
            }
            
            rf_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42, oob_score=True))
            ])
        
        # Grid search
        grid_search = GridSearchCV(
            rf_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Comprehensive metrics
        auc_score = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Handle OOB score (may not be available with SMOTE)
        try:
            oob_score = best_model.named_steps['classifier'].oob_score_
        except:
            oob_score = np.nan
            
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'auc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'oob_score': oob_score,
            'confusion_matrix': cm,
            'roc_data': (fpr, tpr),
            'model_type': 'Random Forest',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_
        }
    
    def extract_feature_importance(self, model, model_type, feature_names, dataset_name, feature_set):
        """Extract feature importance from fitted models"""
        
        if model_type == 'Logistic Regression':
            # Get coefficients (log odds ratios)
            classifier = model.named_steps['classifier']
            
            # Handle preprocessing to get feature names
            if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names_out = model.named_steps['preprocessor'].get_feature_names_out()
            else:
                feature_names_out = feature_names  # Fallback
            
            coefficients = classifier.coef_[0]
            
            importance_df = pd.DataFrame({
                'feature': feature_names_out[:len(coefficients)],
                'coefficient': coefficients,
                'odds_ratio': np.exp(coefficients),
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
        elif model_type == 'Random Forest':
            # Get feature importance from random forest
            classifier = model.named_steps['classifier']
            
            # Handle preprocessing to get feature names
            if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names_out = model.named_steps['preprocessor'].get_feature_names_out()
            else:
                feature_names_out = feature_names  # Fallback
            
            importance_scores = classifier.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names_out[:len(importance_scores)],
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        # Store feature importance
        key = f"{dataset_name}_{feature_set}_{model_type}"
        self.feature_importance[key] = importance_df
        
        return importance_df
    
    def run_modeling_pipeline(self, dataset_name, df, feature_sets):
        """Run complete modeling pipeline for a single dataset"""
        
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        if 'TARGET' not in df.columns:
            print(f"No TARGET column found in {dataset_name}, skipping...")
            return
        
        # Analyze class balance
        balance_info = self.analyze_class_balance(df['TARGET'], dataset_name)
        
        for feature_set_name, features in feature_sets.items():
            print(f"\nFeature set: {feature_set_name} ({len(features)} features)")
            
            # Prepare data
            X = df[features].copy()
            y = df['TARGET'].copy()
            
            # Skip if no features
            if X.empty:
                print(f"No features available for {feature_set_name}, skipping...")
                continue
            
            # Train-test split (stratified to maintain class distribution)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(X_train, feature_set_name)
            
            # Test different balancing strategies
            balance_methods = ['none', 'balanced_weights', 'smote']
            
            for balance_method in balance_methods:
                print(f"  Testing balance method: {balance_method}")
                
                # Fit Logistic Regression
                try:
                    lr_results = self.fit_logistic_regression(
                        X_train, X_test, y_train, y_test, preprocessor, balance_method
                    )
                    
                    # Store results
                    self.results.append({
                        'dataset': dataset_name,
                        'feature_set': feature_set_name,
                        'model': f'Logistic Regression ({balance_method})',
                        'balance_method': balance_method,
                        'auc': lr_results['auc'],
                        'accuracy': lr_results['accuracy'],
                        'precision': lr_results['precision'],
                        'recall': lr_results['recall'],
                        'f1_score': lr_results['f1_score'],
                        'pseudo_r2_oob': lr_results['pseudo_r2'],
                        'n_features': len(features),
                        'imbalance_ratio': balance_info['imbalance_ratio']
                    })
                    
                    # Store ROC data
                    roc_key = f"{dataset_name}_{feature_set_name}_LR_{balance_method}"
                    self.roc_data[roc_key] = lr_results['roc_data']
                    
                    # Extract feature importance (only for baseline)
                    if balance_method == 'none':
                        self.extract_feature_importance(
                            lr_results['model'], 'Logistic Regression', 
                            features, dataset_name, feature_set_name
                        )
                    
                    print(f"    LR ({balance_method}): AUC={lr_results['auc']:.4f}, "
                          f"F1={lr_results['f1_score']:.4f}, Recall={lr_results['recall']:.4f}")
                    
                except Exception as e:
                    print(f"    Error fitting LR ({balance_method}): {e}")
                
                # Fit Random Forest
                try:
                    rf_results = self.fit_random_forest(
                        X_train, X_test, y_train, y_test, preprocessor, balance_method
                    )
                    
                    # Store results
                    self.results.append({
                        'dataset': dataset_name,
                        'feature_set': feature_set_name,
                        'model': f'Random Forest ({balance_method})',
                        'balance_method': balance_method,
                        'auc': rf_results['auc'],
                        'accuracy': rf_results['accuracy'],
                        'precision': rf_results['precision'],
                        'recall': rf_results['recall'],
                        'f1_score': rf_results['f1_score'],
                        'pseudo_r2_oob': rf_results['oob_score'],
                        'n_features': len(features),
                        'imbalance_ratio': balance_info['imbalance_ratio']
                    })
                    
                    # Store ROC data
                    roc_key = f"{dataset_name}_{feature_set_name}_RF_{balance_method}"
                    self.roc_data[roc_key] = rf_results['roc_data']
                    
                    # Extract feature importance (only for baseline)
                    if balance_method == 'none':
                        self.extract_feature_importance(
                            rf_results['model'], 'Random Forest',
                            features, dataset_name, feature_set_name
                        )
                    
                    oob_str = f", OOB={rf_results['oob_score']:.4f}" if not np.isnan(rf_results['oob_score']) else ""
                    print(f"    RF ({balance_method}): AUC={rf_results['auc']:.4f}, "
                          f"F1={rf_results['f1_score']:.4f}, Recall={rf_results['recall']:.4f}{oob_str}")
                    
                except Exception as e:
                    print(f"    Error fitting RF ({balance_method}): {e}")
    
    def create_roc_plots(self):
        """Create ROC curve plots for each dataset"""
        
        # Group by dataset
        datasets = set([key.split('_')[0] + '_' + key.split('_')[1] + '_' + key.split('_')[2] 
                       for key in self.roc_data.keys()])
        
        for dataset in datasets:
            # Create subplot for this dataset
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'ROC Curves - {dataset}', fontsize=16)
            
            feature_sets = ['digital_footprint', 'credit_score', 'controls', 'all_predictors']
            
            for idx, feature_set in enumerate(feature_sets):
                ax = axes[idx // 2, idx % 2]
                
                # Plot LR ROC curve
                lr_key = f"{dataset}_{feature_set}_LR"
                if lr_key in self.roc_data:
                    fpr, tpr = self.roc_data[lr_key]
                    auc_score = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.3f})', linewidth=2)
                
                # Plot RF ROC curve
                rf_key = f"{dataset}_{feature_set}_RF"
                if rf_key in self.roc_data:
                    fpr, tpr = self.roc_data[rf_key]
                    auc_score = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.3f})', linewidth=2)
                
                # Plot diagonal line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{feature_set.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'roc_curves_{dataset}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_performance_charts(self):
        """Create bar charts for AUC and Pseudo-R²/OOB scores"""
        
        if not self.results:
            print("No results to plot")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # AUC comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # AUC by model and feature set
        pivot_auc = results_df.pivot_table(
            values='auc', index=['dataset', 'feature_set'], columns='model', aggfunc='mean'
        )
        
        pivot_auc.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('AUC by Model and Feature Set', fontsize=14)
        axes[0].set_ylabel('AUC Score')
        axes[0].legend(title='Model')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Pseudo-R² / OOB Score comparison
        pivot_score = results_df.pivot_table(
            values='pseudo_r2_oob', index=['dataset', 'feature_set'], columns='model', aggfunc='mean'
        )
        
        pivot_score.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('Pseudo-R² (LR) / OOB Score (RF) by Model and Feature Set', fontsize=14)
        axes[1].set_ylabel('Score')
        axes[1].legend(title='Model')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_plots(self):
        """Create feature importance comparison plots"""
        
        if not self.feature_importance:
            print("No feature importance data to plot")
            return
        
        # Create plots for each dataset and feature set combination
        for key, importance_df in self.feature_importance.items():
            
            if len(importance_df) == 0:
                continue
                
            dataset, feature_set, model_type = key.split('_', 2)
            
            # Limit to top 15 features
            top_features = importance_df.head(15)
            
            plt.figure(figsize=(12, 8))
            
            if model_type == 'Logistic Regression':
                # Plot odds ratios
                y_pos = np.arange(len(top_features))
                plt.barh(y_pos, top_features['odds_ratio'])
                plt.yticks(y_pos, top_features['feature'])
                plt.xlabel('Odds Ratio')
                plt.title(f'Top 15 Feature Odds Ratios - {model_type}\n'
                         f'{dataset} - {feature_set}')
                plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
                
            else:  # Random Forest
                # Plot feature importance
                y_pos = np.arange(len(top_features))
                plt.barh(y_pos, top_features['importance'])
                plt.yticks(y_pos, top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 15 Feature Importance - {model_type}\n'
                         f'{dataset} - {feature_set}')
            
            plt.tight_layout()
            filename = f'feature_importance_{key}.png'
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_complete_analysis(self):
        """Run the complete modeling analysis"""
        
        print("="*80)
        print("COMPREHENSIVE CREDIT SCORING MODEL ANALYSIS")
        print("="*80)
        
        # Load datasets
        datasets = self.load_datasets()
        
        if not datasets:
            print("No datasets found!")
            return None
        
        # Process each dataset
        for dataset_name, df in datasets.items():
            # Define feature sets for this dataset
            feature_sets = self.define_feature_sets(df)
            
            # Run modeling pipeline
            self.run_modeling_pipeline(dataset_name, df, feature_sets)
        
        # Create summary table
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            print("\n" + "="*80)
            print("MASTER RESULTS TABLE")
            print("="*80)
            
            # Display results
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', '{:.4f}'.format)
            
            print(results_df.to_string(index=False))
            
            # Save results
            results_df.to_csv(self.results_dir / 'modeling_results.csv', index=False)
            print(f"\n✓ Results saved to: {self.results_dir / 'modeling_results.csv'}")
            
            # Create visualizations
            print("\nGenerating visualizations...")
            self.create_roc_plots()
            self.create_performance_charts()
            self.create_feature_importance_plots()
            
            print(f"✓ Visualizations saved to: {self.results_dir}")
            
            return results_df
        
        else:
            print("No results generated!")
            return None

def main():
    """Main execution function"""
    
    # Initialize modeler
    modeler = CreditScoringModeler(data_dir="data", results_dir="results")
    
    # Run complete analysis
    results = modeler.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    final_results = main() 