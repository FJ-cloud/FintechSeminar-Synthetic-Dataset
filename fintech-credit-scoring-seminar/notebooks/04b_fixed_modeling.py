#!/usr/bin/env python3
"""
Fixed Enhanced Modeling Pipeline - No GUI Backend Issues

This script fixes the matplotlib backend issues and provides a robust
modeling pipeline for credit scoring analysis.
"""

import numpy as np
import pandas as pd

# Fix matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core sklearn imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score,
    precision_recall_curve, f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced models (with error handling)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from category_encoders import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False

from collections import Counter

# Set plotting style
plt.style.use('default')  # Use default instead of seaborn to avoid issues
sns.set_palette("husl")

class FixedCreditScoringModeler:
    """
    Fixed comprehensive modeling pipeline without GUI backend issues
    """
    
    def __init__(self, data_dir="../data", results_dir="../results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Feature sets
        self.feature_sets = {
            'digital_footprint': [
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
            ],
            'credit_score': ['credit_score_quintile'],
            'controls': [
                'age', 'gender', 'order_amount', 'item_category', 'month',
                'age_quintile', 'order_amount_quintile'
            ],
            'all_predictors': []
        }
        
        # Results storage
        self.results = []
        self.roc_data = {}
        self.class_balance_info = {}
        
    def load_datasets(self):
        """Load datasets with quality assessment"""
        datasets = {}
        
        dataset_files = {
            'synthetic_basic_unscorable': 'synthetic_digital_footprint_with_target_unscorable.csv',
            'synthetic_copula_unscorable': 'synthetic_digital_footprint_copula_unscrorable.csv',
            'synthetic_ctgan_unscorable': 'synthetic_digital_footprint_ctgan_unscorable.csv',
            'synthetic_basic_scorable': 'synthetic_digital_footprint_with_target.csv',
            'synthetic_copula_scorable': 'synthetic_digital_footprint_copula.csv',
            'synthetic_ctgan_scorable': 'synthetic_digital_footprint_ctgan.csv',
        }
        
        for name, filename in dataset_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    
                    # Basic quality check
                    print(f"📊 {name}: Shape {df.shape}, Missing: {df.isnull().sum().sum()}")
                    
                    # Handle missing values
                    if df.isnull().sum().sum() > 0:
                        print("   🔧 Imputing missing values...")
                        # Simple imputation
                        num_cols = df.select_dtypes(include=['number']).columns
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns
                        
                        for col in num_cols:
                            if df[col].isnull().sum() > 0:
                                df[col].fillna(df[col].median(), inplace=True)
                        
                        for col in cat_cols:
                            if df[col].isnull().sum() > 0:
                                df[col].fillna(df[col].mode().iloc[0], inplace=True)
                    
                    # Check target distribution
                    if 'TARGET' in df.columns:
                        target_dist = df['TARGET'].value_counts()
                        print(f"   Target: {target_dist.to_dict()}")
                    
                    datasets[name] = df
                    print(f"✅ Loaded {name}")
                    
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
        
        return datasets
    
    def analyze_class_balance(self, y, dataset_name):
        """Analyze class imbalance"""
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
        
        print(f"  📈 Class balance: {class_counts[1]:,} defaults ({balance_info['default_rate']:.1%})")
        print(f"      Imbalance ratio: {balance_info['imbalance_ratio']:.1f}:1")
        
        self.class_balance_info[dataset_name] = balance_info
        return balance_info
    
    def create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Create preprocessing steps
        preprocessor_steps = []
        
        if categorical_features:
            preprocessor_steps.append(
                ('categorical', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features)
            )
        
        if numerical_features:
            preprocessor_steps.append(
                ('numerical', StandardScaler(), numerical_features)
            )
        
        if preprocessor_steps:
            preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='passthrough'
            )
        else:
            preprocessor = 'passthrough'
        
        return preprocessor
    
    def compute_optimal_threshold(self, y_true, y_prob):
        """Compute optimal F1 threshold"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        return optimal_threshold
    
    def fit_logistic_regression(self, X_train, X_test, y_train, y_test, preprocessor, balance_method):
        """Fit Logistic Regression with balancing"""
        
        # Configure balancing
        if balance_method == 'balanced_weights':
            class_weight = 'balanced'
            resampler = None
        elif balance_method == 'smote':
            class_weight = None
            resampler = SMOTE(random_state=42)
        elif balance_method == 'adasyn':
            class_weight = None
            resampler = ADASYN(random_state=42)
        elif balance_method == 'custom_weights':
            imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
            class_weight = {0: 1, 1: min(imbalance_ratio / 2, 10)}
            resampler = None
        else:  # 'none'
            class_weight = None
            resampler = None
        
        # Build pipeline
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if resampler is not None:
            pipeline_steps.append(('resampler', resampler))
        
        pipeline_steps.append(('classifier', LogisticRegression(
            random_state=42, max_iter=1000, class_weight=class_weight
        )))
        
        # Create pipeline
        if resampler is not None:
            lr_pipeline = ImbPipeline(pipeline_steps)
        else:
            lr_pipeline = Pipeline(pipeline_steps)
        
        # Simple hyperparameter tuning
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
        
        grid_search = RandomizedSearchCV(
            lr_pipeline, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, n_iter=6, random_state=42
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_prob = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.compute_optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        # Metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob, optimal_threshold)
        
        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'roc_data': (fpr, tpr),
            'model_type': 'Logistic Regression',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_,
            **metrics
        }
    
    def fit_random_forest(self, X_train, X_test, y_train, y_test, preprocessor, balance_method):
        """Fit Random Forest with balancing"""
        
        # Configure balancing
        if balance_method == 'balanced_weights':
            class_weight = 'balanced'
            resampler = None
        elif balance_method == 'smote':
            class_weight = None
            resampler = SMOTE(random_state=42)
        elif balance_method == 'adasyn':
            class_weight = None
            resampler = ADASYN(random_state=42)
        elif balance_method == 'custom_weights':
            imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
            class_weight = {0: 1, 1: min(imbalance_ratio / 2, 10)}
            resampler = None
        else:
            class_weight = None
            resampler = None
        
        # Build pipeline
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if resampler is not None:
            pipeline_steps.append(('resampler', resampler))
        
        pipeline_steps.append(('classifier', RandomForestClassifier(
            random_state=42, n_jobs=-1, oob_score=True, class_weight=class_weight
        )))
        
        if resampler is not None:
            rf_pipeline = ImbPipeline(pipeline_steps)
        else:
            rf_pipeline = Pipeline(pipeline_steps)
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__max_features': ['sqrt', 'log2']
        }
        
        grid_search = RandomizedSearchCV(
            rf_pipeline, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, n_iter=10, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_prob = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.compute_optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        # Metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob, optimal_threshold)
        
        # OOB score
        try:
            oob_score = best_model.named_steps['classifier'].oob_score_
        except:
            oob_score = np.nan
        
        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'oob_score': oob_score,
            'roc_data': (fpr, tpr),
            'model_type': 'Random Forest',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_,
            **metrics
        }
    
    def fit_xgboost(self, X_train, X_test, y_train, y_test, preprocessor, balance_method):
        """Fit XGBoost with balancing"""
        
        if not XGBOOST_AVAILABLE:
            return None
        
        # Configure balancing
        if balance_method == 'balanced_weights':
            scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
            resampler = None
        elif balance_method in ['smote', 'adasyn']:
            scale_pos_weight = 1
            resampler = SMOTE(random_state=42) if balance_method == 'smote' else ADASYN(random_state=42)
        elif balance_method == 'custom_weights':
            scale_pos_weight = min(sum(y_train == 0) / sum(y_train == 1) / 2, 10)
            resampler = None
        else:
            scale_pos_weight = 1
            resampler = None
        
        # Build pipeline
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if resampler is not None:
            pipeline_steps.append(('resampler', resampler))
        
        pipeline_steps.append(('classifier', XGBClassifier(
            random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )))
        
        if resampler is not None:
            xgb_pipeline = ImbPipeline(pipeline_steps)
        else:
            xgb_pipeline = Pipeline(pipeline_steps)
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.1, 0.2]
        }
        
        grid_search = RandomizedSearchCV(
            xgb_pipeline, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, n_iter=8, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_prob = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.compute_optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        # Metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob, optimal_threshold)
        
        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'roc_data': (fpr, tpr),
            'model_type': 'XGBoost',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_,
            **metrics
        }
    
    def compute_metrics(self, y_true, y_pred, y_prob, optimal_threshold):
        """Compute comprehensive metrics"""
        
        auc_score = roc_auc_score(y_true, y_prob)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        avg_precision = average_precision_score(y_true, y_prob)
        
        return {
            'auc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_precision': avg_precision,
            'optimal_threshold': optimal_threshold
        }
    
    def perform_cross_validation(self, X, y, model_pipeline):
        """Perform cross-validation"""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for speed
        
        try:
            auc_scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=1)  # Single job to avoid issues
            f1_scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='f1', n_jobs=1)
            
            return {
                'cv_auc_mean': np.mean(auc_scores),
                'cv_auc_std': np.std(auc_scores),
                'cv_f1_mean': np.mean(f1_scores),
                'cv_f1_std': np.std(f1_scores)
            }
        except Exception as e:
            print(f"      ⚠️ CV failed: {e}")
            return {
                'cv_auc_mean': np.nan,
                'cv_auc_std': np.nan,
                'cv_f1_mean': np.nan,
                'cv_f1_std': np.nan
            }
    
    def create_safe_plot(self, plot_func, filename, *args, **kwargs):
        """Safely create plots without GUI issues"""
        try:
            # Ensure we use Agg backend
            plt.switch_backend('Agg')
            
            # Create the plot
            plot_func(*args, **kwargs)
            
            # Save and close
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
            plt.close('all')  # Close all figures
            
        except Exception as e:
            print(f"      ⚠️ Plot error ({filename}): {e}")
            plt.close('all')  # Ensure cleanup
    
    def plot_confusion_matrix(self, y_true, y_pred, title, filename):
        """Plot confusion matrix"""
        def _plot():
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Non-Default', 'Default'],
                       yticklabels=['Non-Default', 'Default'])
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        self.create_safe_plot(_plot, filename)
    
    def run_modeling_pipeline(self, dataset_name, df, feature_sets):
        """Run modeling pipeline for a dataset"""
        
        print(f"\n{'='*60}")
        print(f"🚀 Processing: {dataset_name}")
        print(f"{'='*60}")
        
        if 'TARGET' not in df.columns:
            print(f"No TARGET column found, skipping...")
            return
        
        # Analyze class balance
        balance_info = self.analyze_class_balance(df['TARGET'], dataset_name)
        
        for feature_set_name, features in feature_sets.items():
            print(f"\n📋 Feature set: {feature_set_name} ({len(features)} features)")
            
            # Prepare data
            X = df[features].copy()
            y = df['TARGET'].copy()
            
            if X.empty:
                print(f"No features available, skipping...")
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(X_train)
            
            # Test balance methods
            balance_methods = ['none', 'balanced_weights', 'smote', 'custom_weights']
            
            # Model configurations
            model_configs = [
                ('logistic_regression', self.fit_logistic_regression),
                ('random_forest', self.fit_random_forest),
            ]
            
            if XGBOOST_AVAILABLE:
                model_configs.append(('xgboost', self.fit_xgboost))
            
            for model_name, model_func in model_configs:
                print(f"\n  🤖 {model_name.replace('_', ' ').title()}")
                
                for balance_method in balance_methods:
                    print(f"    ⚖️ {balance_method}")
                    
                    try:
                        # Fit model
                        results = model_func(X_train, X_test, y_train, y_test, preprocessor, balance_method)
                        
                        if results is None:
                            continue
                        
                        # Cross-validation
                        cv_results = self.perform_cross_validation(X, y, results['model'])
                        
                        # Store results
                        result_entry = {
                            'dataset': dataset_name,
                            'feature_set': feature_set_name,
                            'model': f"{results['model_type']} ({balance_method})",
                            'balance_method': balance_method,
                            'n_features': len(features),
                            'imbalance_ratio': balance_info['imbalance_ratio'],
                            **{k: v for k, v in results.items() if k not in ['model', 'roc_data']},
                            **cv_results
                        }
                        
                        self.results.append(result_entry)
                        
                        # Store ROC data
                        roc_key = f"{dataset_name}_{feature_set_name}_{model_name}_{balance_method}"
                        self.roc_data[roc_key] = results['roc_data']
                        
                        # Create confusion matrix plot
                        cm_title = f"{results['model_type']} ({balance_method})\n{dataset_name} - {feature_set_name}"
                        cm_filename = f"cm_{dataset_name}_{feature_set_name}_{model_name}_{balance_method}.png"
                        
                        y_pred = (results['model'].predict_proba(X_test)[:, 1] >= results['optimal_threshold']).astype(int)
                        self.plot_confusion_matrix(y_test, y_pred, cm_title, cm_filename)
                        
                        # Print summary
                        print(f"      📊 AUC: {results['auc']:.4f}, F1: {results['f1_score']:.4f}, Recall: {results['recall']:.4f}")
                        
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
    
    def define_feature_sets(self, df):
        """Define feature sets based on available columns"""
        available_columns = set(df.columns) - {'TARGET'}
        
        updated_sets = {}
        for set_name, features in self.feature_sets.items():
            if set_name == 'all_predictors':
                updated_sets[set_name] = list(available_columns)
            else:
                updated_sets[set_name] = [f for f in features if f in available_columns]
        
        updated_sets = {k: v for k, v in updated_sets.items() if v}
        
        print(f"\n📋 Feature sets:")
        for name, features in updated_sets.items():
            print(f"  {name}: {len(features)} features")
        
        return updated_sets
    
    def create_performance_summary(self, results_df):
        """Create performance summary plots"""
        
        def _plot_performance():
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance Summary', fontsize=16)
            
            # AUC by balance method
            balance_summary = results_df.groupby(['balance_method', 'model']).agg({
                'auc': 'mean', 'f1_score': 'mean', 'recall': 'mean'
            }).reset_index()
            
            # Extract model type (remove balance method from model name)
            balance_summary['model_type'] = balance_summary['model'].str.extract(r'([^(]+)')
            
            # AUC comparison
            auc_pivot = balance_summary.pivot(index='balance_method', columns='model_type', values='auc')
            auc_pivot.plot(kind='bar', ax=axes[0,0], rot=45)
            axes[0,0].set_title('AUC by Balance Method')
            axes[0,0].set_ylabel('AUC Score')
            axes[0,0].legend(title='Model Type')
            
            # F1 comparison
            f1_pivot = balance_summary.pivot(index='balance_method', columns='model_type', values='f1_score')
            f1_pivot.plot(kind='bar', ax=axes[0,1], rot=45)
            axes[0,1].set_title('F1-Score by Balance Method')
            axes[0,1].set_ylabel('F1 Score')
            axes[0,1].legend(title='Model Type')
            
            # Recall comparison
            recall_pivot = balance_summary.pivot(index='balance_method', columns='model_type', values='recall')
            recall_pivot.plot(kind='bar', ax=axes[1,0], rot=45)
            axes[1,0].set_title('Recall by Balance Method')
            axes[1,0].set_ylabel('Recall')
            axes[1,0].legend(title='Model Type')
            
            # Best performance by feature set
            best_by_feature = results_df.loc[results_df.groupby('feature_set')['f1_score'].idxmax()]
            best_by_feature.plot(x='feature_set', y='f1_score', kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Best F1-Score by Feature Set')
            axes[1,1].set_ylabel('F1 Score')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
        
        self.create_safe_plot(_plot_performance, 'performance_summary.png')
    
    def run_complete_analysis(self):
        """Run complete analysis"""
        
        print("="*80)
        print("🚀 FIXED COMPREHENSIVE CREDIT SCORING ANALYSIS")
        print("="*80)
        
        # Load datasets
        datasets = self.load_datasets()
        
        if not datasets:
            print("❌ No datasets found!")
            return None
        
        # Process each dataset
        for dataset_name, df in datasets.items():
            feature_sets = self.define_feature_sets(df)
            self.run_modeling_pipeline(dataset_name, df, feature_sets)
        
        # Create results summary
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            print("\n" + "="*80)
            print("📊 RESULTS SUMMARY")
            print("="*80)
            
            # Display key metrics
            display_columns = [
                'dataset', 'feature_set', 'model', 'balance_method',
                'auc', 'f1_score', 'recall', 'precision', 'optimal_threshold'
            ]
            
            # Filter to only show columns that exist
            display_columns = [col for col in display_columns if col in results_df.columns]
            display_df = results_df[display_columns]
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', '{:.4f}'.format)
            
            print(display_df.to_string(index=False))
            
            # Save results
            results_df.to_csv(self.results_dir / 'fixed_modeling_results.csv', index=False)
            print(f"\n✅ Results saved to: {self.results_dir / 'fixed_modeling_results.csv'}")
            
            # Create performance summary
            print("\n🎨 Creating performance visualizations...")
            self.create_performance_summary(results_df)
            
            print(f"✅ Analysis complete! Results in: {self.results_dir}")
            
            return results_df
        
        else:
            print("❌ No results generated!")
            return None

def main():
    """Main execution function"""
    
    # Initialize modeler
    modeler = FixedCreditScoringModeler(data_dir="../data", results_dir="../results")
    
    # Run analysis
    results = modeler.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    final_results = main() 