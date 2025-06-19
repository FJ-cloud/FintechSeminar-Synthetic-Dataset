#!/usr/bin/env python3
"""
Fast Credit Scoring Analysis - Streamlined for Key Insights

This script provides a focused, efficient modeling pipeline that runs in minutes, not hours.
Focuses on the most important comparisons for practical insights.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import time

# No XGBoost for faster, more reliable execution

plt.style.use('default')
sns.set_palette("husl")

class FastCreditScoringAnalyzer:
    """
    Fast, focused credit scoring analysis pipeline
    """
    
    def __init__(self, data_dir="../data", results_dir="../results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Focused feature sets
        self.feature_sets = {
            'digital_footprint': [
                'device_type', 'os', 'email_host', 'channel', 'checkout_time',
                'name_in_email', 'number_in_email', 'is_lowercase', 'email_error'
            ],
            'credit_score': ['credit_score_quintile'],
            'all_features': []  # Will be populated with all available features
        }
        
        self.results = []
        self.start_time = time.time()
        
    def load_key_datasets(self):
        """Load ALL available datasets for comprehensive analysis"""
        key_datasets = {
            'synthetic_basic_scorable': 'synthetic_digital_footprint_with_target.csv',
            'synthetic_copula_scorable': 'synthetic_digital_footprint_copula.csv',
            'synthetic_ctgan_scorable': 'synthetic_digital_footprint_ctgan.csv',
            'synthetic_basic_unscorable': 'synthetic_digital_footprint_with_target_unscorable.csv',
            'synthetic_copula_unscorable': 'synthetic_digital_footprint_copula_unscrorable.csv',
            'synthetic_ctgan_unscorable': 'synthetic_digital_footprint_ctgan_unscorable.csv',
        }
        
        datasets = {}
        for name, filename in key_datasets.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    print(f"✅ {name}: {df.shape[0]:,} rows, {df['TARGET'].mean():.1%} default rate")
                    datasets[name] = df
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
        
        return datasets
    
    def create_simple_preprocessor(self, X):
        """Create simple, fast preprocessor"""
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        
        transformers = []
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features))
        if numerical_features:
            transformers.append(('num', StandardScaler(), numerical_features))
        
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def quick_model_comparison(self, X_train, X_test, y_train, y_test, preprocessor, dataset_name, feature_set_name):
        """Quick comparison of key model configurations with cross-validation"""
        
        results = []
        
        # Model configurations - focus on core models without XGBoost
        configs = [
            ('Logistic Regression', LogisticRegression(random_state=42, max_iter=500), None),
            ('Logistic Regression (Balanced)', LogisticRegression(random_state=42, max_iter=500, class_weight='balanced'), None),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), None),
            ('Random Forest (Balanced)', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), None),
            ('Logistic Regression + SMOTE', LogisticRegression(random_state=42, max_iter=500), SMOTE(random_state=42)),
            ('Random Forest + SMOTE', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), SMOTE(random_state=42)),
        ]
        
        for model_name, model, resampler in configs:
            try:
                # Build pipeline
                pipeline_steps = [('preprocessor', preprocessor)]
                if resampler:
                    pipeline_steps.append(('resampler', resampler))
                pipeline_steps.append(('classifier', model))
                
                # Create pipeline
                if resampler:
                    pipeline = ImbPipeline(pipeline_steps)
                else:
                    pipeline = Pipeline(pipeline_steps)
                
                # Cross-validation scores
                X_combined = pd.concat([X_train, X_test])
                y_combined = pd.concat([y_train, y_test])
                
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                cv_auc_scores = cross_val_score(pipeline, X_combined, y_combined, cv=cv, scoring='roc_auc', n_jobs=1)
                cv_f1_scores = cross_val_score(pipeline, X_combined, y_combined, cv=cv, scoring='f1', n_jobs=1)
                
                # Fit and predict for test metrics
                pipeline.fit(X_train, y_train)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                
                # Simple threshold optimization
                thresholds = np.linspace(0.1, 0.9, 9)
                best_f1 = 0
                best_threshold = 0.5
                
                for thresh in thresholds:
                    y_pred_thresh = (y_prob >= thresh).astype(int)
                    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
                
                y_pred = (y_prob >= best_threshold).astype(int)
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_prob)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                result = {
                    'dataset': dataset_name,
                    'feature_set': feature_set_name,
                    'model': model_name,
                    'auc': auc_score,
                    'f1_score': best_f1,
                    'precision': precision,
                    'recall': recall,
                    'threshold': best_threshold,
                    'n_features': X_train.shape[1],
                    'cv_auc_mean': np.mean(cv_auc_scores),
                    'cv_auc_std': np.std(cv_auc_scores),
                    'cv_f1_mean': np.mean(cv_f1_scores),
                    'cv_f1_std': np.std(cv_f1_scores)
                }
                
                results.append(result)
                print(f"    {model_name:25s} AUC: {auc_score:.3f} (CV: {np.mean(cv_auc_scores):.3f}±{np.std(cv_auc_scores):.3f}), F1: {best_f1:.3f} (CV: {np.mean(cv_f1_scores):.3f}±{np.std(cv_f1_scores):.3f})")
                
            except Exception as e:
                print(f"    {model_name:25s} ❌ Error: {e}")
        
        return results
    
    def create_performance_plots(self, results_df):
        """Create focused performance visualizations"""
        
        # Performance by dataset and feature set
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credit Scoring Model Comparison', fontsize=16)
        
        # AUC comparison
        pivot_auc = results_df.pivot_table(values='auc', index='model', columns='dataset', aggfunc='mean')
        pivot_auc.plot(kind='barh', ax=axes[0,0])
        axes[0,0].set_title('AUC Score by Model and Dataset')
        axes[0,0].set_xlabel('AUC Score')
        
        # F1 comparison
        pivot_f1 = results_df.pivot_table(values='f1_score', index='model', columns='dataset', aggfunc='mean')
        pivot_f1.plot(kind='barh', ax=axes[0,1])
        axes[0,1].set_title('F1 Score by Model and Dataset')
        axes[0,1].set_xlabel('F1 Score')
        
        # Cross-validation comparison
        cv_comparison = results_df.groupby('model').agg({
            'cv_auc_mean': 'mean', 'cv_f1_mean': 'mean'
        }).reset_index()
        
        cv_comparison.plot(x='model', y='cv_auc_mean', kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Cross-Validation AUC by Model')
        axes[1,0].set_ylabel('CV AUC Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Best models summary
        best_models = results_df.loc[results_df.groupby('dataset')['f1_score'].idxmax()]
        best_models.plot(x='dataset', y='f1_score', kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Best F1 Score by Dataset')
        axes[1,1].set_ylabel('F1 Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_roc_comparison(self, datasets, results_df):
        """Create ROC curve comparison for best models"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('ROC Curves - Best Models by Dataset', fontsize=16)
        axes = axes.flatten()  # Flatten for easier indexing
        
        for idx, (dataset_name, df) in enumerate(datasets.items()):
            if idx >= 6:  # Plot all 6 datasets
                break
                
            ax = axes[idx]
            
            # Get best model for this dataset
            dataset_results = results_df[results_df['dataset'] == dataset_name]
            best_model_name = dataset_results.loc[dataset_results['f1_score'].idxmax(), 'model']
            
            # Recreate the best model for ROC curve
            feature_sets = self.define_feature_sets(df)
            
            for feature_set_name, features in feature_sets.items():
                if not features:
                    continue
                    
                X = df[features]
                y = df['TARGET']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                preprocessor = self.create_simple_preprocessor(X_train)
                
                # Simple logistic regression for ROC
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(random_state=42, max_iter=500, class_weight='balanced'))
                ])
                
                pipeline.fit(X_train, y_train)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{feature_set_name} (AUC={auc_score:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{dataset_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def define_feature_sets(self, df):
        """Define feature sets based on available columns"""
        available_columns = set(df.columns) - {'TARGET'}
        
        updated_sets = {}
        for set_name, features in self.feature_sets.items():
            if set_name == 'all_features':
                # Limit to reasonable number of features
                updated_sets[set_name] = list(available_columns)[:15]  # Max 15 features
            else:
                updated_sets[set_name] = [f for f in features if f in available_columns]
        
        return {k: v for k, v in updated_sets.items() if v}
    
    def run_fast_analysis(self):
        """Run the complete fast analysis"""
        
        print("=" * 60)
        print("🚀 COMPREHENSIVE CREDIT SCORING ANALYSIS (All Datasets + CV)")
        print("=" * 60)
        
        # Load datasets
        datasets = self.load_key_datasets()
        
        if not datasets:
            print("❌ No datasets found!")
            return None
        
        # Process each dataset
        for dataset_name, df in datasets.items():
            print(f"\n📊 Processing: {dataset_name}")
            print(f"    Class balance: {Counter(df['TARGET'])}")
            
            feature_sets = self.define_feature_sets(df)
            
            for feature_set_name, features in feature_sets.items():
                if not features:
                    continue
                    
                print(f"\n  🔍 Feature set: {feature_set_name} ({len(features)} features)")
                
                X = df[features]
                y = df['TARGET']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Create preprocessor
                preprocessor = self.create_simple_preprocessor(X_train)
                
                # Run quick model comparison
                feature_results = self.quick_model_comparison(
                    X_train, X_test, y_train, y_test, preprocessor, dataset_name, feature_set_name
                )
                
                self.results.extend(feature_results)
        
        # Create results summary
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            print("\n" + "=" * 60)
            print("📈 RESULTS SUMMARY")
            print("=" * 60)
            
            # Display top performers
            top_results = results_df.nlargest(10, 'cv_f1_mean')[['dataset', 'feature_set', 'model', 'cv_auc_mean', 'cv_f1_mean', 'auc', 'f1_score']]
            print("\n🏆 Top 10 Models by Cross-Validation F1 Score:")
            print(top_results.to_string(index=False, float_format='%.3f'))
            
            # Summary by dataset
            print("\n📊 Best Performance by Dataset:")
            best_by_dataset = results_df.loc[results_df.groupby('dataset')['cv_f1_mean'].idxmax()]
            summary_cols = ['dataset', 'model', 'feature_set', 'cv_auc_mean', 'cv_f1_mean', 'auc', 'f1_score']
            print(best_by_dataset[summary_cols].to_string(index=False, float_format='%.3f'))
            
            # Save results
            results_df.to_csv(self.results_dir / 'comprehensive_modeling_results.csv', index=False)
            
            # Create visualizations
            print(f"\n🎨 Creating visualizations...")
            self.create_performance_plots(results_df)
            self.create_roc_comparison(datasets, results_df)
            
            total_time = time.time() - self.start_time
            print(f"\n✅ Analysis complete in {total_time:.1f} seconds!")
            print(f"📁 Results saved to: {self.results_dir}")
            
            return results_df
        
        else:
            print("❌ No results generated!")
            return None

def main():
    """Main execution function"""
    analyzer = FastCreditScoringAnalyzer()
    results = analyzer.run_fast_analysis()
    return results

if __name__ == "__main__":
    final_results = main() 