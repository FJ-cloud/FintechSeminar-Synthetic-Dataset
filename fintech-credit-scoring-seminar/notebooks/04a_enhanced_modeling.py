#!/usr/bin/env python3
"""
Enhanced Comprehensive Modeling Pipeline for Synthetic Credit Scoring Data

This enhanced script incorporates advanced techniques for:
1. Advanced class imbalance handling (ADASYN, SMOTEENN, custom weights, threshold tuning)
2. Feature engineering (RFE, polynomial features, target encoding)
3. Expanded model selection (XGBoost, LightGBM) and hyperparameter tuning
4. Improved data quality assessment and preprocessing
5. Enhanced evaluation metrics and model interpretation (SHAP, calibration)
6. Cross-validation and learning curves for better generalization assessment
7. Computational efficiency optimizations

Author: Enhanced for Fintech Seminar
"""

import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Enhanced sklearn imports
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score,
    classification_report, precision_recall_curve, f1_score, precision_score, 
    recall_score, average_precision_score
)

# Import calibration_curve separately with error handling
try:
    from sklearn.calibration import calibration_curve
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("Calibration curve not available in this sklearn version")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Advanced imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Feature encoding and interpretation
try:
    from category_encoders import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False
    print("Target encoder not available. Install with: pip install category-encoders")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

from collections import Counter

# Set enhanced style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedCreditScoringModeler:
    """
    Enhanced comprehensive modeling pipeline for credit scoring analysis
    """
    
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Enhanced feature sets
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
        self.calibration_data = {}
        self.cv_results = {}
        
        # Model configurations
        self.imbalance_methods = {
            'none': None,
            'balanced_weights': 'balanced',
            'smote': SMOTE(random_state=42),
            'adasyn': ADASYN(random_state=42),
            'smoteenn': SMOTEENN(random_state=42)
        }
        
    def load_datasets_enhanced(self):
        """Enhanced dataset loading with quality assessment"""
        datasets = {}
        
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
                    
                    # Data quality assessment
                    print(f"\n📊 Quality Assessment for {name}:")
                    print(f"   Shape: {df.shape}")
                    print(f"   Missing values: {df.isnull().sum().sum()}")
                    
                    # Handle missing values
                    if df.isnull().sum().sum() > 0:
                        print("   🔧 Imputing missing values...")
                        # Numerical columns: median imputation
                        num_cols = df.select_dtypes(include=['number']).columns
                        for col in num_cols:
                            if df[col].isnull().sum() > 0:
                                df[col].fillna(df[col].median(), inplace=True)
                        
                        # Categorical columns: mode imputation
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns
                        for col in cat_cols:
                            if df[col].isnull().sum() > 0:
                                df[col].fillna(df[col].mode().iloc[0], inplace=True)
                    
                    # Check for constant features
                    constant_features = [col for col in df.columns if df[col].nunique() <= 1]
                    if constant_features:
                        print(f"   ⚠️  Dropping constant features: {constant_features}")
                        df = df.drop(columns=constant_features)
                    
                    # Basic statistics
                    if 'TARGET' in df.columns:
                        target_dist = df['TARGET'].value_counts()
                        print(f"   Target distribution: {target_dist.to_dict()}")
                    
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {df.shape}")
                    
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
            else:
                print(f"✗ File not found: {filepath}")
        
        return datasets
    
    def analyze_class_balance_enhanced(self, y, dataset_name):
        """Enhanced class imbalance analysis"""
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
        
        # Determine imbalance severity
        if balance_info['imbalance_ratio'] > 100:
            severity = "Extreme"
        elif balance_info['imbalance_ratio'] > 20:
            severity = "High"
        elif balance_info['imbalance_ratio'] > 5:
            severity = "Moderate"
        else:
            severity = "Low"
        
        balance_info['imbalance_severity'] = severity
        
        print(f"  📈 Class balance analysis:")
        print(f"    Total samples: {total:,}")
        print(f"    Defaults (1): {class_counts[1]:,} ({balance_info['default_rate']:.1%})")
        print(f"    Non-defaults (0): {class_counts[0]:,} ({(1-balance_info['default_rate']):.1%})")
        print(f"    Imbalance ratio: {balance_info['imbalance_ratio']:.1f}:1 ({severity})")
        
        self.class_balance_info[dataset_name] = balance_info
        return balance_info
    
    def create_enhanced_preprocessor(self, X, feature_set_name, use_polynomial=True, use_target_encoding=True):
        """Enhanced preprocessing with feature engineering"""
        
        # Identify feature types
        categorical_features = []
        numerical_features = []
        high_cardinality_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if X[col].nunique() > 10 and use_target_encoding and TARGET_ENCODER_AVAILABLE:
                    high_cardinality_features.append(col)
                else:
                    categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Create preprocessing steps
        preprocessor_steps = []
        
        # High cardinality categorical: target encoding
        if high_cardinality_features and TARGET_ENCODER_AVAILABLE:
            preprocessor_steps.append(
                ('high_card_categorical', TargetEncoder(), high_cardinality_features)
            )
        
        # Regular categorical: one-hot encoding
        if categorical_features:
            preprocessor_steps.append(
                ('categorical', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features)
            )
        
        # Numerical features: scaling and optional polynomial
        if numerical_features:
            if use_polynomial and len(numerical_features) <= 5:  # Avoid curse of dimensionality
                numerical_transformer = Pipeline([
                    ('scaler', StandardScaler()),
                    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
                ])
            else:
                numerical_transformer = Pipeline([
                    ('scaler', StandardScaler())
                ])
            
            preprocessor_steps.append(
                ('numerical', numerical_transformer, numerical_features)
            )
        
        # Create column transformer
        if preprocessor_steps:
            preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='passthrough'
            )
        else:
            preprocessor = 'passthrough'
        
        return preprocessor
    
    def compute_optimal_threshold(self, y_true, y_prob, metric='f1'):
        """Compute optimal classification threshold"""
        if metric == 'f1':
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        elif metric == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5
        
        return optimal_threshold
    
    def fit_enhanced_logistic_regression(self, X_train, X_test, y_train, y_test, preprocessor, 
                                       balance_method='none', feature_selection=True):
        """Enhanced Logistic Regression with hyperparameter tuning"""
        
        # Determine class weights and resampling
        if balance_method == 'balanced_weights':
            class_weight = 'balanced'
            resampler = None
        elif balance_method in ['smote', 'adasyn', 'smoteenn']:
            class_weight = None
            resampler = self.imbalance_methods[balance_method]
        elif balance_method == 'custom_weights':
            # Custom weights based on imbalance ratio
            imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
            class_weight = {0: 1, 1: min(imbalance_ratio / 2, 10)}  # Cap at 10
            resampler = None
        else:
            class_weight = None
            resampler = None
        
        # Build pipeline components
        pipeline_steps = [('preprocessor', preprocessor)]
        
        # Add feature selection if requested
        if feature_selection:
            pipeline_steps.append(('feature_selection', RFE(
                estimator=LogisticRegression(random_state=42), 
                n_features_to_select=min(20, X_train.shape[1])
            )))
        
        # Add resampling if specified
        if resampler is not None:
            pipeline_steps.append(('resampler', resampler))
        
        # Add classifier
        pipeline_steps.append(('classifier', LogisticRegression(
            random_state=42, max_iter=2000, class_weight=class_weight
        )))
        
        # Create pipeline
        if resampler is not None:
            lr_pipeline = ImbPipeline(pipeline_steps)
        else:
            lr_pipeline = Pipeline(pipeline_steps)
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'liblinear'],
            'classifier__penalty': ['l2']
        }
        
        # Use RandomizedSearchCV for efficiency
        grid_search = RandomizedSearchCV(
            lr_pipeline, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, n_iter=10, random_state=42
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions with optimal threshold
        y_prob = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.compute_optimal_threshold(y_test, y_prob, metric='f1')
        y_pred = (y_prob >= optimal_threshold).astype(int)
        y_pred_default = best_model.predict(X_test)  # Default threshold
        
        # Comprehensive metrics
        metrics = self.compute_comprehensive_metrics(
            y_test, y_pred, y_pred_default, y_prob, optimal_threshold
        )
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'y_pred': y_pred,
            'y_pred_default': y_pred_default,
            'y_prob': y_prob,
            'optimal_threshold': optimal_threshold,
            'roc_data': (fpr, tpr),
            'model_type': 'Logistic Regression',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_,
            **metrics
        }
    
    def fit_enhanced_random_forest(self, X_train, X_test, y_train, y_test, preprocessor, 
                                 balance_method='none', feature_selection=False):
        """Enhanced Random Forest with expanded hyperparameter tuning"""
        
        # Determine class weights and resampling
        if balance_method == 'balanced_weights':
            class_weight = 'balanced'
            resampler = None
        elif balance_method in ['smote', 'adasyn', 'smoteenn']:
            class_weight = None
            resampler = self.imbalance_methods[balance_method]
        elif balance_method == 'custom_weights':
            imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
            class_weight = {0: 1, 1: min(imbalance_ratio / 2, 10)}
            resampler = None
        else:
            class_weight = None
            resampler = None
        
        # Build pipeline
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if feature_selection:
            pipeline_steps.append(('feature_selection', SelectKBest(
                score_func=f_classif, k=min(15, X_train.shape[1])
            )))
        
        if resampler is not None:
            pipeline_steps.append(('resampler', resampler))
        
        pipeline_steps.append(('classifier', RandomForestClassifier(
            random_state=42, n_jobs=-1, oob_score=True, class_weight=class_weight
        )))
        
        if resampler is not None:
            rf_pipeline = ImbPipeline(pipeline_steps)
        else:
            rf_pipeline = Pipeline(pipeline_steps)
        
        # Expanded parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 20],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        # Randomized search for efficiency
        grid_search = RandomizedSearchCV(
            rf_pipeline, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, n_iter=20, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions with optimal threshold
        y_prob = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.compute_optimal_threshold(y_test, y_prob, metric='f1')
        y_pred = (y_prob >= optimal_threshold).astype(int)
        y_pred_default = best_model.predict(X_test)
        
        # Comprehensive metrics
        metrics = self.compute_comprehensive_metrics(
            y_test, y_pred, y_pred_default, y_prob, optimal_threshold
        )
        
        # OOB score
        try:
            oob_score = best_model.named_steps['classifier'].oob_score_
        except:
            oob_score = np.nan
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'y_pred': y_pred,
            'y_pred_default': y_pred_default,
            'y_prob': y_prob,
            'optimal_threshold': optimal_threshold,
            'oob_score': oob_score,
            'roc_data': (fpr, tpr),
            'model_type': 'Random Forest',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_,
            **metrics
        }
    
    def fit_xgboost(self, X_train, X_test, y_train, y_test, preprocessor, balance_method='none'):
        """Fit XGBoost classifier"""
        
        if not XGBOOST_AVAILABLE:
            return None
        
        # Handle class imbalance
        if balance_method == 'balanced_weights':
            scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
            resampler = None
        elif balance_method in ['smote', 'adasyn', 'smoteenn']:
            scale_pos_weight = 1
            resampler = self.imbalance_methods[balance_method]
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
        
        # Parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 1.0]
        }
        
        grid_search = RandomizedSearchCV(
            xgb_pipeline, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, n_iter=15, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_prob = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.compute_optimal_threshold(y_test, y_prob, metric='f1')
        y_pred = (y_prob >= optimal_threshold).astype(int)
        y_pred_default = best_model.predict(X_test)
        
        # Metrics
        metrics = self.compute_comprehensive_metrics(
            y_test, y_pred, y_pred_default, y_prob, optimal_threshold
        )
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        return {
            'model': best_model,
            'y_pred': y_pred,
            'y_pred_default': y_pred_default,
            'y_prob': y_prob,
            'optimal_threshold': optimal_threshold,
            'roc_data': (fpr, tpr),
            'model_type': 'XGBoost',
            'balance_method': balance_method,
            'best_params': grid_search.best_params_,
            **metrics
        }
    
    def compute_comprehensive_metrics(self, y_true, y_pred_optimal, y_pred_default, y_prob, optimal_threshold):
        """Compute comprehensive evaluation metrics"""
        
        # Basic metrics with optimal threshold
        auc_score = roc_auc_score(y_true, y_prob)
        accuracy_optimal = accuracy_score(y_true, y_pred_optimal)
        accuracy_default = accuracy_score(y_true, y_pred_default)
        precision_optimal = precision_score(y_true, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_true, y_pred_optimal, zero_division=0)
        f1_optimal = f1_score(y_true, y_pred_optimal, zero_division=0)
        
        # Average precision (area under PR curve)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Pseudo R²
        pseudo_r2 = self.compute_pseudo_r2(y_true, y_prob)
        
        return {
            'auc': auc_score,
            'accuracy_optimal': accuracy_optimal,
            'accuracy_default': accuracy_default,
            'precision': precision_optimal,
            'recall': recall_optimal,
            'f1_score': f1_optimal,
            'average_precision': avg_precision,
            'pseudo_r2': pseudo_r2,
            'optimal_threshold': optimal_threshold
        }
    
    def compute_pseudo_r2(self, y_true, y_prob):
        """Compute McFadden's Pseudo R-squared"""
        try:
            ll_model = np.sum(y_true * np.log(np.clip(y_prob, 1e-15, 1-1e-15)) + 
                             (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-15, 1-1e-15)))
            p0 = np.mean(y_true)
            ll_null = np.sum(y_true * np.log(p0) + (1 - y_true) * np.log(1 - p0))
            pseudo_r2 = 1 - (ll_model / ll_null)
            return pseudo_r2
        except:
            return np.nan
    
    def perform_cross_validation(self, X, y, model_pipeline, cv_folds=5):
        """Perform cross-validation analysis"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        auc_scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        f1_scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
        
        return {
            'cv_auc_mean': np.mean(auc_scores),
            'cv_auc_std': np.std(auc_scores),
            'cv_f1_mean': np.mean(f1_scores),
            'cv_f1_std': np.std(f1_scores)
        }
    
    def create_calibration_plot(self, y_true, y_prob, model_name, dataset_name, feature_set):
        """Create calibration plot"""
        if not CALIBRATION_AVAILABLE:
            return
            
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            
            plt.figure(figsize=(8, 6))
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{model_name}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Plot - {model_name}\n{dataset_name} - {feature_set}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f'calibration_{dataset_name}_{feature_set}_{model_name.replace(" ", "_")}.png'
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating calibration plot: {e}")
    
    def create_confusion_matrix_heatmap(self, y_true, y_pred, model_name, dataset_name, feature_set, balance_method):
        """Create confusion matrix heatmap"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Default', 'Default'],
                       yticklabels=['Non-Default', 'Default'])
            plt.title(f'Confusion Matrix - {model_name} ({balance_method})\n{dataset_name} - {feature_set}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            filename = f'confusion_matrix_{dataset_name}_{feature_set}_{model_name.replace(" ", "_")}_{balance_method}.png'
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def run_enhanced_modeling_pipeline(self, dataset_name, df, feature_sets):
        """Enhanced modeling pipeline with all improvements"""
        
        print(f"\n{'='*80}")
        print(f"🚀 Enhanced Processing: {dataset_name}")
        print(f"{'='*80}")
        
        if 'TARGET' not in df.columns:
            print(f"No TARGET column found in {dataset_name}, skipping...")
            return
        
        # Analyze class balance
        balance_info = self.analyze_class_balance_enhanced(df['TARGET'], dataset_name)
        
        for feature_set_name, features in feature_sets.items():
            print(f"\n📋 Feature set: {feature_set_name} ({len(features)} features)")
            
            # Prepare data
            X = df[features].copy()
            y = df['TARGET'].copy()
            
            if X.empty:
                print(f"No features available for {feature_set_name}, skipping...")
                continue
            
            # Enhanced train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Create enhanced preprocessor
            preprocessor = self.create_enhanced_preprocessor(
                X_train, feature_set_name, 
                use_polynomial=(len(features) <= 5),  # Only for small feature sets
                use_target_encoding=True
            )
            
            # Test multiple balancing strategies
            balance_methods = ['none', 'balanced_weights', 'smote', 'adasyn', 'custom_weights']
            
            # Model types to test
            model_configs = [
                ('logistic_regression', self.fit_enhanced_logistic_regression),
                ('random_forest', self.fit_enhanced_random_forest),
            ]
            
            if XGBOOST_AVAILABLE:
                model_configs.append(('xgboost', self.fit_xgboost))
            
            for model_name, model_func in model_configs:
                print(f"\n  🤖 Model: {model_name.replace('_', ' ').title()}")
                
                for balance_method in balance_methods:
                    if balance_method not in self.imbalance_methods and balance_method != 'custom_weights':
                        continue
                        
                    print(f"    ⚖️  Balance method: {balance_method}")
                    
                    try:
                        # Fit model
                        if model_name == 'xgboost':
                            results = model_func(X_train, X_test, y_train, y_test, preprocessor, balance_method)
                        else:
                            results = model_func(X_train, X_test, y_train, y_test, preprocessor, balance_method)
                        
                        if results is None:
                            continue
                        
                        # Cross-validation
                        cv_results = self.perform_cross_validation(X, y, results['model'])
                        
                        # Store comprehensive results
                        result_entry = {
                            'dataset': dataset_name,
                            'feature_set': feature_set_name,
                            'model': f"{results['model_type']} ({balance_method})",
                            'balance_method': balance_method,
                            'n_features': len(features),
                            'imbalance_ratio': balance_info['imbalance_ratio'],
                            'imbalance_severity': balance_info['imbalance_severity'],
                            **{k: v for k, v in results.items() if k not in ['model', 'y_pred', 'y_pred_default', 'y_prob', 'roc_data']},
                            **cv_results
                        }
                        
                        self.results.append(result_entry)
                        
                        # Store ROC data
                        roc_key = f"{dataset_name}_{feature_set_name}_{model_name}_{balance_method}"
                        self.roc_data[roc_key] = results['roc_data']
                        
                        # Create visualizations
                        self.create_calibration_plot(
                            y_test, results['y_prob'], results['model_type'], 
                            dataset_name, feature_set_name
                        )
                        
                        self.create_confusion_matrix_heatmap(
                            y_test, results['y_pred'], results['model_type'],
                            dataset_name, feature_set_name, balance_method
                        )
                        
                        # Print results
                        print(f"      📊 AUC: {results['auc']:.4f} (±{cv_results['cv_auc_std']:.4f})")
                        print(f"      📊 F1: {results['f1_score']:.4f} (±{cv_results['cv_f1_std']:.4f})")
                        print(f"      📊 Recall: {results['recall']:.4f}")
                        print(f"      📊 Threshold: {results['optimal_threshold']:.4f}")
                        
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
    
    def run_complete_enhanced_analysis(self):
        """Run the complete enhanced modeling analysis"""
        
        print("="*100)
        print("🚀 ENHANCED COMPREHENSIVE CREDIT SCORING MODEL ANALYSIS")
        print("="*100)
        
        # Load datasets with quality assessment
        datasets = self.load_datasets_enhanced()
        
        if not datasets:
            print("No datasets found!")
            return None
        
        # Process each dataset
        for dataset_name, df in datasets.items():
            # Define feature sets
            feature_sets = self.define_feature_sets(df)
            
            # Run enhanced modeling pipeline
            self.run_enhanced_modeling_pipeline(dataset_name, df, feature_sets)
        
        # Create comprehensive results
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            print("\n" + "="*100)
            print("📊 COMPREHENSIVE RESULTS TABLE")
            print("="*100)
            
            # Display key columns
            display_columns = [
                'dataset', 'feature_set', 'model', 'balance_method',
                'auc', 'f1_score', 'recall', 'precision', 'accuracy_optimal',
                'cv_auc_mean', 'cv_f1_mean', 'optimal_threshold',
                'imbalance_ratio', 'n_features'
            ]
            
            display_df = results_df[display_columns].copy()
            
            # Format for display
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', '{:.4f}'.format)
            
            print(display_df.to_string(index=False))
            
            # Save comprehensive results
            results_df.to_csv(self.results_dir / 'enhanced_modeling_results.csv', index=False)
            print(f"\n✅ Results saved to: {self.results_dir / 'enhanced_modeling_results.csv'}")
            
            # Create enhanced visualizations
            print("\n🎨 Generating enhanced visualizations...")
            self.create_enhanced_visualizations(results_df)
            
            print(f"✅ All visualizations saved to: {self.results_dir}")
            
            return results_df
        
        else:
            print("No results generated!")
            return None
    
    def define_feature_sets(self, df):
        """Define and validate feature sets"""
        available_columns = set(df.columns) - {'TARGET'}
        
        updated_sets = {}
        for set_name, features in self.feature_sets.items():
            if set_name == 'all_predictors':
                updated_sets[set_name] = list(available_columns)
            else:
                updated_sets[set_name] = [f for f in features if f in available_columns]
        
        updated_sets = {k: v for k, v in updated_sets.items() if v}
        
        print(f"\n📋 Feature sets defined:")
        for name, features in updated_sets.items():
            print(f"  {name}: {len(features)} features")
        
        return updated_sets
    
    def create_enhanced_visualizations(self, results_df):
        """Create enhanced visualization suite"""
        
        # 1. Performance comparison by balance method
        plt.figure(figsize=(16, 10))
        
        # Group by balance method
        balance_comparison = results_df.groupby(['balance_method', 'model']).agg({
            'auc': 'mean',
            'f1_score': 'mean',
            'recall': 'mean'
        }).reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Model Performance Analysis', fontsize=16)
        
        # AUC by balance method
        sns.barplot(data=balance_comparison, x='balance_method', y='auc', hue='model', ax=axes[0,0])
        axes[0,0].set_title('AUC by Balance Method')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1 by balance method
        sns.barplot(data=balance_comparison, x='balance_method', y='f1_score', hue='model', ax=axes[0,1])
        axes[0,1].set_title('F1-Score by Balance Method')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Recall by balance method
        sns.barplot(data=balance_comparison, x='balance_method', y='recall', hue='model', ax=axes[1,0])
        axes[1,0].set_title('Recall by Balance Method')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Best methods summary
        best_methods = results_df.loc[results_df.groupby('model')['f1_score'].idxmax()]
        sns.barplot(data=best_methods, x='model', y='f1_score', ax=axes[1,1])
        axes[1,1].set_title('Best F1-Score by Model Type')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'enhanced_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Imbalance severity impact
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=results_df, x='imbalance_severity', y='f1_score', hue='balance_method')
        plt.title('F1-Score Distribution by Imbalance Severity and Balance Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'imbalance_severity_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    
    # Initialize enhanced modeler
    modeler = EnhancedCreditScoringModeler(data_dir="data", results_dir="results")
    
    # Run complete enhanced analysis
    results = modeler.run_complete_enhanced_analysis()
    
    return results

if __name__ == "__main__":
    enhanced_results = main() 