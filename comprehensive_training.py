#!/usr/bin/env python3
"""
Comprehensive Model Training for FraudGuard
Trains multiple models and generates all data needed for professional dashboards
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, roc_curve, average_precision_score
)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def create_realistic_fraud_data(n_samples=50000, n_features=30, fraud_rate=0.002):
    """Create realistic fraud detection dataset"""
    print(f"Creating realistic fraud dataset with {n_samples} samples...")
    
    # Create highly imbalanced dataset (0.2% fraud rate - realistic)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=25,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[1-fraud_rate, fraud_rate],
        flip_y=0.005,  # Very low noise
        random_state=42,
        class_sep=0.8  # Make it challenging but learnable
    )
    
    # Create realistic feature names and values
    feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, n_features-1)]
    
    # Make Time and Amount more realistic
    X[:, 0] = np.random.exponential(scale=30000, size=n_samples)  # Time in seconds over ~8 hours
    X[:, 1] = np.random.lognormal(mean=4, sigma=1.8, size=n_samples)  # Transaction amounts $1-$10k+
    
    # Make fraud transactions have different patterns
    fraud_indices = np.where(y == 1)[0]
    # Fraudulent transactions tend to be higher amounts
    X[fraud_indices, 1] *= np.random.uniform(1.5, 3.0, len(fraud_indices))
    # And occur at unusual times
    X[fraud_indices, 0] *= np.random.uniform(0.3, 0.7, len(fraud_indices))
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    
    print(f"Dataset created: {len(df)} samples, {sum(y)} fraudulent ({sum(y)/len(y)*100:.3f}%)")
    return df, feature_names

def train_comprehensive_models(X_train, X_test, y_train, y_test, feature_names):
    """Train comprehensive set of models with proper evaluation"""
    models = {}
    results = {}
    feature_importance = {}
    
    print("Training comprehensive model suite...")
    
    # 1. Logistic Regression (baseline)
    print("  Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced',  # Handle imbalance
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    # Feature importance for LR (absolute coefficients)
    feature_importance['logistic_regression'] = dict(zip(
        feature_names, np.abs(lr.coef_[0])
    ))
    
    # 2. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    feature_importance['random_forest'] = dict(zip(
        feature_names, rf.feature_importances_
    ))
    
    # 3. XGBoost (if available)
    if HAS_XGB:
        print("  Training XGBoost...")
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        feature_importance['xgboost'] = dict(zip(
            feature_names, xgb_model.feature_importances_
        ))
    
    # 4. LightGBM (if available)
    if HAS_LGB:
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        feature_importance['lightgbm'] = dict(zip(
            feature_names, lgb_model.feature_importances_
        ))
    
    # 5. CatBoost (if available)
    if HAS_CATBOOST:
        print("  Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            class_weights=[1, 10],  # Handle imbalance
            random_seed=42,
            verbose=False
        )
        cb_model.fit(X_train, y_train)
        models['catboost'] = cb_model
        
        feature_importance['catboost'] = dict(zip(
            feature_names, cb_model.feature_importances_
        ))
    
    # Evaluate all models
    print("Evaluating models...")
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        results[name] = {
            'classification_report': report,
            'roc_auc_score': float(auc_score),
            'average_precision_score': float(avg_precision),
            'confusion_matrix': conf_matrix.tolist(),
            'cross_val_auc_mean': float(cv_scores.mean()),
            'cross_val_auc_std': float(cv_scores.std()),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            },
            'model_name': name,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'fraud_rate_train': float(sum(y_train) / len(y_train)),
            'fraud_rate_test': float(sum(y_test) / len(y_test))
        }
        
        # Print key metrics
        fraud_f1 = report['1']['f1-score'] if '1' in report else 0
        print(f"    {name:15} - AUC: {auc_score:.4f}, F1: {fraud_f1:.4f}, AP: {avg_precision:.4f}")
    
    return models, results, feature_importance

def create_dashboard_data(results, feature_importance):
    """Create comprehensive data for dashboards"""
    dashboard_data = {
        'model_comparison': {},
        'performance_summary': {},
        'feature_analysis': {},
        'training_metadata': {
            'created_at': datetime.now().isoformat(),
            'total_models': len(results),
            'best_model': None,
            'best_auc': 0
        }
    }
    
    # Model comparison data
    for name, result in results.items():
        fraud_metrics = result['classification_report'].get('1', {})
        
        dashboard_data['model_comparison'][name] = {
            'auc_roc': result['roc_auc_score'],
            'average_precision': result['average_precision_score'],
            'precision': fraud_metrics.get('precision', 0),
            'recall': fraud_metrics.get('recall', 0),
            'f1_score': fraud_metrics.get('f1-score', 0),
            'cv_auc_mean': result['cross_val_auc_mean'],
            'cv_auc_std': result['cross_val_auc_std'],
            'training_samples': result['training_samples'],
            'confusion_matrix': result['confusion_matrix']
        }
        
        # Track best model
        if result['roc_auc_score'] > dashboard_data['training_metadata']['best_auc']:
            dashboard_data['training_metadata']['best_auc'] = result['roc_auc_score']
            dashboard_data['training_metadata']['best_model'] = name
    
    # Performance summary
    dashboard_data['performance_summary'] = {
        'average_auc': np.mean([r['roc_auc_score'] for r in results.values()]),
        'best_auc': max([r['roc_auc_score'] for r in results.values()]),
        'model_count': len(results),
        'fraud_detection_rates': {
            name: results[name]['classification_report'].get('1', {}).get('recall', 0)
            for name in results.keys()
        }
    }
    
    # Feature importance analysis
    if feature_importance:
        # Aggregate feature importance across models
        all_features = set()
        for model_features in feature_importance.values():
            all_features.update(model_features.keys())
        
        feature_summary = {}
        for feature in all_features:
            importances = []
            for model_name, model_features in feature_importance.items():
                if feature in model_features:
                    importances.append(model_features[feature])
            
            if importances:
                feature_summary[feature] = {
                    'mean_importance': float(np.mean(importances)),
                    'std_importance': float(np.std(importances)),
                    'max_importance': float(np.max(importances)),
                    'models_count': len(importances)
                }
        
        # Sort by mean importance
        sorted_features = sorted(
            feature_summary.items(), 
            key=lambda x: x[1]['mean_importance'], 
            reverse=True
        )
        
        dashboard_data['feature_analysis'] = {
            'top_features': dict(sorted_features[:15]),  # Top 15 features
            'feature_importance_by_model': feature_importance
        }
    
    return dashboard_data

def save_comprehensive_artifacts(models, results, feature_importance, dashboard_data, scaler, feature_names):
    """Save all artifacts in organized structure"""
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    preprocessors_dir = artifacts_dir / "preprocessors"
    reports_dir = artifacts_dir / "reports"
    
    # Create directories
    for dir_path in [models_dir, preprocessors_dir, reports_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving comprehensive artifacts...")
    
    # Save scaler and feature names
    joblib.dump(scaler, preprocessors_dir / "scaler.pkl")
    joblib.dump(feature_names, preprocessors_dir / "feature_names.pkl")
    
    # Save models with comprehensive metadata
    for name, model in models.items():
        model_dir = models_dir / name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(model, model_dir / "model.pkl")
        
        # Save comprehensive metadata
        metadata = {
            'model_name': name,
            'created_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'is_trained': True,
            'training_config': getattr(model, 'get_params', lambda: {})(),
            'performance_metrics': results[name],
            'feature_importance': feature_importance.get(name, {}),
            'model_size_mb': 0  # Would calculate actual size
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save metrics separately for compatibility
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(results[name], f, indent=2, default=str)
        
        print(f"  Saved {name} model and metadata")
    
    # Save comprehensive reports
    reports = {
        'model_comparison.json': results,
        'dashboard_data.json': dashboard_data,
        'feature_importance.json': feature_importance,
        'performance_summary.json': dashboard_data['performance_summary']
    }
    
    for filename, data in reports.items():
        with open(reports_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved {filename}")
    
    print(f"✅ All artifacts saved to {artifacts_dir}")

def main():
    """Main comprehensive training function"""
    print("🚀 Starting Comprehensive FraudGuard Model Training...")
    print("=" * 60)
    
    # Create realistic dataset
    df, feature_names = create_realistic_fraud_data(n_samples=50000, fraud_rate=0.002)
    
    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Dataset summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Fraud cases: {sum(y)} ({sum(y)/len(y)*100:.3f}%)")
    print(f"  Normal cases: {len(y) - sum(y)} ({(len(y) - sum(y))/len(y)*100:.3f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train comprehensive models
    models, results, feature_importance = train_comprehensive_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # Create dashboard data
    print("Creating dashboard data...")
    dashboard_data = create_dashboard_data(results, feature_importance)
    
    # Save everything
    save_comprehensive_artifacts(
        models, results, feature_importance, dashboard_data, scaler, feature_names
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✅ Trained {len(models)} models")
    print(f"✅ Best model: {dashboard_data['training_metadata']['best_model']} (AUC: {dashboard_data['training_metadata']['best_auc']:.4f})")
    print(f"✅ Average AUC across models: {dashboard_data['performance_summary']['average_auc']:.4f}")
    print(f"✅ Top 3 features: {list(dashboard_data['feature_analysis']['top_features'].keys())[:3]}")
    
    print("\n📊 Model Performance Summary:")
    for name, metrics in dashboard_data['model_comparison'].items():
        print(f"  {name:15} - AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}")
    
    print(f"\n🎯 All data ready for professional dashboards!")
    print("   Run the web app to see comprehensive model analysis.")

if __name__ == "__main__":
    main()