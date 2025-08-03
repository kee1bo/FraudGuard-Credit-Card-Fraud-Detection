#!/usr/bin/env python3
"""
Quick training script for FraudGuard with real data
Trains models with minimal hyperparameter tuning for faster execution
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fraudguard.logger import fraud_logger

def load_real_data():
    """Load the real creditcard dataset"""
    data_path = "data/creditcard.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    fraud_logger.info(f"Loading real dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    fraud_logger.info(f"Dataset loaded:")
    fraud_logger.info(f"  Shape: {df.shape}")
    fraud_logger.info(f"  Fraud cases: {df['Class'].sum()} ({df['Class'].mean():.4f})")
    fraud_logger.info(f"  Features: {list(df.columns)}")
    
    return df

def quick_train_models(X_train, X_test, y_train, y_test):
    """Train models quickly with minimal tuning"""
    models = {}
    results = {}
    
    fraud_logger.info("Training models with minimal hyperparameter tuning...")
    
    # Random Forest - quick training
    fraud_logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # Logistic Regression - quick training
    fraud_logger.info("  Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    # Evaluate models
    for name, model in models.items():
        fraud_logger.info(f"  Evaluating {name}...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'classification_report': report,
            'roc_auc_score': auc_score,
            'confusion_matrix': conf_matrix.tolist(),
            'model_name': name
        }
        
        fraud_logger.info(f"    {name} Results:")
        fraud_logger.info(f"      AUC-ROC: {auc_score:.4f}")
        fraud_logger.info(f"      Precision: {report['1']['precision']:.4f}")
        fraud_logger.info(f"      Recall: {report['1']['recall']:.4f}")
        fraud_logger.info(f"      F1-Score: {report['1']['f1-score']:.4f}")
    
    return models, results

def save_models_and_artifacts(models, results, scaler):
    """Save models and artifacts in the expected format"""
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    preprocessors_dir = artifacts_dir / "preprocessors"
    reports_dir = artifacts_dir / "reports"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessors_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = preprocessors_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    fraud_logger.info(f"Saved scaler to {scaler_path}")
    
    # Save models
    for name, model in models.items():
        model_dir = models_dir / name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': name,
            'created_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'is_trained': True,
            'training_config': {
                'random_state': 42,
                'algorithm': name,
                'class_weight': 'balanced'
            }
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results[name], f, indent=2, default=str)
        
        fraud_logger.info(f"Saved {name} model to {model_dir}")
    
    # Save overall results
    comparison_path = reports_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    performance_path = reports_dir / "performance_metrics.json"
    with open(performance_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    fraud_logger.info(f"Saved reports to {reports_dir}")

def main():
    """Main function for quick training"""
    fraud_logger.info("Starting quick training with real creditcard data...")
    
    try:
        # Load real data
        df = load_real_data()
        
        # Prepare data
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        fraud_logger.info(f"Data split:")
        fraud_logger.info(f"  Train: {X_train.shape}, Fraud: {y_train.sum()}/{len(y_train)} ({y_train.mean():.4f})")
        fraud_logger.info(f"  Test: {X_test.shape}, Fraud: {y_test.sum()}/{len(y_test)} ({y_test.mean():.4f})")
        
        # Scale features
        fraud_logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models, results = quick_train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save everything
        save_models_and_artifacts(models, results, scaler)
        
        fraud_logger.info("\n✅ Quick training completed successfully!")
        fraud_logger.info("Models are ready for the professional web interface.")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  AUC-ROC: {result['roc_auc_score']:.4f}")
            print(f"  Precision: {result['classification_report']['1']['precision']:.4f}")
            print(f"  Recall: {result['classification_report']['1']['recall']:.4f}")
            print(f"  F1-Score: {result['classification_report']['1']['f1-score']:.4f}")
        print("="*60)
        
    except Exception as e:
        fraud_logger.error(f"Quick training failed: {e}")
        raise

if __name__ == "__main__":
    main()