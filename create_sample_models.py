#!/usr/bin/env python3
"""
Simple script to create sample models for testing the professional UI
This generates synthetic fraud data and trains basic models
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
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

def create_synthetic_fraud_data(n_samples=10000, n_features=30, fraud_rate=0.02):
    """Create synthetic fraud detection dataset"""
    print(f"Creating synthetic fraud dataset with {n_samples} samples...")
    
    # Create imbalanced classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[1-fraud_rate, fraud_rate],  # Imbalanced classes
        flip_y=0.01,  # Add some noise
        random_state=42
    )
    
    # Create feature names similar to credit card data
    feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, n_features-1)]
    
    # Make Time and Amount more realistic
    X[:, 0] = np.random.exponential(scale=50000, size=n_samples)  # Time in seconds
    X[:, 1] = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)  # Transaction amounts
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    
    print(f"Dataset created: {len(df)} samples, {sum(y)} fraudulent ({sum(y)/len(y)*100:.2f}%)")
    return df

def train_simple_models(X_train, X_test, y_train, y_test):
    """Train simple models for testing"""
    models = {}
    results = {}
    
    print("Training models...")
    
    # Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # Logistic Regression
    print("  Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    # Evaluate models
    for name, model in models.items():
        print(f"  Evaluating {name}...")
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
        
        print(f"    {name} - AUC: {auc_score:.4f}, F1: {report['1']['f1-score']:.4f}")
    
    return models, results

def save_models_and_metadata(models, results, scaler):
    """Save models and metadata in the expected format"""
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    preprocessors_dir = artifacts_dir / "preprocessors"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessors_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = preprocessors_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
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
                'algorithm': name
            }
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results[name], f, indent=2, default=str)
        
        print(f"Saved {name} model to {model_dir}")
    
    # Save overall results
    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    comparison_path = reports_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved model comparison to {comparison_path}")

def main():
    """Main function to create sample models"""
    print("Creating sample models for FraudGuard testing...")
    
    # Create synthetic data
    df = create_synthetic_fraud_data(n_samples=5000, n_features=30)
    
    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models, results = train_simple_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save everything
    save_models_and_metadata(models, results, scaler)
    
    print("\n✅ Sample models created successfully!")
    print("You can now run the web application and see the professional UI with real data.")
    print("\nTo start the app:")
    print("  source venv/bin/activate")
    print("  python3 run_app.py")

if __name__ == "__main__":
    main()