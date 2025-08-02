#!/usr/bin/env python3
'''
Script to create test data for fraud detection system
'''

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from fraudguard.logger import fraud_logger
    from fraudguard.constants.constants import RAW_DATA_DIR, TEST_DATA_DIR
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create test data for fraud detection')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--fraud-rate', type=float, default=0.002,
                       help='Fraud rate (default: 0.2%)')
    parser.add_argument('--output', type=str, default='test_data.csv',
                       help='Output filename')
    parser.add_argument('--realistic', action='store_true',
                       help='Generate more realistic patterns')
    
    return parser.parse_args()

def generate_normal_transaction():
    """Generate a normal transaction"""
    # Normal business hours (8 AM - 6 PM)
    time = np.random.uniform(28800, 64800)
    
    # Normal amounts (log-normal distribution)
    amount = np.random.lognormal(mean=4, sigma=1.5)
    amount = max(1, min(amount, 1000))  # Cap at $1000
    
    # PCA features (relatively normal distribution)
    v_features = np.random.randn(28) * 1.5
    
    return time, amount, v_features

def generate_fraud_transaction():
    """Generate a fraudulent transaction"""
    patterns = ['high_amount', 'odd_time', 'unusual_pattern']
    pattern = np.random.choice(patterns)
    
    if pattern == 'high_amount':
        # High amount transactions
        time = np.random.uniform(0, 86400)  # Any time
        amount = np.random.lognormal(mean=7, sigma=1)  # Higher amounts
        amount = max(1000, min(amount, 25000))
        v_features = np.random.randn(28) * 3  # More extreme PCA values
        
    elif pattern == 'odd_time':
        # Late night/early morning transactions
        if np.random.random() < 0.5:
            time = np.random.uniform(0, 21600)  # Midnight to 6 AM
        else:
            time = np.random.uniform(79200, 86400)  # 10 PM to Midnight
        amount = np.random.lognormal(mean=5.5, sigma=1.2)
        amount = max(100, min(amount, 5000))
        v_features = np.random.randn(28) * 2.5
        
    else:  # unusual_pattern
        # Unusual transaction patterns
        time = np.random.uniform(0, 86400)
        amount = np.random.lognormal(mean=6, sigma=1.5)
        amount = max(500, min(amount, 10000))
        # Create more extreme PCA patterns
        v_features = np.random.randn(28) * 4
        # Add some correlations to make it more suspicious
        v_features[0] = v_features[1] * 2 + np.random.randn() * 0.5
        v_features[2] = -v_features[3] * 1.5 + np.random.randn() * 0.5
    
    return time, amount, v_features

def generate_realistic_dataset(n_samples, fraud_rate):
    """Generate a realistic dataset with temporal patterns"""
    fraud_logger.info(f"Generating {n_samples} samples with {fraud_rate:.1%} fraud rate...")
    
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        time, amount, v_features = generate_normal_transaction()
        
        row = {
            'Time': time,
            'Amount': amount,
            'Class': 0
        }
        
        # Add V features
        for j, v_val in enumerate(v_features, 1):
            row[f'V{j}'] = v_val
            
        data.append(row)
    
    # Generate fraudulent transactions
    for i in range(n_fraud):
        time, amount, v_features = generate_fraud_transaction()
        
        row = {
            'Time': time,
            'Amount': amount,
            'Class': 1
        }
        
        # Add V features
        for j, v_val in enumerate(v_features, 1):
            row[f'V{j}'] = v_val
            
        data.append(row)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Sort by time to make it more realistic
    df = df.sort_values('Time').reset_index(drop=True)
    
    fraud_logger.info(f"Generated dataset with {len(df)} transactions")
    fraud_logger.info(f"Fraud transactions: {df['Class'].sum()} ({df['Class'].mean():.1%})")
    fraud_logger.info(f"Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    
    return df

def generate_simple_dataset(n_samples, fraud_rate):
    """Generate a simple synthetic dataset"""
    fraud_logger.info(f"Generating simple dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate basic features
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),  # 2 days in seconds
        'Amount': np.random.lognormal(4, 1.5, n_samples),
        'Class': np.random.binomial(1, fraud_rate, n_samples)
    }
    
    # Ensure amounts are positive
    data['Amount'] = np.maximum(data['Amount'], 1)
    
    # Generate V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples) * 2
    
    # Make fraud transactions slightly different
    fraud_mask = data['Class'] == 1
    if fraud_mask.sum() > 0:
        # Higher amounts for fraud
        data['Amount'][fraud_mask] *= np.random.uniform(2, 5, fraud_mask.sum())
        
        # More extreme V features for fraud
        for i in range(1, 29):
            data[f'V{i}'][fraud_mask] *= np.random.uniform(1.5, 3, fraud_mask.sum())
    
    df = pd.DataFrame(data)
    return df

def validate_dataset(df):
    """Validate the generated dataset"""
    issues = []
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        issues.append("Dataset contains missing values")
    
    # Check class distribution
    fraud_rate = df['Class'].mean()
    if fraud_rate == 0:
        issues.append("No fraud transactions generated")
    elif fraud_rate > 0.1:
        issues.append(f"Fraud rate too high: {fraud_rate:.1%}")
    
    # Check feature ranges
    if df['Amount'].min() <= 0:
        issues.append("Invalid transaction amounts (<=0)")
    
    if df['Time'].min() < 0:
        issues.append("Invalid time values (<0)")
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        issues.append("Dataset contains infinite values")
    
    return issues

def create_train_test_split(df, test_size=0.2):
    """Create train/test split maintaining fraud ratio"""
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain fraud ratio
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['Class']
    )
    
    return train_df, test_df

def main():
    """Main test data creation script"""
    args = parse_arguments()
    
    print("FraudGuard AI - Test Data Creation Script")
    print("========================================")
    
    # Create output directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Generate dataset
    if args.realistic:
        df = generate_realistic_dataset(args.samples, args.fraud_rate)
    else:
        df = generate_simple_dataset(args.samples, args.fraud_rate)
    
    # Validate dataset
    issues = validate_dataset(df)
    if issues:
        fraud_logger.warning("Dataset validation issues:")
        for issue in issues:
            fraud_logger.warning(f"  - {issue}")
    else:
        fraud_logger.info("Dataset validation passed")
    
    # Save full dataset
    output_path = TEST_DATA_DIR / args.output
    df.to_csv(output_path, index=False)
    fraud_logger.info(f"Saved full dataset to: {output_path}")
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    fraud_logger.info(f"Saved training data to: {train_path}")
    fraud_logger.info(f"Saved test data to: {test_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Total samples: {len(df):,}")
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Fraud rate: {df['Class'].mean():.3%}")
    print(f"Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():,.2f}")
    print(f"Time range: {df['Time'].min():.0f} - {df['Time'].max():.0f} seconds")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"V features range: {df[[f'V{i}' for i in range(1, 29)]].min().min():.2f} to {df[[f'V{i}' for i in range(1, 29)]].max().max():.2f}")
    
    print(f"\nFiles created:")
    print(f"  - Full dataset: {output_path}")
    print(f"  - Training set: {train_path}")
    print(f"  - Test set: {test_path}")

if __name__ == "__main__":
    main()