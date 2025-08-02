# src/fraudguard/components/data_ingestion.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException
from fraudguard.constants.constants import *

class DataIngestion:
    """Handle data loading and initial splitting"""
    
    def __init__(self, config):
        self.config = config
        
    def initiate_data_ingestion(self):
        """Load and split the dataset"""
        try:
            fraud_logger.info("Starting data ingestion...")
            
            # Create improved synthetic data
            df = self._create_improved_synthetic_data()
            
            fraud_logger.info(f"Dataset shape: {df.shape}")
            fraud_logger.info(f"Fraud ratio: {df['Class'].mean():.4f}")
            
            # Split the data
            train_set, test_set = train_test_split(
                df, 
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=df['Class']
            )
            
            # Save the datasets
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            train_set.to_csv(RAW_DATA_DIR / "train.csv", index=False)
            test_set.to_csv(RAW_DATA_DIR / "test.csv", index=False)
            
            fraud_logger.info("Data ingestion completed successfully")
            
            return (
                str(RAW_DATA_DIR / "train.csv"),
                str(RAW_DATA_DIR / "test.csv")
            )
            
        except Exception as e:
            raise FraudGuardException(f"Data ingestion failed: {str(e)}", sys)
    
    def _create_improved_synthetic_data(self):
        """Create improved synthetic data with clear fraud patterns"""
        np.random.seed(42)
        n_samples = 15000  # Increase sample size
        fraud_rate = 0.10  # 10% fraud rate for better learning
        
        # Generate normal transactions
        n_normal = int(n_samples * (1 - fraud_rate))
        n_fraud = n_samples - n_normal
        
        # Normal transactions: centered around 0 with small variance
        X_normal = np.random.normal(0, 1, (n_normal, 30))
        y_normal = np.zeros(n_normal)
        
        # Fraudulent transactions: clearly different patterns
        X_fraud = np.random.normal(0, 1, (n_fraud, 30))
        y_fraud = np.ones(n_fraud)
        
        # Make fraud transactions clearly distinguishable
        for i in range(n_fraud):
            # Pattern 1: High values in first 10 features
            if i % 3 == 0:
                X_fraud[i, :10] += np.random.normal(4, 1, 10)
            # Pattern 2: Low values in middle features
            elif i % 3 == 1:
                X_fraud[i, 10:20] -= np.random.normal(3, 1, 10)
            # Pattern 3: Extreme values in last features
            else:
                X_fraud[i, 20:] *= np.random.uniform(3, 6, 10)
        
        # Combine data
        X = np.vstack([X_normal, X_fraud])
        y = np.concatenate([y_normal, y_fraud])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y.astype(int)
        
        # Make realistic amounts
        df['Amount'] = np.abs(df['Amount']) * 150 + 20
        
        # Make fraud amounts significantly higher
        fraud_mask = df['Class'] == 1
        df.loc[fraud_mask, 'Amount'] *= np.random.uniform(3, 10, fraud_mask.sum())
        
        # Make realistic times
        df['Time'] = np.random.randint(0, 172800, n_samples)
        
        # Make fraud more likely at unusual times
        unusual_times = list(range(0, 21600)) + list(range(79200, 86400))  # Late night
        df.loc[fraud_mask, 'Time'] = np.random.choice(unusual_times, size=fraud_mask.sum())
        
        fraud_logger.info(f"Created improved synthetic data:")
        fraud_logger.info(f"  Total samples: {len(df)}")
        fraud_logger.info(f"  Fraud cases: {fraud_mask.sum()} ({fraud_mask.mean():.1%})")
        fraud_logger.info(f"  Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
        
        return df
