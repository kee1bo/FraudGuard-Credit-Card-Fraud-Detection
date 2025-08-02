import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from fraudguard.logger import fraud_logger

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering for credit card data"""
    
    def __init__(self, time_features=True, amount_features=True, interaction_features=False):
        self.time_features = time_features
        self.amount_features = amount_features
        self.interaction_features = interaction_features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Apply feature engineering transformations"""
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier manipulation
            feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
        
        # Time-based features
        if self.time_features and 'Time' in X_df.columns:
            X_df['Hour'] = (X_df['Time'] / 3600) % 24
            X_df['Day'] = (X_df['Time'] / 86400) % 7
            X_df['Time_sin'] = np.sin(2 * np.pi * X_df['Hour'] / 24)
            X_df['Time_cos'] = np.cos(2 * np.pi * X_df['Hour'] / 24)
        
        # Amount-based features
        if self.amount_features and 'Amount' in X_df.columns:
            X_df['Amount_log'] = np.log1p(X_df['Amount'])
            X_df['Amount_sqrt'] = np.sqrt(X_df['Amount'])
            
            # Amount binning
            X_df['Amount_bin'] = pd.cut(X_df['Amount'], bins=10, labels=False)
        
        # Interaction features (if enabled)
        if self.interaction_features:
            # Create some basic interaction features
            for i in range(1, 6):  # First 5 V features
                for j in range(i+1, 6):
                    X_df[f'V{i}_V{j}_interaction'] = X_df[f'V{i}'] * X_df[f'V{j}']
        
        return X_df.values