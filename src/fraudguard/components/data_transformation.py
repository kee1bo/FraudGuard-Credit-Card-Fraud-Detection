import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from fraudguard.utils.common import save_object
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException
from fraudguard.constants.constants import *
import sys

class DataTransformation:
    """Handle data preprocessing and feature engineering"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        
    def get_data_transformer(self):
        """Create preprocessing pipeline"""
        try:
            scaling_method = self.config.get('scaling_method', 'standard')
            
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
            
            fraud_logger.info(f"Created {scaling_method} scaler")
            return scaler
            
        except Exception as e:
            raise FraudGuardException(f"Error creating preprocessor: {str(e)}", sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """Transform the training and test datasets"""
        try:
            fraud_logger.info("Starting data transformation...")
            
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Separate features and target
            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]
            
            # Get preprocessor
            self.preprocessor = self.get_data_transformer()
            
            # Transform the data
            X_train_scaled = self.preprocessor.fit_transform(X_train)
            X_test_scaled = self.preprocessor.transform(X_test)
            
            # Save preprocessor
            save_object(self.preprocessor, PREPROCESSOR_DIR / "scaler.pkl")
            
            # Save feature names
            feature_info = {
                'feature_names': list(X_train.columns),
                'n_features': len(X_train.columns)
            }
            save_object(feature_info, ARTIFACTS_DIR / "feature_names.pkl")
            
            fraud_logger.info("Data transformation completed successfully")
            
            return (
                X_train_scaled, X_test_scaled,
                y_train.values, y_test.values
            )
            
        except Exception as e:
            raise FraudGuardException(f"Data transformation failed: {str(e)}", sys)