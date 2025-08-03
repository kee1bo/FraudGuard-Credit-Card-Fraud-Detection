"""
Base abstract class for feature mapping models.
Defines the interface for transforming user-friendly inputs to PCA components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import joblib
import json
from pathlib import Path

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, MappingModelMetadata
)


class BaseFeatureMapper(ABC):
    """Abstract base class for all feature mapping models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_config = kwargs
        self.metadata = None
        self.feature_scaler = None
        
    @abstractmethod
    def _create_model(self):
        """Create the actual mapping model instance"""
        pass
    
    @abstractmethod
    def fit(self, X_interpretable: np.ndarray, y_pca_components: np.ndarray, **kwargs):
        """
        Train mapping from interpretable features to PCA components
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            y_pca_components: Array of shape (n_samples, 28) for V1-V28
        """
        pass
    
    @abstractmethod
    def predict(self, X_interpretable: np.ndarray) -> np.ndarray:
        """
        Predict PCA component values from interpretable features
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            
        Returns:
            Array of shape (n_samples, 28) with estimated V1-V28 values
        """
        pass
    
    def predict_single(self, user_input: UserTransactionInput) -> np.ndarray:
        """
        Predict PCA components for a single user input
        
        Args:
            user_input: UserTransactionInput object
            
        Returns:
            Array of shape (28,) with estimated V1-V28 values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Convert user input to feature array
        interpretable_features = self._convert_user_input_to_features(user_input)
        
        # Predict PCA components
        pca_estimates = self.predict(interpretable_features.reshape(1, -1))
        
        return pca_estimates[0]
    
    def _convert_user_input_to_features(self, user_input: UserTransactionInput) -> np.ndarray:
        """Convert UserTransactionInput to numerical feature array"""
        features = []
        
        # Transaction amount
        features.append(user_input.transaction_amount)
        
        # Merchant category (encoded as integer)
        merchant_categories = list(user_input.merchant_category.__class__)
        merchant_encoded = merchant_categories.index(user_input.merchant_category)
        features.append(merchant_encoded)
        
        # Location risk (encoded as integer)
        location_risks = list(user_input.location_risk.__class__)
        location_encoded = location_risks.index(user_input.location_risk)
        features.append(location_encoded)
        
        # Spending pattern (encoded as integer)
        spending_patterns = list(user_input.spending_pattern.__class__)
        spending_encoded = spending_patterns.index(user_input.spending_pattern)
        features.append(spending_encoded)
        
        # Time context features
        time_ctx = user_input.time_context
        # Convert hour to sin/cos for cyclical encoding
        hour_sin = np.sin(2 * np.pi * time_ctx.hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * time_ctx.hour_of_day / 24)
        features.extend([hour_sin, hour_cos, time_ctx.day_of_week, int(time_ctx.is_weekend)])
        
        return np.array(features)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported by the model"""
        return None
    
    def evaluate_mapping_quality(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate mapping quality on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate correlation preservation (average correlation across features)
        correlations = []
        for i in range(y_test.shape[1]):
            corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'avg_correlation': avg_correlation,
            'correlation_preservation': max(0.0, avg_correlation)
        }
    
    def save_model(self, path: str):
        """Save model and metadata"""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path / "model.pkl")
        
        # Save feature scaler if exists
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, model_path / "scaler.pkl")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_config': self.training_config,
            'metadata': self.metadata.__dict__ if self.metadata else None
        }
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, path: str):
        """Load model and metadata"""
        model_path = Path(path)
        
        # Load model
        self.model = joblib.load(model_path / "model.pkl")
        
        # Load feature scaler if exists
        scaler_path = model_path / "scaler.pkl"
        if scaler_path.exists():
            self.feature_scaler = joblib.load(scaler_path)
        
        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.model_name = metadata['model_name']
            self.is_trained = metadata['is_trained']
            self.training_config = metadata['training_config']
            if metadata.get('metadata'):
                self.metadata = MappingModelMetadata(**metadata['metadata'])