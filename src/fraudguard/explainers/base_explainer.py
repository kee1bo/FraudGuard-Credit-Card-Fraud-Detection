# src/fraudguard/explainers/base_explainer.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseExplainer(ABC):
    """Abstract base class for all explainers"""
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    def explain_instance(self, X_instance: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate explanation for a single instance"""
        pass
    
    @abstractmethod
    def generate_plot(self, explanation: Dict[str, Any]) -> str:
        """Generate visualization and return as base64 string"""
        pass
    
    def validate_input(self, X_instance: np.ndarray) -> np.ndarray:
        """Validate and preprocess input data"""
        if not isinstance(X_instance, np.ndarray):
            X_instance = np.array(X_instance)
        
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        return X_instance
    
    def get_feature_names(self, n_features: int, feature_names: List[str] = None) -> List[str]:
        """Get feature names, generating defaults if not provided"""
        if feature_names is None:
            return [f'Feature_{i}' for i in range(n_features)]
        
        if len(feature_names) != n_features:
            # Pad or truncate feature names to match n_features
            if len(feature_names) < n_features:
                feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), n_features)])
            else:
                feature_names = feature_names[:n_features]
        
        return feature_names