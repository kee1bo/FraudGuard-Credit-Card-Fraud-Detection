from abc import ABC, abstractmethod
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class BaseModel(ABC):
    """Abstract base class for all fraud detection models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_config = kwargs
        self.metrics = {}
        
    @abstractmethod
    def _create_model(self):
        """Create the actual model instance"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model"""
        pass
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'model_name': self.model_name
        }
        return self.metrics
    
    def save_model(self, path: str):
        """Save model and metadata"""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path / "model.pkl")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_config': self.training_config,
            'metrics': self.metrics
        }
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, path: str):
        """Load model and metadata"""
        model_path = Path(path)
        
        # Load model
        self.model = joblib.load(model_path / "model.pkl")
        
        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.model_name = metadata['model_name']
            self.is_trained = metadata['is_trained']
            self.training_config = metadata['training_config']
            self.metrics = metadata.get('metrics', {})