# src/fraudguard/models/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest implementation for fraud detection"""
    
    def __init__(self, **kwargs):
        super().__init__("Random Forest", **kwargs)
        self.model = self._create_model()
    
    def _create_model(self):
        default_params = {
            'n_estimators': 200,  # More trees
            'max_depth': 15,      # Deeper trees
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced_subsample',  # Better for imbalanced data
            'n_jobs': -1,
            'bootstrap': True,
            'oob_score': True
        }
        params = {**default_params, **self.training_config}
        return RandomForestClassifier(**params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
