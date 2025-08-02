from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression implementation for fraud detection"""
    
    def __init__(self, **kwargs):
        super().__init__("Logistic Regression", **kwargs)
        self.model = self._create_model()
    
    def _create_model(self):
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        params = {**default_params, **self.training_config}
        return LogisticRegression(**params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
