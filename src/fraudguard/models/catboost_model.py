from catboost import CatBoostClassifier
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """CatBoost implementation for fraud detection"""
    
    def __init__(self, **kwargs):
        super().__init__("CatBoost", **kwargs)
        self.model = self._create_model()
    
    def _create_model(self):
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_state': 42,
            'auto_class_weights': 'Balanced',
            'verbose': False
        }
        params = {**default_params, **self.training_config}
        return CatBoostClassifier(**params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        self.is_trained = True
        return self