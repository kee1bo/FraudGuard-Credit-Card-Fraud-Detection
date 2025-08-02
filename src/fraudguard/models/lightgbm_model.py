import lightgbm as lgb
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    """LightGBM implementation for fraud detection"""
    
    def __init__(self, **kwargs):
        super().__init__("LightGBM", **kwargs)
        self.model = self._create_model()
    
    def _create_model(self):
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'is_unbalance': True
        }
        params = {**default_params, **self.training_config}
        return lgb.LGBMClassifier(**params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(10)],
            verbose=False
        )
        self.is_trained = True
        return self