# src/fraudguard/models/xgboost_model.py

import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost implementation for fraud detection"""
    
    def __init__(self, **kwargs):
        super().__init__("XGBoost", **kwargs)
        self.model = self._create_model()
    
    def _create_model(self):
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,  # More trees
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': 10,  # Give more weight to fraud class
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        params = {**default_params, **self.training_config}
        return xgb.XGBClassifier(**params)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Calculate class weight for imbalanced data
        fraud_ratio = sum(y_train) / len(y_train)
        scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=False
        )
        self.is_trained = True
        return self
