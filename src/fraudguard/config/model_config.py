from typing import Dict, Any
from fraudguard.constants.constants import DEFAULT_MODEL_PARAMS

class ModelConfiguration:
    """Model-specific configuration management"""
    
    def __init__(self):
        self.model_params = DEFAULT_MODEL_PARAMS.copy()
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        return self.model_params.get(model_type, {})
    
    def update_model_params(self, model_type: str, params: Dict[str, Any]):
        """Update parameters for a specific model"""
        if model_type in self.model_params:
            self.model_params[model_type].update(params)
        else:
            self.model_params[model_type] = params
    
    def get_hyperparameter_grid(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameter grid for tuning"""
        grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 62, 124]
            },
            'catboost': {
                'iterations': [50, 100, 200],
                'depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        return grids.get(model_type, {})