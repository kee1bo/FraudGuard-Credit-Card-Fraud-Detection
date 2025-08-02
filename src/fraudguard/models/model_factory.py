from typing import Dict, Any
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .ensemble_model import EnsembleModel

class ModelFactory:
    """Factory class for creating different model types"""
    
    _models = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'ensemble': EnsembleModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs):
        """Create a model instance"""
        if model_type not in cls._models:
            raise ValueError(f"Model type '{model_type}' not supported. "
                           f"Available models: {list(cls._models.keys())}")
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types"""
        return list(cls._models.keys())
