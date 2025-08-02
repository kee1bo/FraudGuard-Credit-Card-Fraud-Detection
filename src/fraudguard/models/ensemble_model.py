from sklearn.ensemble import VotingClassifier
from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .random_forest import RandomForestModel
from .lightgbm_model import LightGBMModel

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self, **kwargs):
        super().__init__("Ensemble", **kwargs)
        self.base_models = {}
        self.model = self._create_model()
    
    def _create_model(self):
        # Initialize base models
        self.base_models = {
            'xgb': XGBoostModel().model,
            'rf': RandomForestModel().model,
            'lgb': LightGBMModel().model
        }
        
        # Create voting classifier
        estimators = [(name, model) for name, model in self.base_models.items()]
        return VotingClassifier(estimators=estimators, voting='soft')
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self