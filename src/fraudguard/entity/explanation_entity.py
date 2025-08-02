from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class SHAPExplanation:
    shap_values: List[float]
    base_value: float
    instance_values: List[float]
    feature_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'shap_values': self.shap_values,
            'base_value': self.base_value,
            'instance_values': self.instance_values,
            'feature_names': self.feature_names
        }

@dataclass
class LIMEExplanation:
    feature_importance: Dict[str, float]
    score: float
    intercept: float
    instance_values: List[float]
    feature_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_importance': self.feature_importance,
            'score': self.score,
            'intercept': self.intercept,
            'instance_values': self.instance_values,
            'feature_names': self.feature_names
        }