from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class TransactionData:
    time: float
    amount: float
    v_features: List[float]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        result = {
            'Time': self.time,
            'Amount': self.amount
        }
        for i, value in enumerate(self.v_features, 1):
            result[f'V{i}'] = value
        return result

@dataclass
class PredictionResult:
    prediction: int
    probability_fraud: float
    probability_normal: float
    risk_score: float
    model_used: str
    transaction_data: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'prediction': self.prediction,
            'probability_fraud': self.probability_fraud,
            'probability_normal': self.probability_normal,
            'risk_score': self.risk_score,
            'model_used': self.model_used,
            'transaction_data': self.transaction_data,
            'explanation': self.explanation
        }