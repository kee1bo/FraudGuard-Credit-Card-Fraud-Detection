"""
Feature mapping entity definitions for the Intelligent Feature Mapping Pipeline.
These entities define the data structures for transforming user-friendly inputs
into the complete 30-feature vector required by fraud detection models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class MerchantCategory(Enum):
    """Enumeration of merchant categories for user-friendly input"""
    GROCERY = "grocery"
    GAS_STATION = "gas"
    ONLINE_RETAIL = "online"
    RESTAURANT = "restaurant"
    ATM_WITHDRAWAL = "atm"
    DEPARTMENT_STORE = "department_store"
    PHARMACY = "pharmacy"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    OTHER = "other"


class LocationRisk(Enum):
    """Enumeration of location risk levels"""
    NORMAL = "normal"
    SLIGHTLY_UNUSUAL = "slightly_unusual"
    HIGHLY_UNUSUAL = "highly_unusual"
    FOREIGN_COUNTRY = "foreign"


class SpendingPattern(Enum):
    """Enumeration of spending pattern deviations"""
    TYPICAL = "typical"
    SLIGHTLY_HIGHER = "slightly_higher"
    MUCH_HIGHER = "much_higher"
    SUSPICIOUS = "suspicious"


@dataclass
class TimeContext:
    """Time context information for transaction"""
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)
    is_weekend: bool
    is_holiday: bool = False
    
    def __post_init__(self):
        if not (0 <= self.hour_of_day <= 23):
            raise ValueError("hour_of_day must be between 0 and 23")
        if not (0 <= self.day_of_week <= 6):
            raise ValueError("day_of_week must be between 0 and 6")


@dataclass
class UserTransactionInput:
    """User-friendly transaction input with 5 interpretable features"""
    transaction_amount: float
    merchant_category: MerchantCategory
    time_context: TimeContext
    location_risk: LocationRisk
    spending_pattern: SpendingPattern
    
    def __post_init__(self):
        if self.transaction_amount < 0:
            raise ValueError("transaction_amount must be non-negative")


@dataclass
class MappingExplanation:
    """Explanation of how user inputs were mapped to PCA features"""
    input_contributions: Dict[str, float]  # SHAP values for each input
    pca_estimates: Dict[str, float]        # Estimated V1-V28 values
    confidence_intervals: Dict[str, Tuple[float, float]]  # Uncertainty bounds
    business_interpretation: str           # Human-readable explanation
    mapping_method: str                    # Which mapping model was used


@dataclass
class QualityMetrics:
    """Quality assessment metrics for feature mapping"""
    correlation_preservation: float       # How well correlations are maintained (0-1)
    distribution_similarity: float        # KL divergence from ULB distribution
    prediction_consistency: float         # Consistency with fraud model expectations (0-1)
    mapping_uncertainty: float            # Average uncertainty in PCA estimates (0-1)
    confidence_score: float               # Overall mapping confidence (0-1)


@dataclass
class MappingResult:
    """Complete result of feature mapping operation"""
    feature_vector: np.ndarray            # Complete 30-feature vector
    confidence_score: float               # Mapping quality confidence (0-1)
    mapping_explanation: MappingExplanation
    quality_metrics: QualityMetrics
    processing_time_ms: float
    original_input: UserTransactionInput
    
    def get_feature_dict(self) -> Dict[str, float]:
        """Convert feature vector to dictionary with proper feature names"""
        feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        return dict(zip(feature_names, self.feature_vector))


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: Dict[str, str]  # Field -> suggestion mapping


@dataclass
class TrainingDataPoint:
    """Training data structure for mapping model training"""
    # Original ULB features
    time: float
    amount: float
    v1_to_v28: np.ndarray  # Original PCA components (28 features)
    class_label: int
    
    # Derived interpretable features (for training mapping models)
    merchant_category_encoded: int
    time_context_features: np.ndarray  # [hour_sin, hour_cos, day_of_week, is_weekend]
    location_risk_score: float
    spending_pattern_score: float
    
    def to_interpretable_features(self) -> np.ndarray:
        """Convert to feature array for mapping model training"""
        features = [
            self.amount,
            self.merchant_category_encoded,
            self.location_risk_score,
            self.spending_pattern_score
        ]
        features.extend(self.time_context_features)
        return np.array(features)


@dataclass
class MappingModelMetadata:
    """Metadata for mapping models"""
    model_name: str
    model_type: str  # 'random_forest', 'xgboost', 'neural_network', 'ensemble'
    version: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None