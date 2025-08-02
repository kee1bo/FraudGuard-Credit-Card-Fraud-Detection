from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class DataTransformationConfig:
    preprocessor_path: str
    scaling_method: str = "standard"
    handle_outliers: bool = True
    outlier_method: str = "iqr"

@dataclass
class ModelTrainerConfig:
    model_path: str
    train_all_models: bool = True
    cross_validation: bool = True
    cv_folds: int = 5
    hyperparameter_tuning: bool = False

@dataclass
class ExplainabilityConfig:
    explainer_path: str
    enable_shap: bool = True
    enable_lime: bool = True
    shap_background_size: int = 100
    lime_num_features: int = 10