import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODEL_DIR = ARTIFACTS_DIR / "models"
PREPROCESSOR_DIR = ARTIFACTS_DIR / "preprocessors"
EXPLAINER_DIR = ARTIFACTS_DIR / "explainers"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Data file paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_DATA_DIR = DATA_DIR / "test"

# Feature names for ULB dataset
ULB_FEATURE_NAMES = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
TARGET_COLUMN = 'Class'

# Model types
AVAILABLE_MODELS = [
    'logistic_regression',
    'random_forest',
    'xgboost',
    'lightgbm',
    'catboost',
    'ensemble'
]

# Evaluation metrics
CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'average_precision'
]

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'catboost': {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': False
    }
}

# Web application settings
FLASK_CONFIG = {
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key'),
    'DEBUG': os.environ.get('FLASK_DEBUG', 'True').lower() == 'true',
    'HOST': os.environ.get('FLASK_HOST', '0.0.0.0'),
    'PORT': int(os.environ.get('FLASK_PORT', 5000))
}
