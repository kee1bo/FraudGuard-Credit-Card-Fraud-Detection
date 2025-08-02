import os
from pathlib import Path
from fraudguard.utils.common import load_yaml
from fraudguard.entity.config_entity import *
from fraudguard.constants.constants import *

class ConfigurationManager:
    """Manage all configuration settings"""
    
    def __init__(self, config_filepath: str = "config.yaml"):
        try:
            self.config = load_yaml(config_filepath)
        except:
            # Use default configuration
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'data': {
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'scaling_method': 'standard',
                'handle_outliers': True
            },
            'models': {
                'train_all': True,
                'cross_validation': True
            },
            'explainability': {
                'enable_shap': True,
                'enable_lime': True
            }
        }
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        config = self.config.get('data', {})
        
        return DataIngestionConfig(
            raw_data_path=str(RAW_DATA_DIR),
            train_data_path=str(RAW_DATA_DIR / "train.csv"),
            test_data_path=str(RAW_DATA_DIR / "test.csv"),
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42)
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration"""
        config = self.config.get('preprocessing', {})
        
        return DataTransformationConfig(
            preprocessor_path=str(PREPROCESSOR_DIR / "scaler.pkl"),
            scaling_method=config.get('scaling_method', 'standard'),
            handle_outliers=config.get('handle_outliers', True),
            outlier_method=config.get('outlier_method', 'iqr')
        )
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration"""
        config = self.config.get('models', {})
        
        return ModelTrainerConfig(
            model_path=str(MODEL_DIR),
            train_all_models=config.get('train_all', True),
            cross_validation=config.get('cross_validation', True),
            cv_folds=config.get('cv_folds', 5),
            hyperparameter_tuning=config.get('hyperparameter_tuning', False)
        )
    
    def get_explainability_config(self) -> ExplainabilityConfig:
        """Get explainability configuration"""
        config = self.config.get('explainability', {})
        
        return ExplainabilityConfig(
            explainer_path=str(EXPLAINER_DIR),
            enable_shap=config.get('enable_shap', True),
            enable_lime=config.get('enable_lime', True),
            shap_background_size=config.get('shap_background_size', 100),
            lime_num_features=config.get('lime_num_features', 10)
        )
