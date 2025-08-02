# src/fraudguard/pipeline/training_pipeline.py

import sys
import numpy as np
from fraudguard.components.data_ingestion import DataIngestion
from fraudguard.components.data_transformation import DataTransformation
from fraudguard.components.model_trainer import ModelTrainer
from fraudguard.explainers.shap_explainer import SHAPExplainer
from fraudguard.utils.common import save_object, load_yaml, create_directories
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException
from fraudguard.constants.constants import *

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, config_path="config.yaml"):
        try:
            self.config = load_yaml(config_path)
        except:
            # Use default config if file not found
            self.config = {
                'data': {'test_size': 0.2, 'random_state': 42},
                'preprocessing': {'scaling_method': 'standard'},
                'models': {'train_all': True},
                'explainability': {'enable_shap': True}
            }
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for artifacts"""
        directories = [
            str(ARTIFACTS_DIR),
            str(DATA_DIR),
            str(MODEL_DIR),
            str(PREPROCESSOR_DIR),
            str(EXPLAINER_DIR),
            str(REPORTS_DIR),
            str(RAW_DATA_DIR),
            str(PROCESSED_DATA_DIR)
        ]
        create_directories(directories)
    
    def run_pipeline(self):
        """Execute the complete training pipeline"""
        try:
            fraud_logger.info("Starting training pipeline...")
            
            # Step 1: Data Ingestion
            fraud_logger.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion(self.config.get('data', {}))
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            fraud_logger.info("Step 2: Data Transformation")
            data_transformation = DataTransformation(self.config.get('preprocessing', {}))
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            
            # Step 3: Model Training
            fraud_logger.info("Step 3: Model Training")
            model_trainer = ModelTrainer(self.config.get('models', {}))
            trained_models, model_results, best_model_type = model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )
            
            # Step 4: Create Explainers
            if self.config.get('explainability', {}).get('enable_shap', True):
                fraud_logger.info("Step 4: Creating Explainers")
                self._create_explainers(trained_models, X_train[:100])
            
            fraud_logger.info("Training pipeline completed successfully!")
            
            return {
                'trained_models': trained_models,
                'model_results': model_results,
                'best_model': best_model_type
            }
            
        except Exception as e:
            raise FraudGuardException(f"Training pipeline failed: {str(e)}", sys)
    
    def _create_explainers(self, models, X_background):
        """Create and save explainers for trained models"""
        try:
            fraud_logger.info("Creating explainers...")
            
            # Ensure X_background is numpy array
            if hasattr(X_background, 'values'):
                X_background = X_background.values
            elif not isinstance(X_background, np.ndarray):
                X_background = np.array(X_background)
            
            for model_name, model in models.items():
                try:
                    fraud_logger.info(f"Creating explainer for {model_name}...")
                    
                    # Create SHAP explainer for supported models
                    if model_name in ['xgboost', 'random_forest', 'lightgbm', 'catboost']:
                        explainer = SHAPExplainer(model, X_background)
                        
                        # Test the explainer with a sample
                        try:
                            test_explanation = explainer.explain_instance(
                                X_background[0], 
                                feature_names=['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                            )
                            fraud_logger.info(f"Explainer test successful for {model_name}")
                        except Exception as e:
                            fraud_logger.warning(f"Explainer test failed for {model_name}: {e}")
                        
                        # Save explainer (though SHAP explainers are hard to serialize)
                        # We'll create a metadata file instead
                        explainer_metadata = {
                            'model_name': model_name,
                            'explainer_type': 'SHAP',
                            'background_shape': X_background.shape,
                            'created': True
                        }
                        
                        metadata_path = EXPLAINER_DIR / f"{model_name}_metadata.json"
                        save_object(explainer_metadata, str(metadata_path))
                        
                        fraud_logger.info(f"Created explainer metadata for {model_name}")
                        
                except Exception as e:
                    fraud_logger.error(f"Error creating explainer for {model_name}: {e}")
                    continue
            
        except Exception as e:
            fraud_logger.error(f"Error creating explainers: {e}")
    
    def _save_explainer_background_data(self, X_background):
        """Save background data for explainer creation"""
        try:
            background_path = EXPLAINER_DIR / "background_data.pkl"
            save_object(X_background, str(background_path))
            fraud_logger.info("Saved background data for explainers")
        except Exception as e:
            fraud_logger.warning(f"Could not save background data: {e}")