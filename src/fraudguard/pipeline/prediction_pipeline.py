# src/fraudguard/pipeline/prediction_pipeline.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from fraudguard.models.model_factory import ModelFactory
from fraudguard.explainers.shap_explainer import SHAPExplainer
from fraudguard.utils.common import load_object, load_json
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException
from fraudguard.constants.constants import *

class PredictionPipeline:
    """End-to-end prediction pipeline with explanations"""
    
    def __init__(self, model_artifacts_path: str = "artifacts"):
        self.model_artifacts_path = Path(model_artifacts_path)
        self.preprocessor = None
        self.models = {}
        self.explainers = {}
        self.feature_names = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all necessary artifacts"""
        try:
            # Load preprocessor
            preprocessor_path = self.model_artifacts_path / "preprocessors" / "scaler.pkl"
            if preprocessor_path.exists():
                self.preprocessor = load_object(str(preprocessor_path))
                fraud_logger.info("Preprocessor loaded successfully")
            else:
                fraud_logger.warning(f"Preprocessor not found at {preprocessor_path}")
            
            # Load feature names
            feature_info_path = self.model_artifacts_path / "feature_names.pkl"
            if feature_info_path.exists():
                feature_info = load_object(str(feature_info_path))
                self.feature_names = feature_info['feature_names']
                fraud_logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            else:
                # Default feature names for ULB dataset
                self.feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                fraud_logger.warning("Using default feature names")
            
            # Load all models
            self._load_models()
            
        except Exception as e:
            fraud_logger.error(f"Error loading artifacts: {e}")
    
    def _load_models(self):
        """Load all available models and their explainers"""
        models_dir = self.model_artifacts_path / "models"
        if not models_dir.exists():
            fraud_logger.warning("Models directory not found")
            return
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_type = model_dir.name
                try:
                    # Load model
                    model = ModelFactory.create_model(model_type)
                    model.load_model(str(model_dir))
                    self.models[model_type] = model
                    fraud_logger.info(f"Loaded model: {model_type}")
                    
                    # Try to load corresponding explainer
                    self._load_explainer(model_type, model)
                    
                except Exception as e:
                    fraud_logger.warning(f"Could not load {model_type}: {e}")
    
    def _load_explainer(self, model_type: str, model):
        """Load explainer for a specific model"""
        explainer_dir = self.model_artifacts_path / "explainers"
        explainer_path = explainer_dir / f"{model_type}_shap.pkl"
        
        if explainer_path.exists():
            try:
                self.explainers[model_type] = load_object(str(explainer_path))
                fraud_logger.info(f"Loaded explainer for {model_type}")
            except Exception as e:
                fraud_logger.warning(f"Could not load explainer for {model_type}: {e}")
                # Create a new explainer if loading fails
                self._create_explainer(model_type, model)
        else:
            fraud_logger.warning(f"No explainer found for {model_type}")
            # Create a new explainer
            self._create_explainer(model_type, model)
    
    def _create_explainer(self, model_type: str, model):
        """Create a new explainer for a model"""
        try:
            # Check if model supports SHAP (tree-based models)
            if model_type in ['xgboost', 'random_forest', 'catboost']:
                # Create dummy background data for explainer
                background_data = np.random.randn(100, len(self.feature_names))
                explainer = SHAPExplainer(model, background_data)
                self.explainers[model_type] = explainer
                
                # Save the explainer
                explainer_dir = self.model_artifacts_path / "explainers"
                explainer_dir.mkdir(exist_ok=True)
                explainer_path = explainer_dir / f"{model_type}_shap.pkl"
                # Note: We can't save SHAP explainers easily, so we'll create them on demand
                
                fraud_logger.info(f"Created explainer for {model_type}")
        except Exception as e:
            fraud_logger.warning(f"Could not create explainer for {model_type}: {e}")
    
    def predict_single_transaction(self, 
                                 transaction_data: Dict[str, Any], 
                                 model_type: str = 'xgboost',
                                 include_explanation: bool = True) -> Dict[str, Any]:
        """Make prediction for a single transaction"""
        try:
            if model_type not in self.models:
                available = list(self.models.keys())
                raise ValueError(f"Model {model_type} not available. Available models: {available}")
            
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Preprocess if preprocessor is available
            if self.preprocessor:
                X_processed = self.preprocessor.transform(df)
            else:
                X_processed = df.values
            
            # Make prediction
            model = self.models[model_type]
            prediction = model.predict(X_processed)[0]
            probability = model.predict_proba(X_processed)[0]
            
            result = {
                'prediction': int(prediction),
                'probability_fraud': float(probability[1]),
                'probability_normal': float(probability[0]),
                'risk_score': float(probability[1] * 100),
                'model_used': model_type,
                'transaction_data': transaction_data
            }
            
            # Add explanation if requested and available
            if include_explanation:
                explanation_result = self._generate_explanation(model_type, X_processed[0], df.iloc[0])
                result.update(explanation_result)
            
            return result
            
        except Exception as e:
            fraud_logger.error(f"Prediction failed: {e}")
            raise FraudGuardException(f"Prediction failed: {str(e)}")
    
    def _generate_explanation(self, model_type: str, X_processed: np.ndarray, original_data: pd.Series):
        """Generate explanation for a prediction"""
        explanation_result = {}
        
        try:
            if model_type in self.explainers:
                explainer = self.explainers[model_type]
                explanation = explainer.explain_instance(X_processed, self.feature_names)
                explanation_result['explanation'] = explanation
                
                # Generate waterfall plot
                try:
                    plot_data = explainer.generate_plot(explanation)
                    explanation_result['waterfall_plot'] = plot_data
                except Exception as e:
                    fraud_logger.warning(f"Could not generate plot: {e}")
                    explanation_result['waterfall_plot'] = None
            else:
                # Create basic feature importance if no explainer available
                explanation_result['explanation'] = self._create_basic_explanation(
                    model_type, X_processed, original_data
                )
                explanation_result['waterfall_plot'] = None
                
        except Exception as e:
            fraud_logger.warning(f"Could not generate explanation: {e}")
            explanation_result['explanation'] = None
            explanation_result['waterfall_plot'] = None
        
        return explanation_result
    
    def _create_basic_explanation(self, model_type: str, X_processed: np.ndarray, original_data: pd.Series):
        """Create basic explanation when SHAP is not available"""
        model = self.models[model_type]
        
        # Get feature importance if available
        if hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        elif hasattr(model.model, 'coef_'):
            importance = np.abs(model.model.coef_[0])
        else:
            # Create dummy importance
            importance = np.random.random(len(self.feature_names))
        
        # Create explanation in SHAP-like format
        explanation = {
            'shap_values': (importance * X_processed).tolist(),
            'base_value': 0.5,  # Dummy base value
            'instance_values': X_processed.tolist(),
            'feature_names': self.feature_names
        }
        
        return explanation
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_model_metrics(self, model_type: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not available")
        
        return self.models[model_type].metrics