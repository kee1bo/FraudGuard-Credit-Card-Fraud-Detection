"""
Intelligent Prediction Pipeline
Integrates feature mapping with existing fraud detection models to provide
end-to-end prediction from user-friendly inputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import joblib

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, MappingExplanation, 
    QualityMetrics, TimeContext, MerchantCategory, LocationRisk, SpendingPattern
)
from fraudguard.models.ensemble_mapper import EnsembleMapper
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper
from fraudguard.components.feature_assembler import FeatureVectorAssembler
from fraudguard.components.mapping_quality_monitor import MappingQualityMonitor
from fraudguard.components.confidence_scorer import ConfidenceScorer
from fraudguard.components.input_validator import InputValidator
from fraudguard.models.model_factory import ModelFactory
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException

# Try to import neural network mapper
try:
    from fraudguard.models.neural_network_mapper import NeuralNetworkMapper
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NeuralNetworkMapper = None
    NEURAL_NETWORK_AVAILABLE = False


class IntelligentPredictionPipeline:
    """End-to-end prediction pipeline using intelligent feature mapping"""
    
    def __init__(self, 
                 model_artifacts_path: str = "artifacts",
                 mapping_artifacts_path: str = "artifacts/feature_mapping",
                 default_mapper_type: str = "ensemble"):
        
        self.model_artifacts_path = Path(model_artifacts_path)
        self.mapping_artifacts_path = Path(mapping_artifacts_path)
        self.default_mapper_type = default_mapper_type
        
        # Core components
        self.feature_mappers = {}
        self.fraud_models = {}
        self.feature_assembler = FeatureVectorAssembler()
        self.quality_monitor = MappingQualityMonitor()
        self.confidence_scorer = ConfidenceScorer()
        self.input_validator = InputValidator()
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load all pipeline components"""
        try:
            fraud_logger.info("Loading intelligent prediction pipeline components...")
            
            # Load feature mappers
            self._load_feature_mappers()
            
            # Load fraud detection models
            self._load_fraud_models()
            
            # Load feature bounds for validation
            self._load_feature_bounds()
            
            # Load quality monitor reference data
            self._load_quality_references()
            
            fraud_logger.info("Intelligent prediction pipeline loaded successfully")
            
        except Exception as e:
            fraud_logger.error(f"Error loading pipeline components: {e}")
            # Continue with limited functionality
    
    def _load_feature_mappers(self):
        """Load available feature mapping models"""
        mappers_dir = self.mapping_artifacts_path / "mappers"
        
        if not mappers_dir.exists():
            fraud_logger.warning(f"Feature mappers directory not found: {mappers_dir}")
            return
        
        # Try to load different mapper types
        mapper_types = {
            'random_forest': RandomForestMapper,
            'xgboost': XGBoostMapper,
            'ensemble': EnsembleMapper
        }
        
        if NEURAL_NETWORK_AVAILABLE:
            mapper_types['neural_network'] = NeuralNetworkMapper
        
        for mapper_name, mapper_class in mapper_types.items():
            mapper_path = mappers_dir / mapper_name
            if mapper_path.exists():
                try:
                    mapper = mapper_class()
                    mapper.load_model(str(mapper_path))
                    self.feature_mappers[mapper_name] = mapper
                    fraud_logger.info(f"Loaded feature mapper: {mapper_name}")
                except Exception as e:
                    fraud_logger.warning(f"Could not load {mapper_name} mapper: {e}")
        
        if not self.feature_mappers:
            fraud_logger.warning("No feature mappers loaded - creating default mapper")
            self.feature_mappers['random_forest'] = RandomForestMapper()
    
    def _load_fraud_models(self):
        """Load existing fraud detection models"""
        models_dir = self.model_artifacts_path / "models"
        
        if not models_dir.exists():
            fraud_logger.warning(f"Models directory not found: {models_dir}")
            return
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_type = model_dir.name
                try:
                    model = ModelFactory.create_model(model_type)
                    model.load_model(str(model_dir))
                    self.fraud_models[model_type] = model
                    fraud_logger.info(f"Loaded fraud model: {model_type}")
                except Exception as e:
                    fraud_logger.warning(f"Could not load {model_type} model: {e}")
    
    def _load_feature_bounds(self):
        """Load feature bounds for validation"""
        bounds_path = self.mapping_artifacts_path / "feature_stats.pkl"
        if bounds_path.exists():
            try:
                feature_stats = joblib.load(bounds_path)
                self.feature_assembler.set_feature_bounds_from_stats(feature_stats)
                fraud_logger.info("Feature bounds loaded for validation")
            except Exception as e:
                fraud_logger.warning(f"Could not load feature bounds: {e}")
    
    def _load_quality_references(self):
        """Load reference data for quality monitoring"""
        ref_path = self.mapping_artifacts_path / "feature_stats.pkl"
        if ref_path.exists():
            self.quality_monitor.load_reference_stats(str(ref_path))
    
    def create_user_input_from_dict(self, input_data: Dict[str, Any]) -> UserTransactionInput:
        """
        Create UserTransactionInput from dictionary data
        
        Args:
            input_data: Dictionary with user input data
            
        Returns:
            UserTransactionInput object
        """
        try:
            # Create time context
            time_context = TimeContext(
                hour_of_day=int(input_data['hour_of_day']),
                day_of_week=int(input_data['day_of_week']),
                is_weekend=int(input_data['day_of_week']) >= 5,
                is_holiday=input_data.get('is_holiday', False)
            )
            
            # Create user input
            user_input = UserTransactionInput(
                transaction_amount=float(input_data['transaction_amount']),
                merchant_category=MerchantCategory(input_data['merchant_category']),
                time_context=time_context,
                location_risk=LocationRisk(input_data['location_risk']),
                spending_pattern=SpendingPattern(input_data['spending_pattern'])
            )
            
            return user_input
            
        except Exception as e:
            raise FraudGuardException(f"Invalid input data: {str(e)}")
    
    def predict_intelligent(self, 
                          input_data: Dict[str, Any],
                          mapper_type: Optional[str] = None,
                          fraud_model_type: str = 'xgboost',
                          include_explanation: bool = True) -> Dict[str, Any]:
        """
        Make intelligent fraud prediction from user-friendly inputs
        
        Args:
            input_data: Dictionary with user input data
            mapper_type: Type of feature mapper to use (None for default)
            fraud_model_type: Type of fraud model to use
            include_explanation: Whether to include explanations
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Create user input object
            user_input = self.create_user_input_from_dict(input_data)
            
            # Validate input
            validation_result = self.input_validator.validate_user_input(user_input)
            if not validation_result.is_valid:
                return {
                    'error': 'Input validation failed',
                    'validation_errors': validation_result.errors,
                    'validation_warnings': validation_result.warnings,
                    'suggestions': validation_result.suggestions
                }
            
            # Select feature mapper
            mapper_type = mapper_type or self.default_mapper_type
            if mapper_type not in self.feature_mappers:
                available_mappers = list(self.feature_mappers.keys())
                if available_mappers:
                    mapper_type = available_mappers[0]
                    fraud_logger.warning(f"Requested mapper {mapper_type} not available, using {mapper_type}")
                else:
                    raise FraudGuardException("No feature mappers available")
            
            mapper = self.feature_mappers[mapper_type]
            
            # Map features
            mapped_pca = mapper.predict_single(user_input)
            
            # Get uncertainty if available
            uncertainty_estimates = None
            if hasattr(mapper, 'predict_with_uncertainty'):
                try:
                    _, uncertainty_estimates = mapper.predict_with_uncertainty(
                        mapper._convert_user_input_to_features(user_input).reshape(1, -1)
                    )
                    uncertainty_estimates = uncertainty_estimates[0]
                except Exception as e:
                    fraud_logger.warning(f"Could not get uncertainty estimates: {e}")
            
            # Assemble complete feature vector
            feature_vector, quality_metrics, validation_messages = self.feature_assembler.validate_and_correct_vector(
                user_input, mapped_pca, uncertainty_estimates
            )
            
            # Calculate confidence score
            confidence_score, confidence_factors = self.confidence_scorer.calculate_overall_confidence(
                user_input, mapped_pca, uncertainty_estimates
            )
            
            # Select fraud model
            if fraud_model_type not in self.fraud_models:
                available_models = list(self.fraud_models.keys())
                if available_models:
                    fraud_model_type = available_models[0]
                    fraud_logger.warning(f"Requested model {fraud_model_type} not available, using {fraud_model_type}")
                else:
                    raise FraudGuardException("No fraud models available")
            
            fraud_model = self.fraud_models[fraud_model_type]
            
            # Make fraud prediction
            prediction = fraud_model.predict(feature_vector.reshape(1, -1))[0]
            probabilities = fraud_model.predict_proba(feature_vector.reshape(1, -1))[0]
            
            # Create mapping explanation
            mapping_explanation = None
            if include_explanation:
                mapping_explanation = self._create_mapping_explanation(
                    user_input, mapped_pca, mapper, confidence_factors
                )
            
            # Create result
            result = {
                'prediction': int(prediction),
                'fraud_probability': float(probabilities[1]),
                'normal_probability': float(probabilities[0]),
                'risk_score': float(probabilities[1] * 100),
                'mapping_confidence': confidence_score,
                'confidence_level': self.confidence_scorer.get_confidence_level(confidence_score),
                'model_used': fraud_model_type,
                'mapper_used': mapper_type,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'validation_warnings': validation_result.warnings,
                'quality_metrics': {
                    'correlation_preservation': quality_metrics.correlation_preservation,
                    'distribution_similarity': quality_metrics.distribution_similarity,
                    'prediction_consistency': quality_metrics.prediction_consistency,
                    'mapping_uncertainty': quality_metrics.mapping_uncertainty
                }
            }
            
            # Add explanation if requested
            if mapping_explanation:
                result['mapping_explanation'] = mapping_explanation
            
            # Add feature importance if available
            if hasattr(mapper, 'get_feature_importance'):
                try:
                    feature_importance = mapper.get_feature_importance()
                    if feature_importance:
                        result['feature_importance'] = feature_importance
                except Exception as e:
                    fraud_logger.warning(f"Could not get feature importance: {e}")
            
            return result
            
        except Exception as e:
            fraud_logger.error(f"Intelligent prediction failed: {e}")
            return {
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def _create_mapping_explanation(self, 
                                  user_input: UserTransactionInput,
                                  mapped_pca: np.ndarray,
                                  mapper,
                                  confidence_factors) -> Dict[str, Any]:
        """Create explanation of the mapping process"""
        try:
            # Get input contributions (simplified)
            input_contributions = {
                'transaction_amount': 0.25,  # Placeholder values
                'merchant_category': 0.20,
                'time_context': 0.15,
                'location_risk': 0.20,
                'spending_pattern': 0.20
            }
            
            # Get PCA estimates
            pca_estimates = {f'V{i+1}': float(mapped_pca[i]) for i in range(len(mapped_pca))}
            
            # Create confidence intervals (simplified)
            confidence_intervals = {
                f'V{i+1}': (float(mapped_pca[i] - 0.5), float(mapped_pca[i] + 0.5))
                for i in range(len(mapped_pca))
            }
            
            # Create business interpretation
            business_interpretation = self._generate_business_interpretation(user_input, confidence_factors)
            
            return {
                'input_contributions': input_contributions,
                'pca_estimates': pca_estimates,
                'confidence_intervals': confidence_intervals,
                'business_interpretation': business_interpretation,
                'mapping_method': mapper.model_name
            }
            
        except Exception as e:
            fraud_logger.warning(f"Could not create mapping explanation: {e}")
            return None
    
    def _generate_business_interpretation(self, 
                                        user_input: UserTransactionInput,
                                        confidence_factors) -> str:
        """Generate business-friendly interpretation of the mapping"""
        interpretation_parts = []
        
        # Amount interpretation
        amount = user_input.transaction_amount
        if amount < 50:
            interpretation_parts.append(f"Low transaction amount (${amount:.2f}) suggests routine purchase")
        elif amount > 1000:
            interpretation_parts.append(f"High transaction amount (${amount:.2f}) increases fraud risk")
        else:
            interpretation_parts.append(f"Moderate transaction amount (${amount:.2f}) is within normal range")
        
        # Merchant interpretation
        merchant = user_input.merchant_category.value
        interpretation_parts.append(f"{merchant} transactions have specific risk patterns")
        
        # Time interpretation
        hour = user_input.time_context.hour_of_day
        if hour >= 22 or hour <= 6:
            interpretation_parts.append("Late night/early morning timing increases suspicion")
        else:
            interpretation_parts.append("Transaction timing appears normal")
        
        # Location interpretation
        location = user_input.location_risk.value
        if location != 'normal':
            interpretation_parts.append(f"Unusual location ({location.replace('_', ' ')}) adds risk")
        
        # Spending pattern interpretation
        spending = user_input.spending_pattern.value
        if spending in ['suspicious', 'much_higher']:
            interpretation_parts.append(f"Spending pattern ({spending.replace('_', ' ')}) is concerning")
        
        # Confidence interpretation
        if confidence_factors.input_consistency < 0.7:
            interpretation_parts.append("Some input inconsistencies detected")
        
        return ". ".join(interpretation_parts) + "."
    
    def get_available_mappers(self) -> List[str]:
        """Get list of available feature mappers"""
        return list(self.feature_mappers.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of available fraud models"""
        return list(self.fraud_models.keys())
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data and provide suggestions
        
        Args:
            input_data: Dictionary with input data to validate
            
        Returns:
            Dictionary with validation results and suggestions
        """
        try:
            # Use input validator for partial validation
            return self.input_validator.validate_and_suggest(**input_data)
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'suggestions': {}
            }
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about the mapping pipeline"""
        stats = {
            'available_mappers': len(self.feature_mappers),
            'available_models': len(self.fraud_models),
            'mapper_types': list(self.feature_mappers.keys()),
            'model_types': list(self.fraud_models.keys()),
            'default_mapper': self.default_mapper_type
        }
        
        # Add mapper-specific stats
        for mapper_name, mapper in self.feature_mappers.items():
            if hasattr(mapper, 'metadata') and mapper.metadata:
                stats[f'{mapper_name}_metadata'] = mapper.metadata.__dict__
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pipeline"""
        health = {
            'status': 'healthy',
            'components': {},
            'issues': []
        }
        
        # Check feature mappers
        if not self.feature_mappers:
            health['status'] = 'unhealthy'
            health['issues'].append('No feature mappers available')
        else:
            health['components']['feature_mappers'] = {
                'count': len(self.feature_mappers),
                'types': list(self.feature_mappers.keys())
            }
        
        # Check fraud models
        if not self.fraud_models:
            health['status'] = 'degraded'
            health['issues'].append('No fraud models available')
        else:
            health['components']['fraud_models'] = {
                'count': len(self.fraud_models),
                'types': list(self.fraud_models.keys())
            }
        
        # Check other components
        health['components']['feature_assembler'] = 'loaded'
        health['components']['quality_monitor'] = 'loaded'
        health['components']['confidence_scorer'] = 'loaded'
        health['components']['input_validator'] = 'loaded'
        
        return health