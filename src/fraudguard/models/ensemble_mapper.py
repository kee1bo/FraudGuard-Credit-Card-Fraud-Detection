"""
Ensemble Feature Mapper
Combines multiple mapping approaches for robust feature mapping.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
from pathlib import Path
import joblib

from fraudguard.models.base_feature_mapper import BaseFeatureMapper
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper
from fraudguard.entity.feature_mapping_entity import (
    MappingModelMetadata, UserTransactionInput
)
from fraudguard.logger import fraud_logger

# Conditional import for neural network
try:
    from fraudguard.models.neural_network_mapper import NeuralNetworkMapper
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NeuralNetworkMapper = None
    NEURAL_NETWORK_AVAILABLE = False


class EnsembleMapper(BaseFeatureMapper):
    """Ensemble of multiple mapping approaches for robustness"""
    
    def __init__(self, 
                 ensemble_methods: List[str] = None,
                 ensemble_weights: Optional[Dict[str, float]] = None,
                 combination_method: str = 'weighted_average',
                 confidence_threshold: float = 0.7,
                 fallback_method: str = 'random_forest',
                 **kwargs):
        super().__init__("ensemble_mapper", **kwargs)
        
        # Default ensemble methods
        if ensemble_methods is None:
            ensemble_methods = ['random_forest', 'xgboost']
            if NEURAL_NETWORK_AVAILABLE:
                ensemble_methods.append('neural_network')
        
        self.ensemble_methods = ensemble_methods
        self.ensemble_weights = ensemble_weights or {}
        self.combination_method = combination_method
        self.confidence_threshold = confidence_threshold
        self.fallback_method = fallback_method
        
        # Initialize individual mappers
        self.mappers = {}
        self.mapper_performances = {}
        self.is_trained = False
    
    def _create_model(self):
        """Create ensemble model (not used directly, but required by base class)"""
        # Ensemble doesn't have a single model, it manages multiple mappers
        return None
        
    def _create_mappers(self):
        """Create individual mapping models"""
        for method in self.ensemble_methods:
            if method == 'random_forest':
                self.mappers[method] = RandomForestMapper(
                    n_estimators=100,
                    random_state=42
                )
            elif method == 'xgboost':
                self.mappers[method] = XGBoostMapper(
                    n_estimators=100,
                    random_state=42
                )
            elif method == 'neural_network' and NEURAL_NETWORK_AVAILABLE:
                self.mappers[method] = NeuralNetworkMapper(
                    hidden_layers=[64, 32],
                    epochs=50,
                    random_state=42
                )
            else:
                fraud_logger.warning(f"Unknown or unavailable mapping method: {method}")
        
        fraud_logger.info(f"Created {len(self.mappers)} individual mappers")
    
    def fit(self, X_interpretable: np.ndarray, y_pca_components: np.ndarray, **kwargs):
        """
        Train ensemble mapping model
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            y_pca_components: Array of shape (n_samples, 28) for V1-V28
        """
        fraud_logger.info("Training Ensemble feature mapper...")
        start_time = time.time()
        
        # Validate input shapes
        if X_interpretable.shape[0] != y_pca_components.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if y_pca_components.shape[1] != 28:
            raise ValueError("y_pca_components must have 28 features (V1-V28)")
        
        # Create individual mappers
        self._create_mappers()
        
        # Split data for validation (to calculate ensemble weights)
        split_idx = int(0.8 * len(X_interpretable))
        X_train, X_val = X_interpretable[:split_idx], X_interpretable[split_idx:]
        y_train, y_val = y_pca_components[:split_idx], y_pca_components[split_idx:]
        
        # Train individual mappers
        for method, mapper in self.mappers.items():
            fraud_logger.info(f"Training {method} mapper...")
            try:
                mapper.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_predictions = mapper.predict(X_val)
                val_mse = np.mean((y_val - val_predictions) ** 2)
                val_mae = np.mean(np.abs(y_val - val_predictions))
                
                # Calculate correlation preservation
                correlations = []
                for i in range(y_val.shape[1]):
                    corr = np.corrcoef(y_val[:, i], val_predictions[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                avg_correlation = np.mean(correlations) if correlations else 0.0
                
                self.mapper_performances[method] = {
                    'mse': val_mse,
                    'mae': val_mae,
                    'correlation': avg_correlation,
                    'score': avg_correlation - val_mse * 0.1  # Combined score
                }
                
                fraud_logger.info(f"{method} mapper - MSE: {val_mse:.4f}, "
                                f"MAE: {val_mae:.4f}, Correlation: {avg_correlation:.4f}")
                
            except Exception as e:
                fraud_logger.error(f"Failed to train {method} mapper: {e}")
                # Remove failed mapper
                del self.mappers[method]
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights()
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        # Create metadata
        self.metadata = MappingModelMetadata(
            model_name=self.model_name,
            model_type="ensemble",
            version="1.0",
            training_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics={
                'training_time_seconds': training_time,
                'n_samples': X_interpretable.shape[0],
                'n_features': X_interpretable.shape[1],
                'ensemble_methods': list(self.mappers.keys()),
                'ensemble_weights': self.ensemble_weights,
                'individual_performances': self.mapper_performances
            }
        )
        
        fraud_logger.info(f"Ensemble mapper trained in {training_time:.2f}s with {len(self.mappers)} models")
        
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on individual model performance"""
        if not self.mapper_performances:
            # Equal weights if no performance data
            n_models = len(self.mappers)
            self.ensemble_weights = {method: 1.0/n_models for method in self.mappers.keys()}
            return
        
        # Calculate weights based on performance scores
        scores = {method: perf['score'] for method, perf in self.mapper_performances.items()}
        
        # Convert to positive weights (higher score = higher weight)
        min_score = min(scores.values())
        if min_score < 0:
            scores = {method: score - min_score + 0.1 for method, score in scores.items()}
        
        # Normalize to sum to 1
        total_score = sum(scores.values())
        self.ensemble_weights = {method: score/total_score for method, score in scores.items()}
        
        fraud_logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def predict(self, X_interpretable: np.ndarray) -> np.ndarray:
        """
        Predict PCA component values using ensemble approach
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            
        Returns:
            Array of shape (n_samples, 28) with estimated V1-V28 values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.mappers:
            raise ValueError("No trained mappers available")
        
        # Get predictions from all mappers
        predictions = {}
        confidences = {}
        
        for method, mapper in self.mappers.items():
            try:
                pred = mapper.predict(X_interpretable)
                predictions[method] = pred
                
                # Calculate confidence (simplified)
                if hasattr(mapper, 'predict_with_uncertainty'):
                    try:
                        _, uncertainty = mapper.predict_with_uncertainty(X_interpretable)
                        confidence = 1.0 - np.mean(uncertainty)
                    except:
                        confidence = 0.8  # Default confidence
                else:
                    confidence = 0.8  # Default confidence
                
                confidences[method] = confidence
                
            except Exception as e:
                fraud_logger.warning(f"Prediction failed for {method}: {e}")
        
        if not predictions:
            raise ValueError("All mappers failed to make predictions")
        
        # Combine predictions based on method
        if self.combination_method == 'weighted_average':
            ensemble_prediction = self._weighted_average_combination(predictions)
        elif self.combination_method == 'confidence_weighted':
            ensemble_prediction = self._confidence_weighted_combination(predictions, confidences)
        elif self.combination_method == 'best_model':
            ensemble_prediction = self._best_model_combination(predictions, confidences)
        else:
            # Default to weighted average
            ensemble_prediction = self._weighted_average_combination(predictions)
        
        return ensemble_prediction
    
    def _weighted_average_combination(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using weighted average"""
        combined_prediction = None
        total_weight = 0.0
        
        for method, pred in predictions.items():
            weight = self.ensemble_weights.get(method, 1.0/len(predictions))
            
            if combined_prediction is None:
                combined_prediction = weight * pred
            else:
                combined_prediction += weight * pred
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_prediction /= total_weight
        
        return combined_prediction
    
    def _confidence_weighted_combination(self, 
                                       predictions: Dict[str, np.ndarray], 
                                       confidences: Dict[str, float]) -> np.ndarray:
        """Combine predictions using confidence-based weighting"""
        combined_prediction = None
        total_weight = 0.0
        
        for method, pred in predictions.items():
            confidence = confidences.get(method, 0.5)
            base_weight = self.ensemble_weights.get(method, 1.0/len(predictions))
            
            # Combine base weight with confidence
            weight = base_weight * confidence
            
            if combined_prediction is None:
                combined_prediction = weight * pred
            else:
                combined_prediction += weight * pred
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_prediction /= total_weight
        
        return combined_prediction
    
    def _best_model_combination(self, 
                              predictions: Dict[str, np.ndarray], 
                              confidences: Dict[str, float]) -> np.ndarray:
        """Use prediction from the most confident model"""
        # Find method with highest confidence
        best_method = max(confidences.keys(), key=lambda k: confidences[k])
        
        # Check if confidence meets threshold
        if confidences[best_method] >= self.confidence_threshold:
            return predictions[best_method]
        else:
            # Fall back to weighted average if no model is confident enough
            return self._weighted_average_combination(predictions)
    
    def predict_with_uncertainty(self, X_interpretable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation from ensemble variance
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from all mappers
        all_predictions = []
        
        for method, mapper in self.mappers.items():
            try:
                pred = mapper.predict(X_interpretable)
                all_predictions.append(pred)
            except Exception as e:
                fraud_logger.warning(f"Prediction failed for {method}: {e}")
        
        if not all_predictions:
            raise ValueError("All mappers failed to make predictions")
        
        # Convert to array and calculate statistics
        all_predictions = np.array(all_predictions)
        
        # Calculate ensemble mean and uncertainty
        ensemble_mean = np.mean(all_predictions, axis=0)
        ensemble_std = np.std(all_predictions, axis=0)
        
        return ensemble_mean, ensemble_std
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get averaged feature importance across all mappers"""
        if not self.is_trained:
            return None
        
        feature_names = [
            'amount', 'merchant_category', 'location_risk', 'spending_pattern',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]
        
        # Collect importance from all mappers
        all_importances = {}
        
        for method, mapper in self.mappers.items():
            try:
                importance = mapper.get_feature_importance()
                if importance:
                    weight = self.ensemble_weights.get(method, 1.0/len(self.mappers))
                    all_importances[method] = {name: imp * weight for name, imp in importance.items()}
            except Exception as e:
                fraud_logger.warning(f"Could not get feature importance from {method}: {e}")
        
        if not all_importances:
            return None
        
        # Average across all mappers
        averaged_importance = {}
        for feature_name in feature_names:
            total_importance = sum(
                importances.get(feature_name, 0.0) 
                for importances in all_importances.values()
            )
            averaged_importance[feature_name] = total_importance
        
        return averaged_importance
    
    def get_individual_predictions(self, X_interpretable: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual mapper"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        individual_predictions = {}
        
        for method, mapper in self.mappers.items():
            try:
                pred = mapper.predict(X_interpretable)
                individual_predictions[method] = pred
            except Exception as e:
                fraud_logger.warning(f"Prediction failed for {method}: {e}")
        
        return individual_predictions
    
    def get_mapper_performances(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for individual mappers"""
        return self.mapper_performances.copy()
    
    def save_model(self, path: str):
        """Save ensemble model and all individual mappers"""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble metadata
        super().save_model(path)
        
        # Save individual mappers
        mappers_dir = model_path / "individual_mappers"
        mappers_dir.mkdir(exist_ok=True)
        
        for method, mapper in self.mappers.items():
            mapper_path = mappers_dir / method
            mapper.save_model(str(mapper_path))
        
        # Save ensemble-specific data
        ensemble_data = {
            'ensemble_weights': self.ensemble_weights,
            'mapper_performances': self.mapper_performances,
            'combination_method': self.combination_method,
            'confidence_threshold': self.confidence_threshold
        }
        
        joblib.dump(ensemble_data, model_path / "ensemble_data.pkl")
        
        fraud_logger.info(f"Ensemble model saved to {path}")
    
    def load_model(self, path: str):
        """Load ensemble model and all individual mappers"""
        model_path = Path(path)
        
        # Load ensemble metadata
        super().load_model(path)
        
        # Load ensemble-specific data
        ensemble_data_path = model_path / "ensemble_data.pkl"
        if ensemble_data_path.exists():
            ensemble_data = joblib.load(ensemble_data_path)
            self.ensemble_weights = ensemble_data['ensemble_weights']
            self.mapper_performances = ensemble_data['mapper_performances']
            self.combination_method = ensemble_data['combination_method']
            self.confidence_threshold = ensemble_data['confidence_threshold']
        
        # Load individual mappers
        mappers_dir = model_path / "individual_mappers"
        if mappers_dir.exists():
            self.mappers = {}
            
            for method_dir in mappers_dir.iterdir():
                if method_dir.is_dir():
                    method = method_dir.name
                    
                    # Create appropriate mapper
                    if method == 'random_forest':
                        mapper = RandomForestMapper()
                    elif method == 'xgboost':
                        mapper = XGBoostMapper()
                    elif method == 'neural_network' and NEURAL_NETWORK_AVAILABLE:
                        mapper = NeuralNetworkMapper()
                    else:
                        fraud_logger.warning(f"Unknown mapper method: {method}")
                        continue
                    
                    # Load the mapper
                    try:
                        mapper.load_model(str(method_dir))
                        self.mappers[method] = mapper
                    except Exception as e:
                        fraud_logger.error(f"Failed to load {method} mapper: {e}")
        
        fraud_logger.info(f"Ensemble model loaded from {path} with {len(self.mappers)} mappers")