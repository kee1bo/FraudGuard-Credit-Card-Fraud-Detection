"""
Random Forest Feature Mapper
Implements multi-output Random Forest regression for mapping user-friendly inputs to PCA components.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import time

from fraudguard.models.base_feature_mapper import BaseFeatureMapper
from fraudguard.entity.feature_mapping_entity import MappingModelMetadata
from fraudguard.logger import fraud_logger


class RandomForestMapper(BaseFeatureMapper):
    """Random Forest multi-output regression for feature mapping"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        super().__init__("random_forest_mapper", **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def _create_model(self):
        """Create Random Forest multi-output regressor"""
        base_rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Use MultiOutputRegressor for handling 28 PCA components
        self.model = MultiOutputRegressor(base_rf, n_jobs=self.n_jobs)
        
        return self.model
    
    def fit(self, X_interpretable: np.ndarray, y_pca_components: np.ndarray, **kwargs):
        """
        Train Random Forest mapping model
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            y_pca_components: Array of shape (n_samples, 28) for V1-V28
        """
        fraud_logger.info("Training Random Forest feature mapper...")
        start_time = time.time()
        
        # Validate input shapes
        if X_interpretable.shape[0] != y_pca_components.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if y_pca_components.shape[1] != 28:
            raise ValueError("y_pca_components must have 28 features (V1-V28)")
        
        # Create model if not exists
        if self.model is None:
            self._create_model()
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X_interpretable)
        y_scaled = self.target_scaler.fit_transform(y_pca_components)
        
        # Train the model
        self.model.fit(X_scaled, y_scaled)
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Calculate training metrics
        train_score = self.model.score(X_scaled, y_scaled)
        
        # Create metadata
        self.metadata = MappingModelMetadata(
            model_name=self.model_name,
            model_type="random_forest",
            version="1.0",
            training_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics={
                'training_score': train_score,
                'training_time_seconds': training_time,
                'n_samples': X_interpretable.shape[0],
                'n_features': X_interpretable.shape[1]
            }
        )
        
        fraud_logger.info(f"Random Forest mapper trained in {training_time:.2f}s with score: {train_score:.4f}")
        
    def predict(self, X_interpretable: np.ndarray) -> np.ndarray:
        """
        Predict PCA component values from interpretable features
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            
        Returns:
            Array of shape (n_samples, 28) with estimated V1-V28 values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input features
        X_scaled = self.feature_scaler.transform(X_interpretable)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform to original scale
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from Random Forest model"""
        if not self.is_trained:
            return None
        
        # Get feature importance from each output regressor
        feature_names = [
            'amount', 'merchant_category', 'location_risk', 'spending_pattern',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]
        
        # Average importance across all output regressors
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        
        return dict(zip(feature_names, avg_importance))
    
    def get_pca_component_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each PCA component separately"""
        if not self.is_trained:
            return {}
        
        feature_names = [
            'amount', 'merchant_category', 'location_risk', 'spending_pattern',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]
        
        component_importance = {}
        
        for i, estimator in enumerate(self.model.estimators_):
            component_name = f'V{i+1}'
            component_importance[component_name] = dict(
                zip(feature_names, estimator.feature_importances_)
            )
        
        return component_importance
    
    def predict_with_uncertainty(self, X_interpretable: np.ndarray) -> tuple:
        """
        Predict with uncertainty estimation using Random Forest variance
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            
        Returns:
            Tuple of (predictions, uncertainties) where uncertainties are standard deviations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input features
        X_scaled = self.feature_scaler.transform(X_interpretable)
        
        # Get predictions from all trees for uncertainty estimation
        all_predictions = []
        
        for estimator in self.model.estimators_:
            # Get predictions from all trees in this estimator
            tree_predictions = []
            for tree in estimator.estimators_:
                pred = tree.predict(X_scaled)
                tree_predictions.append(pred)
            
            # Average predictions from trees and add to list
            avg_pred = np.mean(tree_predictions, axis=0)
            all_predictions.append(avg_pred)
        
        # Convert to array and calculate statistics
        all_predictions = np.array(all_predictions)  # Shape: (n_components, n_samples)
        
        # Calculate mean and std across components
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        # Inverse transform to original scale
        mean_predictions = self.target_scaler.inverse_transform(mean_predictions)
        
        # Scale uncertainties (approximate)
        std_predictions = std_predictions * self.target_scaler.scale_
        
        return mean_predictions, std_predictions
    
    def validate_predictions(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Validate predictions and return detailed metrics"""
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate component-wise metrics
        component_metrics = {}
        
        for i in range(28):
            component_name = f'V{i+1}'
            
            # Calculate metrics for this component
            mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
            mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
            correlation = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
            
            component_metrics[component_name] = {
                'mse': mse,
                'mae': mae,
                'correlation': correlation if not np.isnan(correlation) else 0.0
            }
        
        # Overall metrics
        overall_mse = np.mean((y_test - y_pred) ** 2)
        overall_mae = np.mean(np.abs(y_test - y_pred))
        
        # Average correlation across components
        correlations = [metrics['correlation'] for metrics in component_metrics.values()]
        avg_correlation = np.mean([c for c in correlations if not np.isnan(c)])
        
        return {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'avg_correlation': avg_correlation,
            'component_metrics': component_metrics
        }