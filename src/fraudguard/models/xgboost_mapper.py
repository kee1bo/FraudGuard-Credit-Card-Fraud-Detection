"""
XGBoost Feature Mapper
Implements multi-output XGBoost regression for mapping user-friendly inputs to PCA components.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import time

from fraudguard.models.base_feature_mapper import BaseFeatureMapper
from fraudguard.entity.feature_mapping_entity import MappingModelMetadata
from fraudguard.logger import fraud_logger


class XGBoostMapper(BaseFeatureMapper):
    """XGBoost multi-output regression for feature mapping"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 enable_hyperparameter_tuning: bool = False,
                 **kwargs):
        super().__init__("xgboost_mapper", **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.best_params = None
        
    def _create_model(self):
        """Create XGBoost multi-output regressor"""
        base_xgb = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0  # Reduce XGBoost output
        )
        
        # Use MultiOutputRegressor for handling 28 PCA components
        self.model = MultiOutputRegressor(base_xgb, n_jobs=self.n_jobs)
        
        return self.model
    
    def _perform_hyperparameter_tuning(self, X_scaled: np.ndarray, y_scaled: np.ndarray):
        """Perform hyperparameter tuning using GridSearchCV"""
        fraud_logger.info("Performing hyperparameter tuning for XGBoost mapper...")
        
        # Define parameter grid
        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__max_depth': [3, 6, 9],
            'estimator__learning_rate': [0.05, 0.1, 0.2],
            'estimator__subsample': [0.8, 0.9, 1.0],
            'estimator__colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create base model for tuning
        base_xgb = xgb.XGBRegressor(
            random_state=self.random_state,
            n_jobs=1,  # Use single job for grid search
            verbosity=0
        )
        multi_output_model = MultiOutputRegressor(base_xgb, n_jobs=1)
        
        # Perform grid search (use smaller sample for speed)
        sample_size = min(1000, X_scaled.shape[0])
        indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
        X_sample = X_scaled[indices]
        y_sample = y_scaled[indices]
        
        grid_search = GridSearchCV(
            multi_output_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        grid_search.fit(X_sample, y_sample)
        
        self.best_params = grid_search.best_params_
        fraud_logger.info(f"Best parameters found: {self.best_params}")
        
        # Update model with best parameters
        best_xgb = xgb.XGBRegressor(
            n_estimators=self.best_params['estimator__n_estimators'],
            max_depth=self.best_params['estimator__max_depth'],
            learning_rate=self.best_params['estimator__learning_rate'],
            subsample=self.best_params['estimator__subsample'],
            colsample_bytree=self.best_params['estimator__colsample_bytree'],
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0
        )
        
        self.model = MultiOutputRegressor(best_xgb, n_jobs=self.n_jobs)
    
    def fit(self, X_interpretable: np.ndarray, y_pca_components: np.ndarray, **kwargs):
        """
        Train XGBoost mapping model
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            y_pca_components: Array of shape (n_samples, 28) for V1-V28
        """
        fraud_logger.info("Training XGBoost feature mapper...")
        start_time = time.time()
        
        # Validate input shapes
        if X_interpretable.shape[0] != y_pca_components.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if y_pca_components.shape[1] != 28:
            raise ValueError("y_pca_components must have 28 features (V1-V28)")
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X_interpretable)
        y_scaled = self.target_scaler.fit_transform(y_pca_components)
        
        # Perform hyperparameter tuning if enabled
        if self.enable_hyperparameter_tuning:
            self._perform_hyperparameter_tuning(X_scaled, y_scaled)
        else:
            # Create model with default parameters
            self._create_model()
        
        # Train the model
        self.model.fit(X_scaled, y_scaled)
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Calculate training metrics
        train_score = self.model.score(X_scaled, y_scaled)
        
        # Create metadata
        performance_metrics = {
            'training_score': train_score,
            'training_time_seconds': training_time,
            'n_samples': X_interpretable.shape[0],
            'n_features': X_interpretable.shape[1]
        }
        
        if self.best_params:
            performance_metrics['best_params'] = self.best_params
        
        self.metadata = MappingModelMetadata(
            model_name=self.model_name,
            model_type="xgboost",
            version="1.0",
            training_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics=performance_metrics
        )
        
        fraud_logger.info(f"XGBoost mapper trained in {training_time:.2f}s with score: {train_score:.4f}")
        
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
        """Get feature importance from XGBoost model"""
        if not self.is_trained:
            return None
        
        feature_names = [
            'amount', 'merchant_category', 'location_risk', 'spending_pattern',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]
        
        # Average importance across all output regressors
        importances = []
        for estimator in self.model.estimators_:
            # XGBoost feature importance
            importance = estimator.feature_importances_
            importances.append(importance)
        
        avg_importance = np.mean(importances, axis=0)
        
        return dict(zip(feature_names, avg_importance))
    
    def get_detailed_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get detailed feature importance including gain, weight, and cover"""
        if not self.is_trained:
            return {}
        
        feature_names = [
            'amount', 'merchant_category', 'location_risk', 'spending_pattern',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]
        
        importance_types = ['weight', 'gain', 'cover']
        detailed_importance = {imp_type: {} for imp_type in importance_types}
        
        for imp_type in importance_types:
            importances = []
            for estimator in self.model.estimators_:
                try:
                    # Get importance of specific type
                    booster = estimator.get_booster()
                    importance = booster.get_score(importance_type=imp_type)
                    
                    # Convert to array format
                    importance_array = np.zeros(len(feature_names))
                    for i, feature_name in enumerate([f'f{i}' for i in range(len(feature_names))]):
                        importance_array[i] = importance.get(feature_name, 0.0)
                    
                    importances.append(importance_array)
                except:
                    # Fallback to default feature importance
                    importances.append(estimator.feature_importances_)
            
            avg_importance = np.mean(importances, axis=0)
            detailed_importance[imp_type] = dict(zip(feature_names, avg_importance))
        
        return detailed_importance
    
    def predict_with_leaf_indices(self, X_interpretable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with leaf indices for interpretability
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            
        Returns:
            Tuple of (predictions, leaf_indices)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input features
        X_scaled = self.feature_scaler.transform(X_interpretable)
        
        # Make predictions
        predictions = self.predict(X_interpretable)
        
        # Get leaf indices for each estimator
        leaf_indices = []
        for estimator in self.model.estimators_:
            try:
                leaves = estimator.apply(X_scaled)
                leaf_indices.append(leaves)
            except:
                # If apply method not available, use dummy indices
                leaf_indices.append(np.zeros((X_scaled.shape[0], estimator.n_estimators)))
        
        leaf_indices = np.array(leaf_indices)
        
        return predictions, leaf_indices
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics"""
        if not self.is_trained:
            return {}
        
        complexity_metrics = {}
        
        for i, estimator in enumerate(self.model.estimators_):
            component_name = f'V{i+1}'
            
            try:
                booster = estimator.get_booster()
                
                # Get tree information
                tree_info = booster.get_dump(dump_format='json')
                
                # Calculate complexity metrics
                total_nodes = sum(len(tree.split(',')) for tree in tree_info)
                avg_depth = np.mean([tree.count(':') for tree in tree_info])
                
                complexity_metrics[component_name] = {
                    'n_estimators': estimator.n_estimators,
                    'total_nodes': total_nodes,
                    'avg_depth': avg_depth
                }
            except:
                # Fallback metrics
                complexity_metrics[component_name] = {
                    'n_estimators': getattr(estimator, 'n_estimators', 0),
                    'total_nodes': 0,
                    'avg_depth': 0
                }
        
        return complexity_metrics