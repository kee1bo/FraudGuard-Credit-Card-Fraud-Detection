"""
Mapping Model Training Pipeline
Trains feature mapping models using the ULB dataset to create the statistical
mapping from user-friendly inputs to PCA components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fraudguard.components.ulb_dataset_analyzer import ULBDatasetAnalyzer, analyze_ulb_dataset
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper
from fraudguard.models.ensemble_mapper import EnsembleMapper
from fraudguard.entity.feature_mapping_entity import TrainingDataPoint, MappingModelMetadata
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException

# Try to import neural network mapper
try:
    from fraudguard.models.neural_network_mapper import NeuralNetworkMapper
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NeuralNetworkMapper = None
    NEURAL_NETWORK_AVAILABLE = False


class MappingTrainingPipeline:
    """Pipeline for training feature mapping models"""
    
    def __init__(self, 
                 dataset_path: str = "data/creditcard.csv",
                 output_dir: str = "artifacts/feature_mapping",
                 test_size: float = 0.2,
                 validation_size: float = 0.1,
                 random_state: int = 42):
        
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "mappers").mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset_analyzer = None
        self.training_data = None
        self.trained_mappers = {}
        
        # Training statistics
        self.training_stats = {
            'dataset_size': 0,
            'training_size': 0,
            'validation_size': 0,
            'test_size': 0,
            'training_time': 0,
            'models_trained': []
        }
    
    def prepare_training_data(self, sample_size: Optional[int] = 10000) -> List[TrainingDataPoint]:
        """
        Prepare training data from ULB dataset
        
        Args:
            sample_size: Number of samples to use (None for full dataset)
            
        Returns:
            List of TrainingDataPoint objects
        """
        fraud_logger.info("Preparing training data from ULB dataset...")
        
        try:
            # Initialize dataset analyzer
            self.dataset_analyzer = ULBDatasetAnalyzer(self.dataset_path)
            
            # Perform complete analysis
            self.dataset_analyzer.load_dataset()
            self.dataset_analyzer.analyze_feature_distributions()
            self.dataset_analyzer.compute_correlation_matrix()
            self.dataset_analyzer.create_merchant_categories()
            
            # Create training dataset
            self.training_data = self.dataset_analyzer.create_training_dataset(sample_size)
            
            # Save analysis results
            self.dataset_analyzer.save_analysis_results(str(self.output_dir))
            
            # Update statistics
            self.training_stats['dataset_size'] = len(self.training_data)
            
            fraud_logger.info(f"Training data prepared: {len(self.training_data)} samples")
            return self.training_data
            
        except Exception as e:
            raise FraudGuardException(f"Failed to prepare training data: {str(e)}")
    
    def split_training_data(self, training_data: List[TrainingDataPoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split training data into train/validation/test sets
        
        Args:
            training_data: List of TrainingDataPoint objects
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        fraud_logger.info("Splitting training data...")
        
        # Convert training data to arrays
        X = np.array([point.to_interpretable_features() for point in training_data])
        y = np.array([point.v1_to_v28 for point in training_data])
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        # Update statistics
        self.training_stats.update({
            'training_size': len(X_train),
            'validation_size': len(X_val),
            'test_size': len(X_test)
        })
        
        fraud_logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest_mapper(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_val: np.ndarray, y_val: np.ndarray) -> RandomForestMapper:
        """Train Random Forest mapping model"""
        fraud_logger.info("Training Random Forest mapper...")
        
        try:
            mapper = RandomForestMapper(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
            
            # Train the mapper
            mapper.fit(X_train, y_train)
            
            # Validate
            validation_metrics = mapper.validate_predictions(X_val, y_val)
            
            # Save the mapper
            mapper_path = self.output_dir / "mappers" / "random_forest"
            mapper.save_model(str(mapper_path))
            
            self.trained_mappers['random_forest'] = {
                'mapper': mapper,
                'validation_metrics': validation_metrics
            }
            
            fraud_logger.info(f"Random Forest mapper trained - Validation MSE: {validation_metrics['overall_mse']:.4f}")
            return mapper
            
        except Exception as e:
            fraud_logger.error(f"Failed to train Random Forest mapper: {e}")
            raise
    
    def train_xgboost_mapper(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> XGBoostMapper:
        """Train XGBoost mapping model"""
        fraud_logger.info("Training XGBoost mapper...")
        
        try:
            mapper = XGBoostMapper(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                enable_hyperparameter_tuning=False  # Disable for faster training
            )
            
            # Train the mapper
            mapper.fit(X_train, y_train)
            
            # Validate
            validation_metrics = mapper.evaluate_mapping_quality(X_val, y_val)
            
            # Save the mapper
            mapper_path = self.output_dir / "mappers" / "xgboost"
            mapper.save_model(str(mapper_path))
            
            self.trained_mappers['xgboost'] = {
                'mapper': mapper,
                'validation_metrics': validation_metrics
            }
            
            fraud_logger.info(f"XGBoost mapper trained - Validation MSE: {validation_metrics['mse']:.4f}")
            return mapper
            
        except Exception as e:
            fraud_logger.error(f"Failed to train XGBoost mapper: {e}")
            raise
    
    def train_neural_network_mapper(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Optional[Any]:
        """Train Neural Network mapping model"""
        if not NEURAL_NETWORK_AVAILABLE:
            fraud_logger.warning("Neural Network mapper not available - skipping")
            return None
        
        fraud_logger.info("Training Neural Network mapper...")
        
        try:
            mapper = NeuralNetworkMapper(
                hidden_layers=[64, 32],
                epochs=50,
                batch_size=32,
                learning_rate=0.001,
                early_stopping_patience=10,
                random_state=self.random_state
            )
            
            # Train the mapper
            mapper.fit(X_train, y_train)
            
            # Validate
            validation_metrics = mapper.evaluate_mapping_quality(X_val, y_val)
            
            # Save the mapper
            mapper_path = self.output_dir / "mappers" / "neural_network"
            mapper.save_model(str(mapper_path))
            
            self.trained_mappers['neural_network'] = {
                'mapper': mapper,
                'validation_metrics': validation_metrics
            }
            
            fraud_logger.info(f"Neural Network mapper trained - Validation MSE: {validation_metrics['mse']:.4f}")
            return mapper
            
        except Exception as e:
            fraud_logger.error(f"Failed to train Neural Network mapper: {e}")
            return None
    
    def train_ensemble_mapper(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> EnsembleMapper:
        """Train Ensemble mapping model"""
        fraud_logger.info("Training Ensemble mapper...")
        
        try:
            # Determine available methods
            ensemble_methods = ['random_forest', 'xgboost']
            if NEURAL_NETWORK_AVAILABLE:
                ensemble_methods.append('neural_network')
            
            mapper = EnsembleMapper(
                ensemble_methods=ensemble_methods,
                combination_method='weighted_average',
                confidence_threshold=0.7
            )
            
            # Train the mapper
            mapper.fit(X_train, y_train)
            
            # Validate
            validation_metrics = mapper.evaluate_mapping_quality(X_val, y_val)
            
            # Save the mapper
            mapper_path = self.output_dir / "mappers" / "ensemble"
            mapper.save_model(str(mapper_path))
            
            self.trained_mappers['ensemble'] = {
                'mapper': mapper,
                'validation_metrics': validation_metrics
            }
            
            fraud_logger.info(f"Ensemble mapper trained - Validation MSE: {validation_metrics['mse']:.4f}")
            return mapper
            
        except Exception as e:
            fraud_logger.error(f"Failed to train Ensemble mapper: {e}")
            raise
    
    def evaluate_mappers(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained mappers on test set"""
        fraud_logger.info("Evaluating trained mappers on test set...")
        
        evaluation_results = {}
        
        for mapper_name, mapper_info in self.trained_mappers.items():
            try:
                mapper = mapper_info['mapper']
                
                # Make predictions
                y_pred = mapper.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Calculate correlation preservation
                correlations = []
                for i in range(y_test.shape[1]):
                    corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                avg_correlation = np.mean(correlations) if correlations else 0.0
                
                evaluation_results[mapper_name] = {
                    'test_mse': mse,
                    'test_mae': mae,
                    'avg_correlation': avg_correlation,
                    'correlation_preservation': max(0.0, avg_correlation)
                }
                
                fraud_logger.info(f"{mapper_name} - Test MSE: {mse:.4f}, MAE: {mae:.4f}, Correlation: {avg_correlation:.4f}")
                
            except Exception as e:
                fraud_logger.error(f"Failed to evaluate {mapper_name}: {e}")
                evaluation_results[mapper_name] = {'error': str(e)}
        
        return evaluation_results
    
    def run_complete_training(self, sample_size: Optional[int] = 10000) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            sample_size: Number of samples to use for training
            
        Returns:
            Dictionary with training results and statistics
        """
        start_time = time.time()
        fraud_logger.info("Starting complete mapping model training pipeline...")
        
        try:
            # Step 1: Prepare training data
            training_data = self.prepare_training_data(sample_size)
            
            # Step 2: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_training_data(training_data)
            
            # Step 3: Train individual mappers
            trained_models = []
            
            # Train Random Forest
            try:
                self.train_random_forest_mapper(X_train, y_train, X_val, y_val)
                trained_models.append('random_forest')
            except Exception as e:
                fraud_logger.error(f"Random Forest training failed: {e}")
            
            # Train XGBoost
            try:
                self.train_xgboost_mapper(X_train, y_train, X_val, y_val)
                trained_models.append('xgboost')
            except Exception as e:
                fraud_logger.error(f"XGBoost training failed: {e}")
            
            # Train Neural Network (if available)
            if NEURAL_NETWORK_AVAILABLE:
                try:
                    nn_mapper = self.train_neural_network_mapper(X_train, y_train, X_val, y_val)
                    if nn_mapper:
                        trained_models.append('neural_network')
                except Exception as e:
                    fraud_logger.error(f"Neural Network training failed: {e}")
            
            # Train Ensemble (if we have at least 2 models)
            if len(trained_models) >= 2:
                try:
                    self.train_ensemble_mapper(X_train, y_train, X_val, y_val)
                    trained_models.append('ensemble')
                except Exception as e:
                    fraud_logger.error(f"Ensemble training failed: {e}")
            
            # Step 4: Evaluate all mappers
            evaluation_results = self.evaluate_mappers(X_test, y_test)
            
            # Step 5: Save training metadata
            training_time = time.time() - start_time
            self.training_stats.update({
                'training_time': training_time,
                'models_trained': trained_models
            })
            
            # Save training statistics
            training_metadata = {
                'training_stats': self.training_stats,
                'evaluation_results': evaluation_results,
                'training_config': {
                    'dataset_path': self.dataset_path,
                    'sample_size': sample_size,
                    'test_size': self.test_size,
                    'validation_size': self.validation_size,
                    'random_state': self.random_state
                },
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = self.output_dir / "training_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(training_metadata, f, indent=2, default=str)
            
            fraud_logger.info(f"Training pipeline completed in {training_time:.2f}s")
            fraud_logger.info(f"Successfully trained {len(trained_models)} mappers: {trained_models}")
            
            return {
                'success': True,
                'trained_models': trained_models,
                'training_stats': self.training_stats,
                'evaluation_results': evaluation_results,
                'training_time': training_time
            }
            
        except Exception as e:
            training_time = time.time() - start_time
            fraud_logger.error(f"Training pipeline failed after {training_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'training_time': training_time,
                'trained_models': list(self.trained_mappers.keys())
            }
    
    def create_demo_mappers(self) -> Dict[str, Any]:
        """
        Create demo mappers with synthetic data for testing when ULB dataset is not available
        """
        fraud_logger.info("Creating demo mappers with synthetic data...")
        
        try:
            # Create synthetic training data
            n_samples = 1000
            n_features = 8  # Number of interpretable features
            n_pca_components = 28
            
            # Generate synthetic interpretable features
            X_synthetic = np.random.randn(n_samples, n_features)
            
            # Generate synthetic PCA components with some correlation to interpretable features
            # This simulates the relationship we would learn from real data
            W = np.random.randn(n_pca_components, n_features) * 0.5  # Weight matrix
            y_synthetic = np.dot(X_synthetic, W.T) + np.random.randn(n_samples, n_pca_components) * 0.1
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_synthetic, y_synthetic, test_size=0.2, random_state=self.random_state
            )
            
            trained_models = []
            
            # Train Random Forest with synthetic data
            try:
                mapper = RandomForestMapper(n_estimators=50, random_state=self.random_state)
                mapper.fit(X_train, y_train)
                
                mapper_path = self.output_dir / "mappers" / "random_forest"
                mapper.save_model(str(mapper_path))
                
                self.trained_mappers['random_forest'] = {'mapper': mapper}
                trained_models.append('random_forest')
                
                fraud_logger.info("Demo Random Forest mapper created")
                
            except Exception as e:
                fraud_logger.error(f"Failed to create demo Random Forest mapper: {e}")
            
            # Train XGBoost with synthetic data
            try:
                mapper = XGBoostMapper(n_estimators=50, random_state=self.random_state)
                mapper.fit(X_train, y_train)
                
                mapper_path = self.output_dir / "mappers" / "xgboost"
                mapper.save_model(str(mapper_path))
                
                self.trained_mappers['xgboost'] = {'mapper': mapper}
                trained_models.append('xgboost')
                
                fraud_logger.info("Demo XGBoost mapper created")
                
            except Exception as e:
                fraud_logger.error(f"Failed to create demo XGBoost mapper: {e}")
            
            # Create ensemble if we have multiple models
            if len(trained_models) >= 2:
                try:
                    mapper = EnsembleMapper(ensemble_methods=trained_models)
                    mapper.fit(X_train, y_train)
                    
                    mapper_path = self.output_dir / "mappers" / "ensemble"
                    mapper.save_model(str(mapper_path))
                    
                    self.trained_mappers['ensemble'] = {'mapper': mapper}
                    trained_models.append('ensemble')
                    
                    fraud_logger.info("Demo Ensemble mapper created")
                    
                except Exception as e:
                    fraud_logger.error(f"Failed to create demo Ensemble mapper: {e}")
            
            # Save demo metadata
            demo_metadata = {
                'type': 'demo_mappers',
                'trained_models': trained_models,
                'synthetic_data_size': n_samples,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = self.output_dir / "demo_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(demo_metadata, f, indent=2)
            
            fraud_logger.info(f"Demo mappers created successfully: {trained_models}")
            
            return {
                'success': True,
                'trained_models': trained_models,
                'type': 'demo',
                'message': 'Demo mappers created with synthetic data'
            }
            
        except Exception as e:
            fraud_logger.error(f"Failed to create demo mappers: {e}")
            return {
                'success': False,
                'error': str(e),
                'type': 'demo'
            }


def train_mapping_models(dataset_path: str = "data/creditcard.csv",
                        output_dir: str = "artifacts/feature_mapping",
                        sample_size: Optional[int] = 10000,
                        create_demo: bool = False) -> Dict[str, Any]:
    """
    Convenience function to train mapping models
    
    Args:
        dataset_path: Path to ULB dataset
        output_dir: Output directory for trained models
        sample_size: Number of samples to use (None for full dataset)
        create_demo: Whether to create demo mappers if dataset is not available
        
    Returns:
        Training results dictionary
    """
    pipeline = MappingTrainingPipeline(
        dataset_path=dataset_path,
        output_dir=output_dir
    )
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        if create_demo:
            fraud_logger.warning(f"Dataset not found at {dataset_path}, creating demo mappers")
            return pipeline.create_demo_mappers()
        else:
            return {
                'success': False,
                'error': f'Dataset not found at {dataset_path}',
                'suggestion': 'Set create_demo=True to create demo mappers for testing'
            }
    
    # Run complete training
    return pipeline.run_complete_training(sample_size)