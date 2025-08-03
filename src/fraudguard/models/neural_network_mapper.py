"""
Neural Network Feature Mapper
Implements neural network regression for mapping user-friendly inputs to PCA components.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import time
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None

from fraudguard.models.base_feature_mapper import BaseFeatureMapper
from fraudguard.entity.feature_mapping_entity import MappingModelMetadata
from fraudguard.logger import fraud_logger


class NeuralNetworkMapper(BaseFeatureMapper):
    """Neural network multi-output regression for feature mapping"""
    
    def __init__(self, 
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 random_state: int = 42,
                 **kwargs):
        super().__init__("neural_network_mapper", **kwargs)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for NeuralNetworkMapper. Install with: pip install tensorflow")
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.training_history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _create_model(self, input_dim: int, output_dim: int = 28):
        """Create neural network model architecture"""
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='interpretable_features')
        
        # Hidden layers
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units, 
                activation=self.activation,
                name=f'hidden_{i+1}'
            )(x)
            
            # Add dropout for regularization
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer for 28 PCA components
        outputs = layers.Dense(
            output_dim, 
            activation='linear',  # Linear activation for regression
            name='pca_components'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='feature_mapper')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def fit(self, X_interpretable: np.ndarray, y_pca_components: np.ndarray, **kwargs):
        """
        Train neural network mapping model
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            y_pca_components: Array of shape (n_samples, 28) for V1-V28
        """
        fraud_logger.info("Training Neural Network feature mapper...")
        start_time = time.time()
        
        # Validate input shapes
        if X_interpretable.shape[0] != y_pca_components.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if y_pca_components.shape[1] != 28:
            raise ValueError("y_pca_components must have 28 features (V1-V28)")
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X_interpretable)
        y_scaled = self.target_scaler.fit_transform(y_pca_components)
        
        # Create model
        input_dim = X_scaled.shape[1]
        self._create_model(input_dim)
        
        # Setup callbacks
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        callback_list.append(reduce_lr)
        
        # Train the model
        history = self.model.fit(
            X_scaled, y_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callback_list,
            verbose=0  # Reduce output
        )
        
        self.training_history = history.history
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        final_loss = min(history.history['val_loss'])
        final_mae = min(history.history['val_mae'])
        epochs_trained = len(history.history['loss'])
        
        # Create metadata
        self.metadata = MappingModelMetadata(
            model_name=self.model_name,
            model_type="neural_network",
            version="1.0",
            training_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics={
                'final_val_loss': final_loss,
                'final_val_mae': final_mae,
                'epochs_trained': epochs_trained,
                'training_time_seconds': training_time,
                'n_samples': X_interpretable.shape[0],
                'n_features': X_interpretable.shape[1]
            }
        )
        
        fraud_logger.info(f"Neural Network mapper trained in {training_time:.2f}s "
                         f"({epochs_trained} epochs, val_loss: {final_loss:.4f})")
        
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
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform to original scale
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def predict_with_uncertainty(self, X_interpretable: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout
        
        Args:
            X_interpretable: Array of shape (n_samples, n_interpretable_features)
            n_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input features
        X_scaled = self.feature_scaler.transform(X_interpretable)
        
        # Enable dropout during inference for uncertainty estimation
        predictions = []
        
        for _ in range(n_samples):
            # Make prediction with dropout enabled
            y_pred_scaled = self.model(X_scaled, training=True)
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.numpy())
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)
        
        return mean_predictions, std_predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance using gradient-based method
        Note: This is an approximation for neural networks
        """
        if not self.is_trained:
            return None
        
        feature_names = [
            'amount', 'merchant_category', 'location_risk', 'spending_pattern',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]
        
        # Create dummy input for gradient calculation
        dummy_input = np.zeros((1, len(feature_names)))
        dummy_input = self.feature_scaler.transform(dummy_input)
        dummy_input = tf.constant(dummy_input, dtype=tf.float32)
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            tape.watch(dummy_input)
            predictions = self.model(dummy_input)
            # Sum all outputs to get single scalar
            loss = tf.reduce_sum(predictions)
        
        gradients = tape.gradient(loss, dummy_input)
        
        # Use absolute gradients as importance measure
        importance = np.abs(gradients.numpy()[0])
        
        return dict(zip(feature_names, importance))
    
    def get_layer_activations(self, X_interpretable: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Get activations from a specific layer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting activations")
        
        # Scale input features
        X_scaled = self.feature_scaler.transform(X_interpretable)
        
        if layer_name is None:
            # Get activations from the last hidden layer
            layer_name = f'hidden_{len(self.hidden_layers)}'
        
        # Create intermediate model
        intermediate_layer_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Get activations
        activations = intermediate_layer_model.predict(X_scaled, verbose=0)
        
        return activations
    
    def get_training_history(self) -> Optional[Dict]:
        """Get training history"""
        return self.training_history
    
    def plot_training_history(self):
        """Plot training history (requires matplotlib)"""
        if self.training_history is None:
            fraud_logger.warning("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.training_history['loss'], label='Training Loss')
            ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot MAE
            ax2.plot(self.training_history['mae'], label='Training MAE')
            ax2.plot(self.training_history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            fraud_logger.warning("Matplotlib not available for plotting")
    
    def save_model(self, path: str):
        """Save neural network model"""
        super().save_model(path)
        
        # Also save the Keras model
        if self.model is not None:
            model_path = Path(path)
            self.model.save(model_path / "keras_model.h5")
    
    def load_model(self, path: str):
        """Load neural network model"""
        super().load_model(path)
        
        # Also load the Keras model
        model_path = Path(path)
        keras_model_path = model_path / "keras_model.h5"
        
        if keras_model_path.exists():
            self.model = keras.models.load_model(keras_model_path)
        
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not created yet"
        
        # Capture model summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        return '\n'.join(summary_lines)