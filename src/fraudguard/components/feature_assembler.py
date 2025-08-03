"""
Feature Vector Assembler
Combines user inputs with mapped PCA components into complete 30-feature vectors.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, QualityMetrics
)
from fraudguard.logger import fraud_logger


class FeatureVectorAssembler:
    """Assembles complete 30-feature vectors from user inputs and mapped PCA components"""
    
    def __init__(self, bounds_file: Optional[str] = None):
        self.feature_bounds = None
        self.feature_stats = None
        
        if bounds_file:
            self.load_feature_bounds(bounds_file)
    
    def assemble_feature_vector(self, 
                               user_input: UserTransactionInput,
                               mapped_pca: np.ndarray) -> np.ndarray:
        """
        Combine Time, Amount, and mapped V1-V28 into 30-feature vector
        
        Args:
            user_input: UserTransactionInput object
            mapped_pca: Array of shape (28,) with estimated V1-V28 values
            
        Returns:
            Array of shape (30,) with complete feature vector [Time, V1-V28, Amount]
        """
        if mapped_pca.shape[0] != 28:
            raise ValueError("mapped_pca must have exactly 28 features (V1-V28)")
        
        # Extract time from user input (convert to seconds since first transaction)
        # For now, we'll use a placeholder time calculation
        time_value = self._calculate_time_value(user_input.time_context)
        
        # Extract amount
        amount_value = user_input.transaction_amount
        
        # Assemble feature vector in ULB dataset order: [Time, V1-V28, Amount]
        feature_vector = np.zeros(30)
        feature_vector[0] = time_value  # Time
        feature_vector[1:29] = mapped_pca  # V1-V28
        feature_vector[29] = amount_value  # Amount
        
        return feature_vector
    
    def _calculate_time_value(self, time_context) -> float:
        """
        Calculate time value in seconds since first transaction
        This is a simplified calculation for demonstration
        """
        # Convert time context to seconds (simplified approach)
        # In a real implementation, this would be based on actual transaction timing
        base_seconds = time_context.day_of_week * 24 * 3600  # Days to seconds
        base_seconds += time_context.hour_of_day * 3600  # Hours to seconds
        
        # Add some randomness to simulate real transaction timing
        import random
        random.seed(42)  # For reproducibility
        base_seconds += random.randint(0, 3600)  # Random minutes within hour
        
        return float(base_seconds)
    
    def validate_feature_bounds(self, feature_vector: np.ndarray) -> Tuple[bool, Dict[str, str]]:
        """
        Ensure features are within reasonable bounds based on ULB dataset
        
        Args:
            feature_vector: Array of shape (30,) with complete feature vector
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        if self.feature_bounds is None:
            fraud_logger.warning("Feature bounds not loaded, skipping validation")
            return True, {}
        
        validation_messages = {}
        is_valid = True
        
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        for i, (feature_name, value) in enumerate(zip(feature_names, feature_vector)):
            if feature_name in self.feature_bounds:
                bounds = self.feature_bounds[feature_name]
                min_val, max_val = bounds['min'], bounds['max']
                
                if value < min_val:
                    validation_messages[feature_name] = f"Value {value:.4f} below minimum {min_val:.4f}"
                    is_valid = False
                elif value > max_val:
                    validation_messages[feature_name] = f"Value {value:.4f} above maximum {max_val:.4f}"
                    is_valid = False
        
        return is_valid, validation_messages
    
    def apply_statistical_corrections(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Apply corrections to maintain ULB dataset statistical properties
        
        Args:
            feature_vector: Array of shape (30,) with complete feature vector
            
        Returns:
            Corrected feature vector
        """
        if self.feature_bounds is None:
            fraud_logger.warning("Feature bounds not loaded, skipping corrections")
            return feature_vector
        
        corrected_vector = feature_vector.copy()
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        for i, (feature_name, value) in enumerate(zip(feature_names, feature_vector)):
            if feature_name in self.feature_bounds:
                bounds = self.feature_bounds[feature_name]
                min_val, max_val = bounds['min'], bounds['max']
                
                # Clip values to bounds
                corrected_vector[i] = np.clip(value, min_val, max_val)
                
                # Apply additional statistical corrections if needed
                if feature_name.startswith('V'):
                    # For PCA components, ensure they're within reasonable statistical bounds
                    mean_val = bounds.get('mean', 0.0)
                    std_val = bounds.get('std', 1.0)
                    
                    # If value is more than 3 standard deviations away, bring it closer
                    if abs(value - mean_val) > 3 * std_val:
                        if value > mean_val:
                            corrected_vector[i] = mean_val + 3 * std_val
                        else:
                            corrected_vector[i] = mean_val - 3 * std_val
        
        return corrected_vector
    
    def calculate_quality_metrics(self, 
                                 feature_vector: np.ndarray,
                                 mapped_pca: np.ndarray,
                                 mapping_uncertainty: Optional[np.ndarray] = None) -> QualityMetrics:
        """
        Calculate quality metrics for the assembled feature vector
        
        Args:
            feature_vector: Complete 30-feature vector
            mapped_pca: Original mapped PCA components
            mapping_uncertainty: Uncertainty estimates for PCA components
            
        Returns:
            QualityMetrics object
        """
        # Extract PCA components from feature vector
        pca_from_vector = feature_vector[1:29]
        
        # Calculate correlation preservation (simplified)
        correlation_preservation = 1.0  # Placeholder - would need reference correlations
        
        # Calculate distribution similarity (simplified)
        distribution_similarity = 0.8  # Placeholder - would need KL divergence calculation
        
        # Calculate prediction consistency (simplified)
        prediction_consistency = 0.9  # Placeholder - would need model predictions
        
        # Calculate mapping uncertainty
        if mapping_uncertainty is not None:
            avg_uncertainty = np.mean(mapping_uncertainty)
        else:
            avg_uncertainty = 0.1  # Default low uncertainty
        
        # Calculate overall confidence score
        confidence_score = (correlation_preservation + 
                          (1.0 - distribution_similarity) + 
                          prediction_consistency + 
                          (1.0 - avg_uncertainty)) / 4.0
        
        return QualityMetrics(
            correlation_preservation=correlation_preservation,
            distribution_similarity=distribution_similarity,
            prediction_consistency=prediction_consistency,
            mapping_uncertainty=avg_uncertainty,
            confidence_score=confidence_score
        )
    
    def load_feature_bounds(self, bounds_file: str):
        """Load feature bounds from saved analysis results"""
        try:
            bounds_path = Path(bounds_file)
            if bounds_path.exists():
                self.feature_bounds = joblib.load(bounds_path)
                fraud_logger.info(f"Feature bounds loaded from {bounds_file}")
            else:
                fraud_logger.warning(f"Feature bounds file not found: {bounds_file}")
        except Exception as e:
            fraud_logger.error(f"Error loading feature bounds: {e}")
    
    def save_feature_bounds(self, bounds_file: str):
        """Save current feature bounds"""
        if self.feature_bounds is not None:
            joblib.dump(self.feature_bounds, bounds_file)
            fraud_logger.info(f"Feature bounds saved to {bounds_file}")
    
    def set_feature_bounds_from_stats(self, feature_stats: Dict):
        """Set feature bounds from ULB dataset statistics"""
        self.feature_bounds = {}
        
        for feature_name, stats in feature_stats.items():
            self.feature_bounds[feature_name] = {
                'min': stats['min'],
                'max': stats['max'],
                'mean': stats['mean'],
                'std': stats['std'],
                'q25': stats['q25'],
                'q75': stats['q75']
            }
        
        fraud_logger.info("Feature bounds set from statistics")
    
    def validate_and_correct_vector(self, 
                                   user_input: UserTransactionInput,
                                   mapped_pca: np.ndarray,
                                   mapping_uncertainty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, QualityMetrics, Dict[str, str]]:
        """
        Complete validation and correction pipeline
        
        Args:
            user_input: UserTransactionInput object
            mapped_pca: Mapped PCA components
            mapping_uncertainty: Optional uncertainty estimates
            
        Returns:
            Tuple of (corrected_feature_vector, quality_metrics, validation_messages)
        """
        # Assemble initial feature vector
        feature_vector = self.assemble_feature_vector(user_input, mapped_pca)
        
        # Validate bounds
        is_valid, validation_messages = self.validate_feature_bounds(feature_vector)
        
        # Apply corrections if needed
        if not is_valid:
            feature_vector = self.apply_statistical_corrections(feature_vector)
            fraud_logger.info("Applied statistical corrections to feature vector")
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(
            feature_vector, mapped_pca, mapping_uncertainty
        )
        
        return feature_vector, quality_metrics, validation_messages
    
    def get_feature_names(self) -> list:
        """Get ordered list of feature names"""
        return ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    def feature_vector_to_dict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Convert feature vector to named dictionary"""
        feature_names = self.get_feature_names()
        return dict(zip(feature_names, feature_vector))