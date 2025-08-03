"""
Mapping Quality Monitor
Assesses the quality of feature mapping by measuring correlation preservation,
distribution similarity, and prediction consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

from fraudguard.entity.feature_mapping_entity import QualityMetrics, MappingResult
from fraudguard.logger import fraud_logger


class MappingQualityMonitor:
    """Monitors and assesses the quality of feature mapping operations"""
    
    def __init__(self, reference_stats_path: Optional[str] = None):
        self.reference_stats = None
        self.reference_correlations = None
        self.reference_distributions = None
        
        if reference_stats_path:
            self.load_reference_stats(reference_stats_path)
    
    def measure_correlation_preservation(self, 
                                       original_features: np.ndarray,
                                       mapped_features: np.ndarray) -> float:
        """
        Measure how well correlations are preserved between original and mapped features
        
        Args:
            original_features: Original V1-V28 features from ULB dataset
            mapped_features: Mapped V1-V28 features from feature mapping
            
        Returns:
            Correlation preservation score (0-1, higher is better)
        """
        if original_features.shape != mapped_features.shape:
            raise ValueError("Original and mapped features must have the same shape")
        
        if original_features.shape[1] != 28:
            raise ValueError("Features must have 28 components (V1-V28)")
        
        # Calculate correlation matrices
        original_corr = np.corrcoef(original_features.T)
        mapped_corr = np.corrcoef(mapped_features.T)
        
        # Handle NaN values
        original_corr = np.nan_to_num(original_corr, nan=0.0)
        mapped_corr = np.nan_to_num(mapped_corr, nan=0.0)
        
        # Calculate correlation between correlation matrices
        # Flatten upper triangular parts (excluding diagonal)
        mask = np.triu(np.ones_like(original_corr, dtype=bool), k=1)
        original_corr_flat = original_corr[mask]
        mapped_corr_flat = mapped_corr[mask]
        
        # Calculate Pearson correlation between the correlation structures
        correlation_preservation = np.corrcoef(original_corr_flat, mapped_corr_flat)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation_preservation):
            correlation_preservation = 0.0
        
        # Convert to 0-1 scale (correlation can be negative)
        correlation_preservation = (correlation_preservation + 1) / 2
        
        return max(0.0, min(1.0, correlation_preservation))
    
    def measure_distribution_similarity(self, 
                                      original_features: np.ndarray,
                                      mapped_features: np.ndarray) -> float:
        """
        Measure distribution similarity using Jensen-Shannon divergence
        
        Args:
            original_features: Original V1-V28 features from ULB dataset
            mapped_features: Mapped V1-V28 features from feature mapping
            
        Returns:
            Distribution similarity score (0-1, higher is better)
        """
        if original_features.shape[1] != mapped_features.shape[1]:
            raise ValueError("Features must have the same number of components")
        
        similarities = []
        
        for i in range(original_features.shape[1]):
            original_col = original_features[:, i]
            mapped_col = mapped_features[:, i]
            
            # Create histograms for comparison
            # Use the same bins for both distributions
            combined_data = np.concatenate([original_col, mapped_col])
            bins = np.linspace(combined_data.min(), combined_data.max(), 50)
            
            original_hist, _ = np.histogram(original_col, bins=bins, density=True)
            mapped_hist, _ = np.histogram(mapped_col, bins=bins, density=True)
            
            # Normalize to create probability distributions
            original_hist = original_hist / (original_hist.sum() + 1e-10)
            mapped_hist = mapped_hist / (mapped_hist.sum() + 1e-10)
            
            # Add small epsilon to avoid zero probabilities
            epsilon = 1e-10
            original_hist = original_hist + epsilon
            mapped_hist = mapped_hist + epsilon
            
            # Renormalize
            original_hist = original_hist / original_hist.sum()
            mapped_hist = mapped_hist / mapped_hist.sum()
            
            # Calculate Jensen-Shannon divergence
            js_divergence = jensenshannon(original_hist, mapped_hist)
            
            # Convert to similarity (1 - divergence)
            similarity = 1.0 - js_divergence
            similarities.append(similarity)
        
        # Return average similarity across all features
        return np.mean(similarities)
    
    def measure_prediction_consistency(self, 
                                     original_features: np.ndarray,
                                     mapped_features: np.ndarray,
                                     fraud_model,
                                     tolerance: float = 0.1) -> float:
        """
        Measure consistency of fraud predictions between original and mapped features
        
        Args:
            original_features: Complete 30-feature vectors with original V1-V28
            mapped_features: Complete 30-feature vectors with mapped V1-V28
            fraud_model: Trained fraud detection model
            tolerance: Tolerance for prediction differences
            
        Returns:
            Prediction consistency score (0-1, higher is better)
        """
        try:
            # Get predictions for both feature sets
            original_predictions = fraud_model.predict_proba(original_features)[:, 1]
            mapped_predictions = fraud_model.predict_proba(mapped_features)[:, 1]
            
            # Calculate absolute differences
            prediction_diffs = np.abs(original_predictions - mapped_predictions)
            
            # Calculate percentage of predictions within tolerance
            consistent_predictions = np.sum(prediction_diffs <= tolerance)
            consistency_rate = consistent_predictions / len(prediction_diffs)
            
            # Also calculate correlation between predictions
            prediction_correlation = np.corrcoef(original_predictions, mapped_predictions)[0, 1]
            if np.isnan(prediction_correlation):
                prediction_correlation = 0.0
            
            # Combine consistency rate and correlation
            consistency_score = (consistency_rate + max(0, prediction_correlation)) / 2
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            fraud_logger.warning(f"Could not measure prediction consistency: {e}")
            return 0.5  # Default moderate score
    
    def calculate_mapping_uncertainty(self, 
                                    mapped_features: np.ndarray,
                                    uncertainty_estimates: Optional[np.ndarray] = None) -> float:
        """
        Calculate overall uncertainty in the mapping
        
        Args:
            mapped_features: Mapped V1-V28 features
            uncertainty_estimates: Optional uncertainty estimates from mapping model
            
        Returns:
            Mapping uncertainty score (0-1, lower is better)
        """
        if uncertainty_estimates is not None:
            # Use provided uncertainty estimates
            avg_uncertainty = np.mean(uncertainty_estimates)
            
            # Normalize to 0-1 scale (assuming uncertainty is in reasonable range)
            normalized_uncertainty = min(1.0, avg_uncertainty / 2.0)
            
        else:
            # Estimate uncertainty from feature variance
            feature_variances = np.var(mapped_features, axis=0)
            avg_variance = np.mean(feature_variances)
            
            # Normalize variance to uncertainty score
            # Higher variance indicates higher uncertainty
            normalized_uncertainty = min(1.0, avg_variance / 5.0)
        
        return max(0.0, normalized_uncertainty)
    
    def assess_mapping_quality(self, 
                             original_pca_features: np.ndarray,
                             mapped_pca_features: np.ndarray,
                             original_full_features: Optional[np.ndarray] = None,
                             mapped_full_features: Optional[np.ndarray] = None,
                             fraud_model = None,
                             uncertainty_estimates: Optional[np.ndarray] = None) -> QualityMetrics:
        """
        Comprehensive quality assessment of feature mapping
        
        Args:
            original_pca_features: Original V1-V28 features (n_samples, 28)
            mapped_pca_features: Mapped V1-V28 features (n_samples, 28)
            original_full_features: Original complete 30-feature vectors (optional)
            mapped_full_features: Mapped complete 30-feature vectors (optional)
            fraud_model: Trained fraud detection model (optional)
            uncertainty_estimates: Uncertainty estimates from mapping (optional)
            
        Returns:
            QualityMetrics object with comprehensive assessment
        """
        fraud_logger.info("Assessing mapping quality...")
        
        # Measure correlation preservation
        correlation_preservation = self.measure_correlation_preservation(
            original_pca_features, mapped_pca_features
        )
        
        # Measure distribution similarity
        distribution_similarity = self.measure_distribution_similarity(
            original_pca_features, mapped_pca_features
        )
        
        # Measure prediction consistency (if fraud model provided)
        if fraud_model is not None and original_full_features is not None and mapped_full_features is not None:
            prediction_consistency = self.measure_prediction_consistency(
                original_full_features, mapped_full_features, fraud_model
            )
        else:
            prediction_consistency = 0.8  # Default reasonable score
        
        # Calculate mapping uncertainty
        mapping_uncertainty = self.calculate_mapping_uncertainty(
            mapped_pca_features, uncertainty_estimates
        )
        
        # Calculate overall confidence score
        confidence_score = (
            correlation_preservation * 0.3 +
            distribution_similarity * 0.3 +
            prediction_consistency * 0.3 +
            (1.0 - mapping_uncertainty) * 0.1
        )
        
        quality_metrics = QualityMetrics(
            correlation_preservation=correlation_preservation,
            distribution_similarity=distribution_similarity,
            prediction_consistency=prediction_consistency,
            mapping_uncertainty=mapping_uncertainty,
            confidence_score=confidence_score
        )
        
        fraud_logger.info(f"Quality assessment completed - Confidence: {confidence_score:.3f}")
        
        return quality_metrics
    
    def validate_against_reference(self, mapped_features: np.ndarray) -> Dict[str, float]:
        """
        Validate mapped features against reference statistics from ULB dataset
        
        Args:
            mapped_features: Mapped V1-V28 features to validate
            
        Returns:
            Dictionary of validation metrics
        """
        if self.reference_stats is None:
            fraud_logger.warning("No reference statistics loaded")
            return {}
        
        validation_metrics = {}
        
        for i in range(min(mapped_features.shape[1], 28)):
            feature_name = f'V{i+1}'
            
            if feature_name in self.reference_stats:
                ref_stats = self.reference_stats[feature_name]
                mapped_col = mapped_features[:, i]
                
                # Calculate statistics for mapped features
                mapped_mean = np.mean(mapped_col)
                mapped_std = np.std(mapped_col)
                mapped_min = np.min(mapped_col)
                mapped_max = np.max(mapped_col)
                
                # Compare with reference
                mean_diff = abs(mapped_mean - ref_stats['mean']) / (abs(ref_stats['mean']) + 1e-10)
                std_diff = abs(mapped_std - ref_stats['std']) / (ref_stats['std'] + 1e-10)
                
                # Check if values are within reasonable bounds
                within_bounds = (mapped_min >= ref_stats['min'] * 0.8 and 
                               mapped_max <= ref_stats['max'] * 1.2)
                
                validation_metrics[feature_name] = {
                    'mean_relative_error': mean_diff,
                    'std_relative_error': std_diff,
                    'within_bounds': within_bounds,
                    'validation_score': max(0.0, 1.0 - (mean_diff + std_diff) / 2)
                }
        
        # Calculate overall validation score
        if validation_metrics:
            overall_score = np.mean([metrics['validation_score'] 
                                   for metrics in validation_metrics.values()])
            validation_metrics['overall_validation_score'] = overall_score
        
        return validation_metrics
    
    def create_quality_report(self, 
                            quality_metrics: QualityMetrics,
                            validation_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create comprehensive quality report
        
        Args:
            quality_metrics: QualityMetrics object
            validation_metrics: Optional validation metrics
            
        Returns:
            Dictionary containing detailed quality report
        """
        report = {
            'overall_assessment': {
                'confidence_score': quality_metrics.confidence_score,
                'quality_level': self._get_quality_level(quality_metrics.confidence_score)
            },
            'detailed_metrics': {
                'correlation_preservation': {
                    'score': quality_metrics.correlation_preservation,
                    'interpretation': self._interpret_correlation_preservation(
                        quality_metrics.correlation_preservation
                    )
                },
                'distribution_similarity': {
                    'score': quality_metrics.distribution_similarity,
                    'interpretation': self._interpret_distribution_similarity(
                        quality_metrics.distribution_similarity
                    )
                },
                'prediction_consistency': {
                    'score': quality_metrics.prediction_consistency,
                    'interpretation': self._interpret_prediction_consistency(
                        quality_metrics.prediction_consistency
                    )
                },
                'mapping_uncertainty': {
                    'score': quality_metrics.mapping_uncertainty,
                    'interpretation': self._interpret_mapping_uncertainty(
                        quality_metrics.mapping_uncertainty
                    )
                }
            },
            'recommendations': self._generate_recommendations(quality_metrics)
        }
        
        if validation_metrics:
            report['validation_results'] = validation_metrics
        
        return report
    
    def _get_quality_level(self, confidence_score: float) -> str:
        """Get quality level description"""
        if confidence_score >= 0.9:
            return "Excellent"
        elif confidence_score >= 0.8:
            return "Good"
        elif confidence_score >= 0.7:
            return "Acceptable"
        elif confidence_score >= 0.6:
            return "Poor"
        else:
            return "Unacceptable"
    
    def _interpret_correlation_preservation(self, score: float) -> str:
        """Interpret correlation preservation score"""
        if score >= 0.9:
            return "Excellent preservation of feature relationships"
        elif score >= 0.8:
            return "Good preservation of most feature relationships"
        elif score >= 0.7:
            return "Acceptable preservation with some relationship loss"
        else:
            return "Poor preservation - significant relationship distortion"
    
    def _interpret_distribution_similarity(self, score: float) -> str:
        """Interpret distribution similarity score"""
        if score >= 0.9:
            return "Mapped features closely match original distributions"
        elif score >= 0.8:
            return "Good distribution matching with minor deviations"
        elif score >= 0.7:
            return "Acceptable distribution matching with some differences"
        else:
            return "Poor distribution matching - significant deviations detected"
    
    def _interpret_prediction_consistency(self, score: float) -> str:
        """Interpret prediction consistency score"""
        if score >= 0.9:
            return "Fraud predictions highly consistent between original and mapped features"
        elif score >= 0.8:
            return "Good prediction consistency with minor variations"
        elif score >= 0.7:
            return "Acceptable prediction consistency with some differences"
        else:
            return "Poor prediction consistency - significant prediction variations"
    
    def _interpret_mapping_uncertainty(self, score: float) -> str:
        """Interpret mapping uncertainty score"""
        if score <= 0.1:
            return "Very low uncertainty - high confidence in mapping"
        elif score <= 0.2:
            return "Low uncertainty - good confidence in mapping"
        elif score <= 0.3:
            return "Moderate uncertainty - reasonable confidence in mapping"
        else:
            return "High uncertainty - low confidence in mapping"
    
    def _generate_recommendations(self, quality_metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        if quality_metrics.correlation_preservation < 0.7:
            recommendations.append(
                "Consider using ensemble mapping or neural network approach to better preserve feature correlations"
            )
        
        if quality_metrics.distribution_similarity < 0.7:
            recommendations.append(
                "Review training data preparation - mapped features may need better statistical alignment"
            )
        
        if quality_metrics.prediction_consistency < 0.7:
            recommendations.append(
                "Validate mapping model training - predictions show significant inconsistency"
            )
        
        if quality_metrics.mapping_uncertainty > 0.3:
            recommendations.append(
                "Consider collecting more training data or using more sophisticated mapping models"
            )
        
        if quality_metrics.confidence_score < 0.6:
            recommendations.append(
                "Overall mapping quality is poor - consider retraining with different approach"
            )
        
        if not recommendations:
            recommendations.append("Mapping quality is good - no immediate improvements needed")
        
        return recommendations
    
    def load_reference_stats(self, stats_path: str):
        """Load reference statistics from ULB dataset analysis"""
        try:
            stats_file = Path(stats_path)
            if stats_file.exists():
                self.reference_stats = joblib.load(stats_path)
                fraud_logger.info(f"Reference statistics loaded from {stats_path}")
            else:
                fraud_logger.warning(f"Reference statistics file not found: {stats_path}")
        except Exception as e:
            fraud_logger.error(f"Error loading reference statistics: {e}")
    
    def save_quality_assessment(self, 
                              quality_metrics: QualityMetrics,
                              report: Dict[str, Any],
                              output_path: str):
        """Save quality assessment results"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            assessment_data = {
                'quality_metrics': quality_metrics.__dict__,
                'quality_report': report,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            joblib.dump(assessment_data, output_path)
            fraud_logger.info(f"Quality assessment saved to {output_path}")
            
        except Exception as e:
            fraud_logger.error(f"Error saving quality assessment: {e}")