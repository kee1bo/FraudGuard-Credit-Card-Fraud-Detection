"""
Confidence Scoring System
Calculates confidence scores for feature mapping based on input quality,
model uncertainty, and mapping consistency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
from pathlib import Path

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, QualityMetrics
)
from fraudguard.logger import fraud_logger


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to overall confidence"""
    input_completeness: float  # How complete/valid are the user inputs
    input_consistency: float   # How consistent are inputs with typical patterns
    model_uncertainty: float   # Uncertainty from the mapping model
    feature_bounds_compliance: float  # How well features stay within expected bounds
    cross_model_agreement: float  # Agreement between different mapping models
    historical_performance: float  # Historical performance of similar mappings


class ConfidenceScorer:
    """Calculates confidence scores for feature mapping operations"""
    
    def __init__(self, 
                 confidence_thresholds: Optional[Dict[str, float]] = None,
                 historical_data_path: Optional[str] = None):
        
        # Default confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6,
            'unacceptable': 0.5
        }
        
        # Historical performance data for similar mappings
        self.historical_performance_data = {}
        
        # Input validation patterns learned from training data
        self.input_patterns = {}
        
        if historical_data_path:
            self.load_historical_data(historical_data_path)
    
    def calculate_input_completeness(self, user_input: UserTransactionInput) -> float:
        """
        Calculate completeness score based on input quality
        
        Args:
            user_input: UserTransactionInput object
            
        Returns:
            Completeness score (0-1, higher is better)
        """
        completeness_score = 1.0
        
        # Check transaction amount validity
        if user_input.transaction_amount <= 0:
            completeness_score -= 0.3
        elif user_input.transaction_amount > 10000:  # Very high amount
            completeness_score -= 0.1
        
        # Check time context validity
        time_ctx = user_input.time_context
        if not (0 <= time_ctx.hour_of_day <= 23):
            completeness_score -= 0.2
        if not (0 <= time_ctx.day_of_week <= 6):
            completeness_score -= 0.2
        
        # Merchant category should be valid (enum ensures this)
        # Location risk should be valid (enum ensures this)
        # Spending pattern should be valid (enum ensures this)
        
        return max(0.0, completeness_score)
    
    def calculate_input_consistency(self, user_input: UserTransactionInput) -> float:
        """
        Calculate consistency score based on typical transaction patterns
        
        Args:
            user_input: UserTransactionInput object
            
        Returns:
            Consistency score (0-1, higher is better)
        """
        consistency_score = 1.0
        
        # Check amount vs merchant category consistency
        amount = user_input.transaction_amount
        merchant = user_input.merchant_category.value
        
        # Define typical amount ranges for different merchant categories
        typical_ranges = {
            'grocery': (10, 200),
            'gas': (20, 100),
            'online': (15, 500),
            'restaurant': (10, 150),
            'atm': (20, 500),
            'department_store': (25, 300),
            'pharmacy': (5, 100),
            'entertainment': (15, 200),
            'travel': (100, 2000),
            'other': (5, 1000)
        }
        
        if merchant in typical_ranges:
            min_amount, max_amount = typical_ranges[merchant]
            if amount < min_amount * 0.5 or amount > max_amount * 2:
                consistency_score -= 0.2
        
        # Check time vs merchant category consistency
        hour = user_input.time_context.hour_of_day
        
        # Define typical hours for different merchant categories
        if merchant == 'grocery' and (hour < 6 or hour > 23):
            consistency_score -= 0.1
        elif merchant == 'gas' and hour < 5:  # Gas stations usually 24/7
            consistency_score -= 0.05
        elif merchant == 'restaurant' and (hour < 6 or hour > 24):
            consistency_score -= 0.1
        elif merchant == 'atm' and hour < 6:
            consistency_score -= 0.05
        
        # Check spending pattern vs amount consistency
        spending = user_input.spending_pattern.value
        
        if spending == 'typical' and amount > 1000:
            consistency_score -= 0.15
        elif spending == 'suspicious' and amount < 100:
            consistency_score -= 0.1
        elif spending == 'much_higher' and amount < 50:
            consistency_score -= 0.2
        
        # Check location risk vs amount consistency
        location = user_input.location_risk.value
        
        if location == 'foreign' and amount > 2000:
            consistency_score -= 0.1  # High amount in foreign location is risky
        elif location == 'normal' and user_input.spending_pattern.value == 'suspicious':
            consistency_score -= 0.1  # Suspicious spending in normal location is inconsistent
        
        return max(0.0, consistency_score)
    
    def calculate_model_uncertainty_score(self, 
                                        uncertainty_estimates: Optional[np.ndarray] = None,
                                        model_confidence: Optional[float] = None) -> float:
        """
        Calculate uncertainty score from mapping model
        
        Args:
            uncertainty_estimates: Uncertainty estimates for each PCA component
            model_confidence: Overall model confidence if available
            
        Returns:
            Uncertainty score (0-1, higher means lower uncertainty)
        """
        if model_confidence is not None:
            return max(0.0, min(1.0, model_confidence))
        
        if uncertainty_estimates is not None:
            # Convert uncertainty to confidence (inverse relationship)
            avg_uncertainty = np.mean(uncertainty_estimates)
            
            # Normalize uncertainty (assuming reasonable range 0-2)
            normalized_uncertainty = min(1.0, avg_uncertainty / 2.0)
            
            # Convert to confidence score
            confidence = 1.0 - normalized_uncertainty
            
            return max(0.0, confidence)
        
        # Default moderate confidence if no uncertainty information
        return 0.7
    
    def calculate_bounds_compliance_score(self, 
                                        mapped_features: np.ndarray,
                                        feature_bounds: Optional[Dict] = None) -> float:
        """
        Calculate how well mapped features comply with expected bounds
        
        Args:
            mapped_features: Mapped PCA features (V1-V28)
            feature_bounds: Expected bounds for each feature
            
        Returns:
            Bounds compliance score (0-1, higher is better)
        """
        if feature_bounds is None:
            # Default reasonable bounds based on typical PCA component ranges
            feature_bounds = {f'V{i}': {'min': -5, 'max': 5} for i in range(1, 29)}
        
        compliance_scores = []
        
        for i, feature_value in enumerate(mapped_features):
            feature_name = f'V{i+1}'
            
            if feature_name in feature_bounds:
                bounds = feature_bounds[feature_name]
                min_val, max_val = bounds['min'], bounds['max']
                
                if min_val <= feature_value <= max_val:
                    compliance_scores.append(1.0)
                else:
                    # Calculate how far outside bounds
                    if feature_value < min_val:
                        excess = (min_val - feature_value) / abs(min_val)
                    else:
                        excess = (feature_value - max_val) / abs(max_val)
                    
                    # Penalize based on how far outside bounds
                    penalty = min(1.0, excess * 0.5)
                    compliance_scores.append(max(0.0, 1.0 - penalty))
            else:
                compliance_scores.append(0.8)  # Default moderate score
        
        return np.mean(compliance_scores)
    
    def calculate_cross_model_agreement(self, 
                                      predictions: Dict[str, np.ndarray]) -> float:
        """
        Calculate agreement between different mapping models
        
        Args:
            predictions: Dictionary of predictions from different models
            
        Returns:
            Cross-model agreement score (0-1, higher is better)
        """
        if len(predictions) < 2:
            return 0.8  # Default score if only one model
        
        # Calculate pairwise correlations between model predictions
        model_names = list(predictions.keys())
        correlations = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                pred1 = predictions[model_names[i]].flatten()
                pred2 = predictions[model_names[j]].flatten()
                
                correlation = np.corrcoef(pred1, pred2)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        if not correlations:
            return 0.5  # Default moderate score
        
        # Average correlation as agreement measure
        avg_correlation = np.mean(correlations)
        
        # Convert correlation to 0-1 scale (correlation can be negative)
        agreement_score = (avg_correlation + 1) / 2
        
        return max(0.0, min(1.0, agreement_score))
    
    def calculate_historical_performance_score(self, 
                                             user_input: UserTransactionInput) -> float:
        """
        Calculate score based on historical performance of similar mappings
        
        Args:
            user_input: UserTransactionInput object
            
        Returns:
            Historical performance score (0-1, higher is better)
        """
        # Create a key for similar transaction patterns
        pattern_key = (
            user_input.merchant_category.value,
            user_input.location_risk.value,
            user_input.spending_pattern.value
        )
        
        if pattern_key in self.historical_performance_data:
            performance_data = self.historical_performance_data[pattern_key]
            return performance_data.get('avg_confidence', 0.7)
        
        # If no historical data, use category-based estimates
        merchant = user_input.merchant_category.value
        
        # Some merchant categories are easier to map accurately
        category_performance = {
            'grocery': 0.85,
            'gas': 0.80,
            'restaurant': 0.82,
            'atm': 0.75,
            'online': 0.70,  # More variable
            'department_store': 0.78,
            'pharmacy': 0.83,
            'entertainment': 0.72,
            'travel': 0.65,  # Most variable
            'other': 0.60
        }
        
        return category_performance.get(merchant, 0.7)
    
    def calculate_overall_confidence(self, 
                                   user_input: UserTransactionInput,
                                   mapped_features: np.ndarray,
                                   uncertainty_estimates: Optional[np.ndarray] = None,
                                   model_predictions: Optional[Dict[str, np.ndarray]] = None,
                                   feature_bounds: Optional[Dict] = None) -> Tuple[float, ConfidenceFactors]:
        """
        Calculate overall confidence score and individual factors
        
        Args:
            user_input: UserTransactionInput object
            mapped_features: Mapped PCA features
            uncertainty_estimates: Optional uncertainty estimates
            model_predictions: Optional predictions from multiple models
            feature_bounds: Optional feature bounds for validation
            
        Returns:
            Tuple of (overall_confidence, confidence_factors)
        """
        # Calculate individual confidence factors
        input_completeness = self.calculate_input_completeness(user_input)
        input_consistency = self.calculate_input_consistency(user_input)
        model_uncertainty = self.calculate_model_uncertainty_score(uncertainty_estimates)
        bounds_compliance = self.calculate_bounds_compliance_score(mapped_features, feature_bounds)
        
        cross_model_agreement = 0.8  # Default
        if model_predictions:
            cross_model_agreement = self.calculate_cross_model_agreement(model_predictions)
        
        historical_performance = self.calculate_historical_performance_score(user_input)
        
        # Create confidence factors object
        confidence_factors = ConfidenceFactors(
            input_completeness=input_completeness,
            input_consistency=input_consistency,
            model_uncertainty=model_uncertainty,
            feature_bounds_compliance=bounds_compliance,
            cross_model_agreement=cross_model_agreement,
            historical_performance=historical_performance
        )
        
        # Calculate weighted overall confidence
        weights = {
            'input_completeness': 0.20,
            'input_consistency': 0.25,
            'model_uncertainty': 0.20,
            'bounds_compliance': 0.15,
            'cross_model_agreement': 0.10,
            'historical_performance': 0.10
        }
        
        overall_confidence = (
            input_completeness * weights['input_completeness'] +
            input_consistency * weights['input_consistency'] +
            model_uncertainty * weights['model_uncertainty'] +
            bounds_compliance * weights['bounds_compliance'] +
            cross_model_agreement * weights['cross_model_agreement'] +
            historical_performance * weights['historical_performance']
        )
        
        return overall_confidence, confidence_factors
    
    def get_confidence_level(self, confidence_score: float) -> str:
        """Get confidence level description"""
        if confidence_score >= self.confidence_thresholds['excellent']:
            return "Excellent"
        elif confidence_score >= self.confidence_thresholds['good']:
            return "Good"
        elif confidence_score >= self.confidence_thresholds['acceptable']:
            return "Acceptable"
        elif confidence_score >= self.confidence_thresholds['poor']:
            return "Poor"
        else:
            return "Unacceptable"
    
    def should_use_fallback(self, confidence_score: float) -> bool:
        """Determine if fallback mapping should be used"""
        return confidence_score < self.confidence_thresholds['acceptable']
    
    def generate_confidence_explanation(self, 
                                      confidence_score: float,
                                      confidence_factors: ConfidenceFactors) -> Dict[str, Any]:
        """
        Generate detailed explanation of confidence score
        
        Args:
            confidence_score: Overall confidence score
            confidence_factors: Individual confidence factors
            
        Returns:
            Dictionary with detailed confidence explanation
        """
        explanation = {
            'overall_confidence': {
                'score': confidence_score,
                'level': self.get_confidence_level(confidence_score),
                'use_fallback': self.should_use_fallback(confidence_score)
            },
            'contributing_factors': {
                'input_completeness': {
                    'score': confidence_factors.input_completeness,
                    'impact': 'High' if confidence_factors.input_completeness < 0.7 else 'Low',
                    'description': self._describe_input_completeness(confidence_factors.input_completeness)
                },
                'input_consistency': {
                    'score': confidence_factors.input_consistency,
                    'impact': 'High' if confidence_factors.input_consistency < 0.7 else 'Low',
                    'description': self._describe_input_consistency(confidence_factors.input_consistency)
                },
                'model_uncertainty': {
                    'score': confidence_factors.model_uncertainty,
                    'impact': 'Medium' if confidence_factors.model_uncertainty < 0.8 else 'Low',
                    'description': self._describe_model_uncertainty(confidence_factors.model_uncertainty)
                },
                'bounds_compliance': {
                    'score': confidence_factors.feature_bounds_compliance,
                    'impact': 'Medium' if confidence_factors.feature_bounds_compliance < 0.8 else 'Low',
                    'description': self._describe_bounds_compliance(confidence_factors.feature_bounds_compliance)
                }
            },
            'recommendations': self._generate_confidence_recommendations(confidence_score, confidence_factors)
        }
        
        return explanation
    
    def _describe_input_completeness(self, score: float) -> str:
        """Describe input completeness score"""
        if score >= 0.9:
            return "All inputs are complete and valid"
        elif score >= 0.8:
            return "Inputs are mostly complete with minor issues"
        elif score >= 0.7:
            return "Some input validation issues detected"
        else:
            return "Significant input validation problems"
    
    def _describe_input_consistency(self, score: float) -> str:
        """Describe input consistency score"""
        if score >= 0.9:
            return "Inputs are highly consistent with typical patterns"
        elif score >= 0.8:
            return "Inputs are mostly consistent with expected patterns"
        elif score >= 0.7:
            return "Some inconsistencies in input patterns detected"
        else:
            return "Inputs show significant inconsistencies"
    
    def _describe_model_uncertainty(self, score: float) -> str:
        """Describe model uncertainty score"""
        if score >= 0.9:
            return "Very low model uncertainty - high mapping confidence"
        elif score >= 0.8:
            return "Low model uncertainty - good mapping confidence"
        elif score >= 0.7:
            return "Moderate model uncertainty"
        else:
            return "High model uncertainty - low mapping confidence"
    
    def _describe_bounds_compliance(self, score: float) -> str:
        """Describe bounds compliance score"""
        if score >= 0.9:
            return "Mapped features are well within expected bounds"
        elif score >= 0.8:
            return "Most mapped features are within expected bounds"
        elif score >= 0.7:
            return "Some mapped features exceed expected bounds"
        else:
            return "Many mapped features are outside expected bounds"
    
    def _generate_confidence_recommendations(self, 
                                           confidence_score: float,
                                           confidence_factors: ConfidenceFactors) -> List[str]:
        """Generate recommendations based on confidence analysis"""
        recommendations = []
        
        if confidence_factors.input_completeness < 0.7:
            recommendations.append("Review and correct input validation issues")
        
        if confidence_factors.input_consistency < 0.7:
            recommendations.append("Check input consistency - some patterns seem unusual")
        
        if confidence_factors.model_uncertainty > 0.3:
            recommendations.append("Consider using ensemble mapping for better confidence")
        
        if confidence_factors.feature_bounds_compliance < 0.8:
            recommendations.append("Mapped features may need bounds correction")
        
        if confidence_score < 0.6:
            recommendations.append("Consider using fallback mapping approach")
        
        if not recommendations:
            recommendations.append("Confidence is good - mapping can be used reliably")
        
        return recommendations
    
    def update_historical_performance(self, 
                                    user_input: UserTransactionInput,
                                    actual_confidence: float):
        """Update historical performance data with new mapping result"""
        pattern_key = (
            user_input.merchant_category.value,
            user_input.location_risk.value,
            user_input.spending_pattern.value
        )
        
        if pattern_key not in self.historical_performance_data:
            self.historical_performance_data[pattern_key] = {
                'count': 0,
                'total_confidence': 0.0,
                'avg_confidence': 0.0
            }
        
        data = self.historical_performance_data[pattern_key]
        data['count'] += 1
        data['total_confidence'] += actual_confidence
        data['avg_confidence'] = data['total_confidence'] / data['count']
    
    def save_historical_data(self, output_path: str):
        """Save historical performance data"""
        try:
            joblib.dump(self.historical_performance_data, output_path)
            fraud_logger.info(f"Historical performance data saved to {output_path}")
        except Exception as e:
            fraud_logger.error(f"Error saving historical data: {e}")
    
    def load_historical_data(self, input_path: str):
        """Load historical performance data"""
        try:
            if Path(input_path).exists():
                self.historical_performance_data = joblib.load(input_path)
                fraud_logger.info(f"Historical performance data loaded from {input_path}")
            else:
                fraud_logger.warning(f"Historical data file not found: {input_path}")
        except Exception as e:
            fraud_logger.error(f"Error loading historical data: {e}")