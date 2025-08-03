"""
Comprehensive Error Handler for Feature Mapping Pipeline
Provides robust error handling, fallback mechanisms, and recovery strategies
for the intelligent feature mapping system.
"""

import traceback
import time
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, QualityMetrics, MappingExplanation
)
from fraudguard.logger import fraud_logger


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    INPUT_VALIDATION = "input_validation"
    MODEL_LOADING = "model_loading"
    FEATURE_MAPPING = "feature_mapping"
    PREDICTION = "prediction"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    NETWORK = "network"
    DATA_CORRUPTION = "data_corruption"


@dataclass
class ErrorInfo:
    """Detailed error information"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: str
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class FallbackStrategy:
    """Base class for fallback strategies"""
    
    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.success_count = 0
        self.failure_count = 0
        self.last_used = None
    
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can handle the error"""
        return True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the fallback strategy"""
        raise NotImplementedError
    
    def get_success_rate(self) -> float:
        """Get success rate of this strategy"""
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.0


class ConservativeMappingFallback(FallbackStrategy):
    """Fallback using conservative PCA estimates"""
    
    def __init__(self):
        super().__init__("conservative_mapping", priority=1)
        
        # Load or create conservative estimates
        self.conservative_estimates = self._create_conservative_estimates()
    
    def _create_conservative_estimates(self) -> Dict[str, np.ndarray]:
        """Create conservative PCA estimates based on typical patterns"""
        estimates = {}
        
        # Conservative estimates for different merchant categories
        estimates['grocery'] = np.random.normal(0, 0.5, 28)
        estimates['restaurant'] = np.random.normal(0, 0.6, 28)
        estimates['gas'] = np.random.normal(0, 0.4, 28)
        estimates['online'] = np.random.normal(0, 0.8, 28)
        estimates['atm'] = np.random.normal(0, 0.7, 28)
        estimates['default'] = np.zeros(28)
        
        return estimates
    
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Can handle feature mapping and prediction errors"""
        return error_info.category in [
            ErrorCategory.FEATURE_MAPPING,
            ErrorCategory.PREDICTION,
            ErrorCategory.MODEL_LOADING
        ]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conservative mapping fallback"""
        try:
            user_input = context.get('user_input')
            if not user_input:
                raise ValueError("No user input provided in context")
            
            # Get conservative estimates
            merchant_key = user_input.merchant_category.value
            if merchant_key in self.conservative_estimates:
                pca_estimates = self.conservative_estimates[merchant_key].copy()
            else:
                pca_estimates = self.conservative_estimates['default'].copy()
            
            # Add some variation based on amount and risk
            amount_factor = min(user_input.transaction_amount / 1000, 2.0)
            risk_factor = self._get_risk_factor(user_input)
            
            pca_estimates = pca_estimates * (1 + amount_factor * 0.1 + risk_factor * 0.2)
            
            # Create conservative result
            result = {
                'prediction': 0,  # Conservative: assume legitimate
                'fraud_probability': min(0.1 + risk_factor * 0.3, 0.5),  # Conservative probability
                'normal_probability': max(0.5, 0.9 - risk_factor * 0.3),
                'risk_score': min(10 + risk_factor * 30, 50),  # Conservative risk score
                'mapping_confidence': 0.6,  # Lower confidence for fallback
                'confidence_level': 'acceptable',
                'model_used': 'conservative_fallback',
                'mapper_used': 'conservative',
                'fallback_used': True,
                'fallback_reason': 'Primary mapping failed',
                'pca_estimates': {f'V{i+1}': float(pca_estimates[i]) for i in range(28)}
            }
            
            self.success_count += 1
            self.last_used = time.time()
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            raise Exception(f"Conservative fallback failed: {str(e)}")
    
    def _get_risk_factor(self, user_input: UserTransactionInput) -> float:
        """Calculate risk factor from user input"""
        risk_factor = 0.0
        
        # Location risk
        location_risks = {
            'normal': 0.0,
            'slightly_unusual': 0.3,
            'highly_unusual': 0.6,
            'foreign': 0.8
        }
        risk_factor += location_risks.get(user_input.location_risk.value, 0.5)
        
        # Spending pattern
        spending_risks = {
            'typical': 0.0,
            'slightly_higher': 0.2,
            'much_higher': 0.5,
            'suspicious': 0.9
        }
        risk_factor += spending_risks.get(user_input.spending_pattern.value, 0.3)
        
        # Time risk (late night/early morning)
        hour = user_input.time_context.hour_of_day
        if hour <= 5 or hour >= 23:
            risk_factor += 0.3
        
        return min(risk_factor, 1.0)


class DatasetAverageFallback(FallbackStrategy):
    """Fallback using dataset averages"""
    
    def __init__(self, dataset_stats_path: Optional[str] = None):
        super().__init__("dataset_average", priority=2)
        self.dataset_stats = self._load_dataset_stats(dataset_stats_path)
    
    def _load_dataset_stats(self, stats_path: Optional[str]) -> Dict[str, Any]:
        """Load dataset statistics or create defaults"""
        if stats_path and Path(stats_path).exists():
            try:
                import joblib
                return joblib.load(stats_path)
            except Exception as e:
                fraud_logger.warning(f"Could not load dataset stats: {e}")
        
        # Default statistics (approximated from ULB dataset)
        return {
            'V1': {'mean': 0.0, 'std': 1.0},
            'V2': {'mean': 0.0, 'std': 1.0},
            'V3': {'mean': 0.0, 'std': 1.0},
            # ... (would include all V1-V28 in real implementation)
            **{f'V{i}': {'mean': 0.0, 'std': 1.0} for i in range(1, 29)}
        }
    
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Can handle feature mapping errors"""
        return error_info.category == ErrorCategory.FEATURE_MAPPING
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dataset average fallback"""
        try:
            # Generate PCA estimates using dataset averages
            pca_estimates = {}
            for i in range(1, 29):
                feature_name = f'V{i}'
                if feature_name in self.dataset_stats:
                    stats = self.dataset_stats[feature_name]
                    # Use mean with small random variation
                    estimate = np.random.normal(stats['mean'], stats['std'] * 0.1)
                    pca_estimates[feature_name] = float(estimate)
                else:
                    pca_estimates[feature_name] = 0.0
            
            result = {
                'pca_estimates': pca_estimates,
                'mapping_confidence': 0.5,
                'fallback_used': True,
                'fallback_reason': 'Using dataset averages'
            }
            
            self.success_count += 1
            self.last_used = time.time()
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            raise Exception(f"Dataset average fallback failed: {str(e)}")


class SimplifiedModelFallback(FallbackStrategy):
    """Fallback using simplified linear model"""
    
    def __init__(self):
        super().__init__("simplified_model", priority=3)
        self.simple_weights = self._create_simple_weights()
    
    def _create_simple_weights(self) -> np.ndarray:
        """Create simple linear weights for mapping"""
        # Simple weights based on typical feature importance
        # [amount, merchant, location, spending, hour_sin, hour_cos, day, weekend]
        weights = np.array([
            [0.3, 0.2, 0.1, 0.3, 0.05, 0.05, 0.0, 0.0],  # V1
            [0.2, 0.3, 0.2, 0.2, 0.05, 0.05, 0.0, 0.0],  # V2
            [0.1, 0.1, 0.4, 0.3, 0.05, 0.05, 0.0, 0.0],  # V3
            # ... simplified weights for all 28 components
        ])
        
        # Create full weight matrix (28 x 8)
        full_weights = np.random.normal(0, 0.1, (28, 8))
        full_weights[:3] = weights  # Use defined weights for first 3 components
        
        return full_weights
    
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Can handle feature mapping errors"""
        return error_info.category == ErrorCategory.FEATURE_MAPPING
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simplified model fallback"""
        try:
            user_input = context.get('user_input')
            if not user_input:
                raise ValueError("No user input provided in context")
            
            # Convert user input to feature array
            features = self._convert_user_input_to_features(user_input)
            
            # Apply simple linear transformation
            pca_estimates_array = np.dot(self.simple_weights, features)
            
            # Convert to dictionary
            pca_estimates = {f'V{i+1}': float(pca_estimates_array[i]) for i in range(28)}
            
            result = {
                'pca_estimates': pca_estimates,
                'mapping_confidence': 0.4,
                'fallback_used': True,
                'fallback_reason': 'Using simplified linear model'
            }
            
            self.success_count += 1
            self.last_used = time.time()
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            raise Exception(f"Simplified model fallback failed: {str(e)}")
    
    def _convert_user_input_to_features(self, user_input: UserTransactionInput) -> np.ndarray:
        """Convert user input to feature array"""
        features = []
        
        # Transaction amount (normalized)
        features.append(min(user_input.transaction_amount / 1000, 5.0))
        
        # Merchant category (encoded)
        merchant_categories = ['grocery', 'gas', 'online', 'restaurant', 'atm', 
                             'department_store', 'pharmacy', 'entertainment', 'travel', 'other']
        merchant_encoded = merchant_categories.index(user_input.merchant_category.value) \
                          if user_input.merchant_category.value in merchant_categories else 9
        features.append(merchant_encoded / 10.0)
        
        # Location risk (encoded)
        location_risks = ['normal', 'slightly_unusual', 'highly_unusual', 'foreign']
        location_encoded = location_risks.index(user_input.location_risk.value) \
                          if user_input.location_risk.value in location_risks else 0
        features.append(location_encoded / 4.0)
        
        # Spending pattern (encoded)
        spending_patterns = ['typical', 'slightly_higher', 'much_higher', 'suspicious']
        spending_encoded = spending_patterns.index(user_input.spending_pattern.value) \
                          if user_input.spending_pattern.value in spending_patterns else 0
        features.append(spending_encoded / 4.0)
        
        # Time features
        hour = user_input.time_context.hour_of_day
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        features.extend([hour_sin, hour_cos])
        
        # Day and weekend
        features.append(user_input.time_context.day_of_week / 7.0)
        features.append(float(user_input.time_context.is_weekend))
        
        return np.array(features)


class ErrorHandler:
    """Comprehensive error handler for the feature mapping pipeline"""
    
    def __init__(self):
        self.fallback_strategies: List[FallbackStrategy] = []
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0
        }
        
        # Initialize fallback strategies
        self._initialize_fallback_strategies()
    
    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies in priority order"""
        self.fallback_strategies = [
            ConservativeMappingFallback(),
            DatasetAverageFallback(),
            SimplifiedModelFallback()
        ]
        
        # Sort by priority
        self.fallback_strategies.sort(key=lambda x: x.priority)
        
        fraud_logger.info(f"Initialized {len(self.fallback_strategies)} fallback strategies")
    
    def handle_error(self, 
                    error: Exception,
                    context: Dict[str, Any],
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Dict[str, Any]:
        """
        Handle error with appropriate fallback strategy
        
        Args:
            error: The exception that occurred
            context: Context information for recovery
            category: Error category
            severity: Error severity
            
        Returns:
            Result from fallback strategy or error information
        """
        # Create error info
        error_info = ErrorInfo(
            error_id=f"err_{int(time.time() * 1000)}",
            category=category,
            severity=severity,
            message=str(error),
            details=traceback.format_exc(),
            timestamp=time.time(),
            context=context.copy(),
            stack_trace=traceback.format_exc()
        )
        
        # Log error
        self._log_error(error_info)
        
        # Update statistics
        self.error_history.append(error_info)
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        self.recovery_stats['total_errors'] += 1
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_info, context)
        
        if recovery_result.get('success', False):
            error_info.recovery_attempted = True
            error_info.recovery_successful = True
            self.recovery_stats['recovered_errors'] += 1
            
            # Return successful recovery result
            result = recovery_result['result']
            result['error_handled'] = True
            result['error_id'] = error_info.error_id
            result['fallback_strategy'] = recovery_result.get('strategy_name')
            
            return result
        else:
            error_info.recovery_attempted = True
            error_info.recovery_successful = False
            self.recovery_stats['failed_recoveries'] += 1
            
            # Return error result
            return {
                'error': str(error),
                'error_id': error_info.error_id,
                'error_category': category.value,
                'error_severity': severity.value,
                'recovery_attempted': True,
                'recovery_successful': False,
                'timestamp': error_info.timestamp
            }
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information"""
        log_level = {
            ErrorSeverity.LOW: fraud_logger.info,
            ErrorSeverity.MEDIUM: fraud_logger.warning,
            ErrorSeverity.HIGH: fraud_logger.error,
            ErrorSeverity.CRITICAL: fraud_logger.critical
        }.get(error_info.severity, fraud_logger.error)
        
        log_level(
            f"Error {error_info.error_id} [{error_info.category.value}]: "
            f"{error_info.message}"
        )
        
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            fraud_logger.error(f"Stack trace: {error_info.stack_trace}")
    
    def _attempt_recovery(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery using fallback strategies"""
        
        # Find suitable fallback strategies
        suitable_strategies = [
            strategy for strategy in self.fallback_strategies
            if strategy.can_handle(error_info)
        ]
        
        if not suitable_strategies:
            return {'success': False, 'reason': 'No suitable fallback strategies'}
        
        # Try strategies in priority order
        for strategy in suitable_strategies:
            try:
                fraud_logger.info(f"Attempting recovery with strategy: {strategy.name}")
                
                result = strategy.execute(context)
                
                fraud_logger.info(f"Recovery successful with strategy: {strategy.name}")
                return {
                    'success': True,
                    'result': result,
                    'strategy_name': strategy.name
                }
                
            except Exception as fallback_error:
                fraud_logger.warning(
                    f"Fallback strategy {strategy.name} failed: {str(fallback_error)}"
                )
                continue
        
        return {'success': False, 'reason': 'All fallback strategies failed'}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'recovery_rate': 0.0,
                'error_categories': {},
                'error_severities': {},
                'fallback_performance': {}
            }
        
        # Error categories
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = self.error_counts.get(category, 0)
        
        # Error severities
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Fallback strategy performance
        fallback_performance = {}
        for strategy in self.fallback_strategies:
            fallback_performance[strategy.name] = {
                'success_count': strategy.success_count,
                'failure_count': strategy.failure_count,
                'success_rate': strategy.get_success_rate(),
                'last_used': strategy.last_used
            }
        
        # Recovery rate
        recovery_rate = (self.recovery_stats['recovered_errors'] / total_errors * 100) \
                       if total_errors > 0 else 0
        
        return {
            'total_errors': total_errors,
            'recovery_stats': self.recovery_stats,
            'recovery_rate_percent': round(recovery_rate, 2),
            'error_categories': category_counts,
            'error_severities': severity_counts,
            'fallback_performance': fallback_performance,
            'recent_errors': [
                {
                    'error_id': error.error_id,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'message': error.message,
                    'timestamp': error.timestamp,
                    'recovered': error.recovery_successful
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def clear_error_history(self, keep_recent: int = 100):
        """Clear old error history, keeping recent entries"""
        if len(self.error_history) > keep_recent:
            self.error_history = self.error_history[-keep_recent:]
        
        fraud_logger.info(f"Error history cleared, keeping {len(self.error_history)} recent entries")
    
    def add_custom_fallback(self, strategy: FallbackStrategy):
        """Add custom fallback strategy"""
        self.fallback_strategies.append(strategy)
        self.fallback_strategies.sort(key=lambda x: x.priority)
        
        fraud_logger.info(f"Added custom fallback strategy: {strategy.name}")
    
    def test_fallback_strategies(self, test_context: Dict[str, Any]) -> Dict[str, Any]:
        """Test all fallback strategies with given context"""
        results = {}
        
        for strategy in self.fallback_strategies:
            try:
                start_time = time.time()
                result = strategy.execute(test_context)
                execution_time = (time.time() - start_time) * 1000
                
                results[strategy.name] = {
                    'success': True,
                    'execution_time_ms': execution_time,
                    'result_keys': list(result.keys()) if isinstance(result, dict) else None
                }
                
            except Exception as e:
                results[strategy.name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results


# Global error handler instance
error_handler = ErrorHandler()