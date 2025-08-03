"""
Professional Performance Monitor for Real-time Model Monitoring
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from scipy import stats
import warnings

from .model_metadata import PerformanceMetrics
from ..logger import fraud_logger


class PerformanceMonitor:
    """Real-time model performance monitoring and drift detection"""
    
    def __init__(self):
        self.baseline_data = {}
        self.drift_threshold = 0.1
        self.performance_window = 100  # Number of recent predictions to consider
        fraud_logger.info("Performance Monitor initialized")
    
    def set_baseline(self, model_id: str, X_baseline: np.ndarray):
        """Set baseline data distribution for drift detection"""
        try:
            self.baseline_data[model_id] = {
                'mean': np.mean(X_baseline, axis=0),
                'std': np.std(X_baseline, axis=0),
                'feature_stats': self._calculate_feature_stats(X_baseline),
                'timestamp': datetime.now()
            }
            fraud_logger.info(f"Baseline set for model {model_id}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to set baseline for {model_id}: {e}")
    
    def calculate_drift_score(self, X_current: np.ndarray, model_id: str = None) -> float:
        """Calculate data drift score using statistical tests"""
        try:
            if model_id and model_id in self.baseline_data:
                baseline = self.baseline_data[model_id]
                return self._calculate_statistical_drift(X_current, baseline)
            else:
                # Simplified drift calculation without baseline
                return self._calculate_simple_drift(X_current)
                
        except Exception as e:
            fraud_logger.error(f"Failed to calculate drift score: {e}")
            return 0.0
    
    def detect_anomalies(self, X_data: np.ndarray, model_id: str = None) -> Dict[str, Any]:
        """Detect anomalies in input data"""
        try:
            anomaly_info = {
                'anomaly_count': 0,
                'anomaly_indices': [],
                'anomaly_scores': [],
                'feature_anomalies': {}
            }
            
            if model_id and model_id in self.baseline_data:
                baseline = self.baseline_data[model_id]
                anomaly_info = self._detect_statistical_anomalies(X_data, baseline)
            else:
                anomaly_info = self._detect_simple_anomalies(X_data)
            
            return anomaly_info
            
        except Exception as e:
            fraud_logger.error(f"Failed to detect anomalies: {e}")
            return {'anomaly_count': 0, 'anomaly_indices': [], 'anomaly_scores': []}
    
    def monitor_prediction_distribution(self, predictions: np.ndarray, 
                                      prediction_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Monitor the distribution of model predictions"""
        try:
            distribution_info = {
                'prediction_mean': float(np.mean(predictions)),
                'prediction_std': float(np.std(predictions)),
                'class_distribution': {},
                'confidence_stats': {}
            }
            
            # Analyze class distribution
            unique, counts = np.unique(predictions, return_counts=True)
            total_predictions = len(predictions)
            
            for class_val, count in zip(unique, counts):
                distribution_info['class_distribution'][str(class_val)] = {
                    'count': int(count),
                    'percentage': float(count / total_predictions * 100)
                }
            
            # Analyze confidence distribution if probabilities available
            if prediction_probabilities is not None:
                max_probs = np.max(prediction_probabilities, axis=1)
                distribution_info['confidence_stats'] = {
                    'mean_confidence': float(np.mean(max_probs)),
                    'std_confidence': float(np.std(max_probs)),
                    'low_confidence_count': int(np.sum(max_probs < 0.6)),
                    'high_confidence_count': int(np.sum(max_probs > 0.9))
                }
            
            return distribution_info
            
        except Exception as e:
            fraud_logger.error(f"Failed to monitor prediction distribution: {e}")
            return {}
    
    def calculate_feature_importance_drift(self, current_importance: Dict[str, float],
                                         baseline_importance: Dict[str, float]) -> float:
        """Calculate drift in feature importance"""
        try:
            if not baseline_importance:
                return 0.0
            
            # Calculate KL divergence between importance distributions
            current_values = np.array([current_importance.get(k, 0) for k in baseline_importance.keys()])
            baseline_values = np.array(list(baseline_importance.values()))
            
            # Normalize to create probability distributions
            current_values = current_values / np.sum(current_values) if np.sum(current_values) > 0 else current_values
            baseline_values = baseline_values / np.sum(baseline_values) if np.sum(baseline_values) > 0 else baseline_values
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            current_values = current_values + epsilon
            baseline_values = baseline_values + epsilon
            
            # Calculate KL divergence
            kl_div = np.sum(current_values * np.log(current_values / baseline_values))
            
            return float(kl_div)
            
        except Exception as e:
            fraud_logger.error(f"Failed to calculate feature importance drift: {e}")
            return 0.0
    
    def generate_performance_alert(self, metrics: PerformanceMetrics, 
                                 thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate performance alert if thresholds are breached"""
        try:
            if not metrics.is_degraded(thresholds):
                return None
            
            alert = {
                'alert_id': f"perf_alert_{metrics.model_id}_{int(metrics.timestamp.timestamp())}",
                'model_id': metrics.model_id,
                'timestamp': metrics.timestamp.isoformat(),
                'severity': self._calculate_alert_severity(metrics, thresholds),
                'degraded_metrics': metrics.get_degradation_reasons(thresholds),
                'current_performance': {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'auc_roc': metrics.auc_roc,
                    'drift_score': metrics.drift_score
                },
                'recommended_actions': self._get_recommended_actions(metrics, thresholds)
            }
            
            fraud_logger.warning(f"Performance alert generated: {alert['alert_id']}")
            return alert
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate performance alert: {e}")
            return None
    
    def _calculate_feature_stats(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive feature statistics"""
        stats_dict = {}
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            stats_dict[f'feature_{i}'] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'q25': float(np.percentile(feature_data, 25)),
                'q50': float(np.percentile(feature_data, 50)),
                'q75': float(np.percentile(feature_data, 75)),
                'skewness': float(stats.skew(feature_data)),
                'kurtosis': float(stats.kurtosis(feature_data))
            }
        
        return stats_dict
    
    def _calculate_statistical_drift(self, X_current: np.ndarray, baseline: Dict[str, Any]) -> float:
        """Calculate drift using statistical tests"""
        try:
            drift_scores = []
            
            for i in range(min(X_current.shape[1], len(baseline['mean']))):
                current_feature = X_current[:, i]
                baseline_mean = baseline['mean'][i]
                baseline_std = baseline['std'][i]
                
                # Z-score based drift
                if baseline_std > 0:
                    current_mean = np.mean(current_feature)
                    z_score = abs(current_mean - baseline_mean) / baseline_std
                    drift_scores.append(min(z_score / 3.0, 1.0))  # Normalize to [0, 1]
                else:
                    drift_scores.append(0.0)
            
            return float(np.mean(drift_scores))
            
        except Exception as e:
            fraud_logger.error(f"Failed to calculate statistical drift: {e}")
            return 0.0
    
    def _calculate_simple_drift(self, X_current: np.ndarray) -> float:
        """Simple drift calculation without baseline"""
        try:
            # Use coefficient of variation as a simple drift indicator
            feature_cvs = []
            
            for i in range(X_current.shape[1]):
                feature_data = X_current[:, i]
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    feature_cvs.append(cv)
            
            if feature_cvs:
                # High coefficient of variation might indicate drift
                avg_cv = np.mean(feature_cvs)
                return min(avg_cv / 2.0, 1.0)  # Normalize
            
            return 0.0
            
        except Exception as e:
            fraud_logger.error(f"Failed to calculate simple drift: {e}")
            return 0.0
    
    def _detect_statistical_anomalies(self, X_data: np.ndarray, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using statistical methods"""
        anomaly_scores = []
        anomaly_indices = []
        
        try:
            for i, sample in enumerate(X_data):
                score = 0.0
                
                for j, feature_val in enumerate(sample):
                    if j < len(baseline['mean']):
                        mean_val = baseline['mean'][j]
                        std_val = baseline['std'][j]
                        
                        if std_val > 0:
                            z_score = abs(feature_val - mean_val) / std_val
                            score += z_score
                
                anomaly_scores.append(score / len(sample))
                
                # Consider anomaly if z-score > 3 (99.7% confidence)
                if score / len(sample) > 3.0:
                    anomaly_indices.append(i)
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': anomaly_scores,
                'feature_anomalies': {}
            }
            
        except Exception as e:
            fraud_logger.error(f"Failed to detect statistical anomalies: {e}")
            return {'anomaly_count': 0, 'anomaly_indices': [], 'anomaly_scores': []}
    
    def _detect_simple_anomalies(self, X_data: np.ndarray) -> Dict[str, Any]:
        """Simple anomaly detection using IQR method"""
        try:
            anomaly_indices = set()
            
            for j in range(X_data.shape[1]):
                feature_data = X_data[:, j]
                q1 = np.percentile(feature_data, 25)
                q3 = np.percentile(feature_data, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Find outliers
                outlier_indices = np.where((feature_data < lower_bound) | (feature_data > upper_bound))[0]
                anomaly_indices.update(outlier_indices)
            
            anomaly_indices = list(anomaly_indices)
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': [1.0] * len(anomaly_indices),
                'feature_anomalies': {}
            }
            
        except Exception as e:
            fraud_logger.error(f"Failed to detect simple anomalies: {e}")
            return {'anomaly_count': 0, 'anomaly_indices': [], 'anomaly_scores': []}
    
    def _calculate_alert_severity(self, metrics: PerformanceMetrics, thresholds: Dict[str, float]) -> str:
        """Calculate alert severity based on performance degradation"""
        degraded_count = len(metrics.get_degradation_reasons(thresholds))
        
        if degraded_count >= 4:
            return "critical"
        elif degraded_count >= 2:
            return "high"
        elif degraded_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_actions(self, metrics: PerformanceMetrics, thresholds: Dict[str, float]) -> List[str]:
        """Get recommended actions based on performance issues"""
        actions = []
        
        if metrics.accuracy < thresholds.get('accuracy', 0.95):
            actions.append("Review model training data quality and consider retraining")
        
        if metrics.precision < thresholds.get('precision', 0.90):
            actions.append("Investigate false positive rate and adjust decision threshold")
        
        if metrics.recall < thresholds.get('recall', 0.85):
            actions.append("Analyze false negative cases and consider feature engineering")
        
        if metrics.drift_score > thresholds.get('drift_score', 0.1):
            actions.append("Data drift detected - validate input data pipeline and consider model update")
        
        if metrics.error_rate > thresholds.get('error_rate', 0.05):
            actions.append("High error rate detected - check model deployment and data preprocessing")
        
        if not actions:
            actions.append("Monitor performance trends and investigate root causes")
        
        return actions