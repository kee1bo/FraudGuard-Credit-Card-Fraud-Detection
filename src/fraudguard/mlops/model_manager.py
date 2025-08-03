"""
Professional ML Model Manager for Lifecycle Management
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .model_metadata import ModelMetadata, ModelVersion, PerformanceMetrics
from .model_registry import ModelRegistry
from .performance_monitor import PerformanceMonitor
from ..logger import fraud_logger
from ..exception import FraudGuardException


class MLModelManager:
    """Professional ML model lifecycle management"""
    
    def __init__(self, registry_path: str = "artifacts/model_registry"):
        self.registry = ModelRegistry(registry_path)
        self.monitor = PerformanceMonitor()
        self.performance_thresholds = {
            'accuracy': 0.99,
            'precision': 0.95,
            'recall': 0.90,
            'f1_score': 0.92,
            'auc_roc': 0.95,
            'drift_score': 0.1,
            'error_rate': 0.01
        }
        fraud_logger.info("ML Model Manager initialized")
    
    def register_model(self, model, metadata: ModelMetadata, artifacts_path: str) -> str:
        """Register model with comprehensive metadata"""
        try:
            # Calculate model size
            if Path(artifacts_path).exists():
                model_files = list(Path(artifacts_path).glob("**/*"))
                total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                metadata.model_size_mb = total_size / (1024 * 1024)
            
            # Register model in registry
            model_id = self.registry.register_model(metadata)
            
            # Create initial version
            version = ModelVersion(
                model_id=model_id,
                version="1.0.0",
                performance_metrics=metadata.performance_metrics,
                training_config=metadata.hyperparameters,
                is_active=True,
                deployment_stage="development"
            )
            
            version_id = self.registry.register_version(version, artifacts_path)
            
            fraud_logger.info(f"Model registered successfully: {metadata.name} ({model_id})")
            return model_id
            
        except Exception as e:
            fraud_logger.error(f"Failed to register model: {e}")
            raise FraudGuardException(f"Model registration failed: {e}")
    
    def deploy_model(self, model_id: str, stage: str = "production") -> bool:
        """Deploy model to specified stage with validation"""
        try:
            # Get active version
            active_version = self.registry.get_active_version(model_id)
            if not active_version:
                raise FraudGuardException(f"No active version found for model {model_id}")
            
            # Validate performance before deployment
            if stage == "production":
                if not self._validate_for_production(active_version):
                    raise FraudGuardException("Model does not meet production requirements")
            
            # Update deployment stage
            active_version.deployment_stage = stage
            self.registry.register_version(active_version, active_version.model_artifacts.get('path', ''))
            
            fraud_logger.info(f"Model {model_id} deployed to {stage}")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Failed to deploy model {model_id}: {e}")
            return False
    
    def monitor_performance(self, model_id: str, X_test: np.ndarray, y_test: np.ndarray) -> PerformanceMetrics:
        """Real-time model performance monitoring"""
        try:
            # Load active model
            model, version = self._load_active_model(model_id)
            if not model:
                raise FraudGuardException(f"Could not load active model for {model_id}")
            
            # Make predictions and measure performance
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            prediction_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Calculate metrics
            metrics = PerformanceMetrics(
                model_id=model_id,
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1_score=f1_score(y_test, y_pred, zero_division=0),
                auc_roc=roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0,
                prediction_latency=prediction_time / len(X_test),
                throughput=len(X_test) / (prediction_time / 1000) if prediction_time > 0 else 0,
                sample_count=len(X_test),
                error_rate=np.mean(y_pred != y_test)
            )
            
            # Calculate drift score (simplified)
            metrics.drift_score = self.monitor.calculate_drift_score(X_test)
            
            # Calculate confidence distribution
            if y_pred_proba is not None:
                metrics.confidence_distribution = {
                    'low': np.mean((y_pred_proba >= 0.0) & (y_pred_proba < 0.3)),
                    'medium': np.mean((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)),
                    'high': np.mean((y_pred_proba >= 0.7) & (y_pred_proba <= 1.0))
                }
            
            # Record performance
            self.registry.record_performance(model_id, version.version_id, metrics)
            
            # Check for degradation
            if metrics.is_degraded(self.performance_thresholds):
                self._handle_performance_degradation(model_id, metrics)
            
            return metrics
            
        except Exception as e:
            fraud_logger.error(f"Failed to monitor performance for {model_id}: {e}")
            raise FraudGuardException(f"Performance monitoring failed: {e}")
    
    def trigger_retraining(self, model_id: str, reason: str) -> str:
        """Automated model retraining workflow"""
        try:
            fraud_logger.info(f"Triggering retraining for {model_id}: {reason}")
            
            # Get model metadata
            model_metadata = self.registry.get_model(model_id)
            if not model_metadata:
                raise FraudGuardException(f"Model {model_id} not found")
            
            # Create retraining job (placeholder - would integrate with actual training pipeline)
            retraining_job_id = f"retrain_{model_id}_{int(time.time())}"
            
            # Log retraining trigger
            fraud_logger.info(f"Retraining job {retraining_job_id} created for model {model_id}")
            
            # In a real implementation, this would:
            # 1. Fetch latest training data
            # 2. Retrain the model with updated hyperparameters
            # 3. Validate the new model
            # 4. Create a new version if validation passes
            # 5. Optionally auto-deploy if performance improves
            
            return retraining_job_id
            
        except Exception as e:
            fraud_logger.error(f"Failed to trigger retraining for {model_id}: {e}")
            raise FraudGuardException(f"Retraining trigger failed: {e}")
    
    def rollback_model(self, model_id: str, target_version: str = None) -> bool:
        """Rollback to a previous model version"""
        try:
            versions = self.registry.list_model_versions(model_id)
            if not versions:
                raise FraudGuardException(f"No versions found for model {model_id}")
            
            if target_version:
                # Rollback to specific version
                target = next((v for v in versions if v.version == target_version), None)
                if not target:
                    raise FraudGuardException(f"Version {target_version} not found")
            else:
                # Rollback to previous version
                active_versions = [v for v in versions if v.is_active]
                if not active_versions:
                    raise FraudGuardException("No active version found")
                
                current_version = active_versions[0]
                previous_versions = [v for v in versions if v.created_at < current_version.created_at]
                if not previous_versions:
                    raise FraudGuardException("No previous version available for rollback")
                
                target = max(previous_versions, key=lambda v: v.created_at)
            
            # Set target version as active
            success = self.registry.set_active_version(model_id, target.version_id)
            
            if success:
                fraud_logger.info(f"Model {model_id} rolled back to version {target.version}")
            
            return success
            
        except Exception as e:
            fraud_logger.error(f"Failed to rollback model {model_id}: {e}")
            return False
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model status"""
        try:
            model_metadata = self.registry.get_model(model_id)
            active_version = self.registry.get_active_version(model_id)
            performance_history = self.registry.get_performance_history(model_id, limit=10)
            
            if not model_metadata:
                return {'error': f'Model {model_id} not found'}
            
            latest_performance = performance_history[0] if performance_history else None
            
            status = {
                'model_id': model_id,
                'name': model_metadata.name,
                'algorithm': model_metadata.algorithm,
                'created_at': model_metadata.created_at.isoformat(),
                'deployment_status': model_metadata.deployment_status,
                'active_version': active_version.version if active_version else None,
                'deployment_stage': active_version.deployment_stage if active_version else None,
                'latest_performance': latest_performance.to_dict() if latest_performance else None,
                'performance_trend': self._calculate_performance_trend(performance_history),
                'health_status': self._calculate_health_status(latest_performance),
                'tags': model_metadata.tags,
                'model_size_mb': model_metadata.model_size_mb
            }
            
            return status
            
        except Exception as e:
            fraud_logger.error(f"Failed to get model status for {model_id}: {e}")
            return {'error': str(e)}
    
    def _load_active_model(self, model_id: str) -> Tuple[Any, Optional[ModelVersion]]:
        """Load the active model for a given model ID"""
        try:
            active_version = self.registry.get_active_version(model_id)
            if not active_version:
                return None, None
            
            model_path = Path(active_version.model_artifacts.get('path', '')) / "model.pkl"
            if not model_path.exists():
                fraud_logger.error(f"Model file not found: {model_path}")
                return None, None
            
            model = joblib.load(model_path)
            return model, active_version
            
        except Exception as e:
            fraud_logger.error(f"Failed to load active model {model_id}: {e}")
            return None, None
    
    def _validate_for_production(self, version: ModelVersion) -> bool:
        """Validate model version meets production requirements"""
        metrics = version.performance_metrics
        
        # Check minimum performance thresholds
        checks = [
            metrics.get('accuracy', 0) >= self.performance_thresholds['accuracy'],
            metrics.get('precision', 0) >= self.performance_thresholds['precision'],
            metrics.get('recall', 0) >= self.performance_thresholds['recall'],
            metrics.get('f1_score', 0) >= self.performance_thresholds['f1_score'],
            metrics.get('auc_roc', 0) >= self.performance_thresholds['auc_roc']
        ]
        
        return all(checks)
    
    def _handle_performance_degradation(self, model_id: str, metrics: PerformanceMetrics):
        """Handle performance degradation detection"""
        reasons = metrics.get_degradation_reasons(self.performance_thresholds)
        fraud_logger.warning(f"Performance degradation detected for {model_id}: {reasons}")
        
        # Trigger retraining if degradation is severe
        if len(reasons) >= 3:  # Multiple metrics degraded
            self.trigger_retraining(model_id, f"Performance degradation: {', '.join(reasons)}")
    
    def _calculate_performance_trend(self, history: List[PerformanceMetrics]) -> str:
        """Calculate performance trend from history"""
        if len(history) < 2:
            return "insufficient_data"
        
        recent_f1 = np.mean([m.f1_score for m in history[:3]])
        older_f1 = np.mean([m.f1_score for m in history[-3:]])
        
        if recent_f1 > older_f1 + 0.01:
            return "improving"
        elif recent_f1 < older_f1 - 0.01:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_health_status(self, latest_performance: Optional[PerformanceMetrics]) -> str:
        """Calculate overall model health status"""
        if not latest_performance:
            return "unknown"
        
        if latest_performance.is_degraded(self.performance_thresholds):
            return "unhealthy"
        elif latest_performance.f1_score >= 0.95:
            return "excellent"
        elif latest_performance.f1_score >= 0.90:
            return "good"
        else:
            return "fair"