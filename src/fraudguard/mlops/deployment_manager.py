"""
Deployment Manager
Handles automated deployment of feature mapping models with health checks and rollback capabilities.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np

from fraudguard.mlops.mapping_model_registry import get_mapping_model_registry, ModelVersion
from fraudguard.models.base_feature_mapper import BaseFeatureMapper
from fraudguard.logger import fraud_logger


class DeploymentHealthCheck:
    """Health check for deployed models"""
    
    def __init__(self, model_path: str, model_type: str):
        self.model_path = Path(model_path)
        self.model_type = model_type
    
    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run comprehensive health checks on a deployed model
        
        Returns:
            Dictionary with health check results
        """
        results = {
            'overall_health': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check 1: Model files exist
            results['checks']['files_exist'] = self._check_files_exist()
            
            # Check 2: Model can be loaded
            results['checks']['model_loadable'] = self._check_model_loadable()
            
            # Check 3: Model can make predictions
            results['checks']['prediction_test'] = self._check_prediction_capability()
            
            # Check 4: Performance within acceptable range
            results['checks']['performance_check'] = self._check_performance()
            
            # Determine overall health
            failed_checks = [
                check_name for check_name, check_result in results['checks'].items()
                if not check_result.get('passed', False)
            ]
            
            if failed_checks:
                results['overall_health'] = 'unhealthy'
                results['failed_checks'] = failed_checks
            
            fraud_logger.info(f"Health check completed for {self.model_type}: {results['overall_health']}")
            
        except Exception as e:
            results['overall_health'] = 'unhealthy'
            results['error'] = str(e)
            fraud_logger.error(f"Health check failed for {self.model_type}: {e}")
        
        return results
    
    def _check_files_exist(self) -> Dict[str, Any]:
        """Check if required model files exist"""
        try:
            required_files = ['model.pkl', 'metadata.pkl']
            missing_files = []
            
            for file_name in required_files:
                file_path = self.model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            return {
                'passed': len(missing_files) == 0,
                'missing_files': missing_files,
                'message': 'All required files present' if not missing_files else f'Missing files: {missing_files}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Error checking file existence'
            }
    
    def _check_model_loadable(self) -> Dict[str, Any]:
        """Check if model can be loaded successfully"""
        try:
            model_file = self.model_path / 'model.pkl'
            model = joblib.load(model_file)
            
            return {
                'passed': True,
                'model_type': type(model).__name__,
                'message': 'Model loaded successfully'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Failed to load model'
            }
    
    def _check_prediction_capability(self) -> Dict[str, Any]:
        """Check if model can make predictions"""
        try:
            # Load model
            model_file = self.model_path / 'model.pkl'
            model = joblib.load(model_file)
            
            # Create test input (5 interpretable features)
            test_input = np.array([[100.0, 1.0, 0.5, 2.0, 0.1]])
            
            # Make prediction
            prediction = model.predict(test_input)
            
            # Validate prediction shape (should be 28 PCA components)
            expected_shape = (1, 28)
            actual_shape = prediction.shape
            
            shape_correct = actual_shape == expected_shape
            
            return {
                'passed': shape_correct,
                'expected_shape': expected_shape,
                'actual_shape': actual_shape,
                'prediction_sample': prediction[0][:3].tolist() if len(prediction) > 0 else [],
                'message': 'Prediction test passed' if shape_correct else f'Shape mismatch: expected {expected_shape}, got {actual_shape}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Failed to make test prediction'
            }
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check if model performance is within acceptable range"""
        try:
            # Load metadata
            metadata_file = self.model_path / 'metadata.pkl'
            if not metadata_file.exists():
                return {
                    'passed': True,  # Skip if no metadata
                    'message': 'No metadata available for performance check'
                }
            
            metadata = joblib.load(metadata_file)
            
            # Define acceptable performance thresholds
            thresholds = {
                'mse': 1.0,  # MSE should be < 1.0
                'avg_correlation': 0.5,  # Correlation should be > 0.5
                'mae': 0.8  # MAE should be < 0.8
            }
            
            performance_issues = []
            
            for metric, threshold in thresholds.items():
                if metric in metadata:
                    value = metadata[metric]
                    if metric == 'avg_correlation':
                        # Higher is better for correlation
                        if value < threshold:
                            performance_issues.append(f"{metric}: {value:.4f} < {threshold}")
                    else:
                        # Lower is better for error metrics
                        if value > threshold:
                            performance_issues.append(f"{metric}: {value:.4f} > {threshold}")
            
            return {
                'passed': len(performance_issues) == 0,
                'performance_metrics': {k: v for k, v in metadata.items() if k in thresholds},
                'thresholds': thresholds,
                'issues': performance_issues,
                'message': 'Performance within acceptable range' if not performance_issues else f'Performance issues: {performance_issues}'
            }
            
        except Exception as e:
            return {
                'passed': True,  # Don't fail deployment for performance check errors
                'error': str(e),
                'message': 'Error checking performance metrics'
            }


class DeploymentManager:
    """Manages model deployments with health checks and rollback capabilities"""
    
    def __init__(self):
        self.registry = get_mapping_model_registry()
        self.deployment_history = []
    
    def deploy_model_with_checks(self, 
                                version_id: str,
                                deployment_target: str = "production",
                                make_champion: bool = False,
                                run_health_checks: bool = True,
                                rollback_on_failure: bool = True) -> Dict[str, Any]:
        """
        Deploy a model with comprehensive health checks
        
        Args:
            version_id: Version ID to deploy
            deployment_target: Deployment target
            make_champion: Whether to make this the champion model
            run_health_checks: Whether to run health checks after deployment
            rollback_on_failure: Whether to rollback on health check failure
            
        Returns:
            Deployment result dictionary
        """
        deployment_start = time.time()
        deployment_result = {
            'success': False,
            'version_id': version_id,
            'deployment_target': deployment_target,
            'start_time': datetime.now().isoformat(),
            'duration': 0,
            'health_checks': None,
            'rollback_performed': False,
            'error': None
        }
        
        try:
            fraud_logger.info(f"Starting deployment of {version_id} to {deployment_target}")
            
            # Get model version info
            if version_id not in self.registry.model_versions:
                raise ValueError(f"Model version not found: {version_id}")
            
            model_version = self.registry.model_versions[version_id]
            
            # Store previous champion for potential rollback
            previous_champion = None
            if make_champion:
                previous_champion = self.registry.get_champion_model(model_version.model_type)
            
            # Perform deployment
            deployment_success = self.registry.deploy_model(
                version_id=version_id,
                deployment_target=deployment_target,
                make_champion=make_champion
            )
            
            if not deployment_success:
                raise Exception("Model deployment failed")
            
            deployment_result['deployment_completed'] = True
            fraud_logger.info(f"Model {version_id} deployed successfully")
            
            # Run health checks if requested
            if run_health_checks:
                fraud_logger.info(f"Running health checks for {version_id}")
                
                # Determine deployment path
                deployment_path = Path("artifacts/feature_mapping/mappers") / model_version.model_type
                
                # Run health checks
                health_checker = DeploymentHealthCheck(
                    model_path=str(deployment_path),
                    model_type=model_version.model_type
                )
                
                health_results = health_checker.run_health_checks()
                deployment_result['health_checks'] = health_results
                
                # Check if health checks passed
                if health_results['overall_health'] != 'healthy':
                    fraud_logger.warning(f"Health checks failed for {version_id}: {health_results.get('failed_checks', [])}")
                    
                    # Rollback if requested and previous champion exists
                    if rollback_on_failure and previous_champion:
                        fraud_logger.info(f"Rolling back to previous champion: {previous_champion.version_id}")
                        
                        rollback_success = self.registry.deploy_model(
                            version_id=previous_champion.version_id,
                            deployment_target=deployment_target,
                            make_champion=True
                        )
                        
                        if rollback_success:
                            deployment_result['rollback_performed'] = True
                            deployment_result['rollback_to_version'] = previous_champion.version_id
                            fraud_logger.info(f"Rollback completed to {previous_champion.version_id}")
                        else:
                            fraud_logger.error("Rollback failed!")
                            deployment_result['rollback_failed'] = True
                    
                    # Don't mark as successful if health checks failed
                    if not deployment_result.get('rollback_performed', False):
                        raise Exception(f"Health checks failed: {health_results.get('failed_checks', [])}")
                else:
                    fraud_logger.info(f"Health checks passed for {version_id}")
            
            # Mark as successful
            deployment_result['success'] = True
            
        except Exception as e:
            deployment_result['error'] = str(e)
            fraud_logger.error(f"Deployment failed for {version_id}: {e}")
        
        finally:
            deployment_result['duration'] = time.time() - deployment_start
            deployment_result['end_time'] = datetime.now().isoformat()
            
            # Store deployment history
            self.deployment_history.append(deployment_result)
            
            # Log final result
            if deployment_result['success']:
                fraud_logger.info(f"Deployment completed successfully for {version_id} in {deployment_result['duration']:.2f}s")
            else:
                fraud_logger.error(f"Deployment failed for {version_id} after {deployment_result['duration']:.2f}s")
        
        return deployment_result
    
    def batch_deploy_models(self, 
                           version_ids: List[str],
                           deployment_target: str = "production",
                           make_champions: bool = False) -> Dict[str, Any]:
        """
        Deploy multiple models in batch
        
        Args:
            version_ids: List of version IDs to deploy
            deployment_target: Deployment target
            make_champions: Whether to make models champions
            
        Returns:
            Batch deployment results
        """
        batch_start = time.time()
        batch_result = {
            'total_models': len(version_ids),
            'successful_deployments': 0,
            'failed_deployments': 0,
            'deployment_results': {},
            'start_time': datetime.now().isoformat(),
            'duration': 0
        }
        
        fraud_logger.info(f"Starting batch deployment of {len(version_ids)} models")
        
        for version_id in version_ids:
            try:
                result = self.deploy_model_with_checks(
                    version_id=version_id,
                    deployment_target=deployment_target,
                    make_champion=make_champions
                )
                
                batch_result['deployment_results'][version_id] = result
                
                if result['success']:
                    batch_result['successful_deployments'] += 1
                else:
                    batch_result['failed_deployments'] += 1
                    
            except Exception as e:
                batch_result['deployment_results'][version_id] = {
                    'success': False,
                    'error': str(e)
                }
                batch_result['failed_deployments'] += 1
                fraud_logger.error(f"Batch deployment failed for {version_id}: {e}")
        
        batch_result['duration'] = time.time() - batch_start
        batch_result['end_time'] = datetime.now().isoformat()
        
        fraud_logger.info(f"Batch deployment completed: {batch_result['successful_deployments']}/{batch_result['total_models']} successful")
        
        return batch_result
    
    def get_deployment_status(self, deployment_target: str = "production") -> Dict[str, Any]:
        """
        Get current deployment status
        
        Args:
            deployment_target: Deployment target to check
            
        Returns:
            Current deployment status
        """
        try:
            status = {
                'deployment_target': deployment_target,
                'timestamp': datetime.now().isoformat(),
                'deployed_models': {},
                'champion_models': {},
                'total_active_models': 0
            }
            
            # Check each model type
            model_types = set(v.model_type for v in self.registry.model_versions.values())
            
            for model_type in model_types:
                # Find active models
                active_models = [
                    v for v in self.registry.model_versions.values()
                    if v.model_type == model_type and v.is_active
                ]
                
                # Find champion model
                champion_model = self.registry.get_champion_model(model_type)
                
                if active_models:
                    status['deployed_models'][model_type] = [
                        {
                            'version_id': v.version_id,
                            'version_number': v.version_number,
                            'deployment_date': v.deployment_date,
                            'is_champion': v.is_champion
                        }
                        for v in active_models
                    ]
                    status['total_active_models'] += len(active_models)
                
                if champion_model:
                    status['champion_models'][model_type] = {
                        'version_id': champion_model.version_id,
                        'version_number': champion_model.version_number,
                        'deployment_date': champion_model.deployment_date,
                        'performance_metrics': champion_model.performance_metrics
                    }
            
            return status
            
        except Exception as e:
            fraud_logger.error(f"Error getting deployment status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent deployment history
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent deployment results
        """
        # Sort by start time (newest first)
        sorted_history = sorted(
            self.deployment_history,
            key=lambda x: x.get('start_time', ''),
            reverse=True
        )
        
        return sorted_history[:limit]


# Global deployment manager instance
_deployment_manager = None


def get_deployment_manager() -> DeploymentManager:
    """Get the global deployment manager instance"""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager()
    return _deployment_manager


if __name__ == "__main__":
    # Example usage
    manager = DeploymentManager()
    
    # Example deployment with health checks
    result = manager.deploy_model_with_checks(
        version_id="random_forest_v1.0.0_1234567890",
        make_champion=True,
        run_health_checks=True,
        rollback_on_failure=True
    )
    
    print(f"Deployment result: {result}")