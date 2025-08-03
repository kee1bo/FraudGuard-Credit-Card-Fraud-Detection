"""
Model Version Manager
Manages versioning, deployment, and A/B testing of feature mapping models.
"""

import json
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import joblib

from fraudguard.entity.feature_mapping_entity import MappingModelMetadata
from fraudguard.logger import fraud_logger


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_type: str
    version_number: str
    created_at: str
    performance_metrics: Dict[str, float]
    model_path: str
    is_active: bool = False
    is_production: bool = False
    deployment_date: Optional[str] = None
    rollback_version: Optional[str] = None
    notes: str = ""


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    test_name: str
    model_a: str  # Version ID
    model_b: str  # Version ID
    traffic_split: float  # Percentage for model B (0.0 to 1.0)
    start_date: str
    end_date: Optional[str] = None
    is_active: bool = True
    success_metric: str = "accuracy"
    min_sample_size: int = 100


class ModelVersionManager:
    """Manages model versions and deployments"""
    
    def __init__(self, 
                 models_dir: str = "artifacts/feature_mapping/mappers",
                 versions_dir: str = "artifacts/feature_mapping/versions",
                 registry_file: str = "artifacts/feature_mapping/model_registry.json"):
        
        self.models_dir = Path(models_dir)
        self.versions_dir = Path(versions_dir)
        self.registry_file = Path(registry_file)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize registry
        self.registry = self._load_registry()
        self.ab_tests = self._load_ab_tests()
        
        fraud_logger.info("Model Version Manager initialized")
    
    def _load_registry(self) -> Dict[str, List[ModelVersion]]:
        """Load model registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to ModelVersion objects
                registry = {}
                for model_type, versions in data.get('models', {}).items():
                    registry[model_type] = [
                        ModelVersion(**version) for version in versions
                    ]
                
                return registry
                
            except Exception as e:
                fraud_logger.error(f"Error loading model registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        try:
            # Convert ModelVersion objects to dicts
            registry_data = {}
            for model_type, versions in self.registry.items():
                registry_data[model_type] = [
                    asdict(version) for version in versions
                ]
            
            data = {
                'models': registry_data,
                'ab_tests': [asdict(test) for test in self.ab_tests],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            fraud_logger.error(f"Error saving model registry: {e}")
    
    def _load_ab_tests(self) -> List[ABTestConfig]:
        """Load A/B test configurations"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                return [
                    ABTestConfig(**test) for test in data.get('ab_tests', [])
                ]
                
            except Exception as e:
                fraud_logger.error(f"Error loading A/B tests: {e}")
        
        return []
    
    def register_model_version(self, 
                             model_type: str,
                             model_path: str,
                             performance_metrics: Dict[str, float],
                             version_number: Optional[str] = None,
                             notes: str = "") -> str:
        """
        Register a new model version
        
        Args:
            model_type: Type of model (e.g., 'random_forest', 'xgboost')
            model_path: Path to the trained model
            performance_metrics: Performance metrics from validation
            version_number: Optional version number (auto-generated if None)
            notes: Optional notes about this version
            
        Returns:
            Version ID of the registered model
        """
        # Generate version ID and number
        timestamp = int(time.time())
        version_id = f"{model_type}_v{timestamp}"
        
        if version_number is None:
            # Auto-generate version number
            existing_versions = self.registry.get(model_type, [])
            version_number = f"1.{len(existing_versions)}"
        
        # Create version directory
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy model files to version directory
        source_path = Path(model_path)
        if source_path.is_dir():
            shutil.copytree(source_path, version_dir / "model", dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, version_dir / "model.pkl")
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            version_number=version_number,
            created_at=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            model_path=str(version_dir),
            notes=notes
        )
        
        # Add to registry
        if model_type not in self.registry:
            self.registry[model_type] = []
        
        self.registry[model_type].append(model_version)
        
        # Save registry
        self._save_registry()
        
        fraud_logger.info(f"Registered model version: {version_id}")
        return version_id
    
    def get_model_versions(self, model_type: str) -> List[ModelVersion]:
        """Get all versions for a model type"""
        return self.registry.get(model_type, [])
    
    def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model type"""
        versions = self.registry.get(model_type, [])
        for version in versions:
            if version.is_active:
                return version
        return None
    
    def get_production_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the production version for a model type"""
        versions = self.registry.get(model_type, [])
        for version in versions:
            if version.is_production:
                return version
        return None
    
    def deploy_version(self, 
                      version_id: str,
                      to_production: bool = False,
                      rollback_version: Optional[str] = None) -> bool:
        """
        Deploy a model version
        
        Args:
            version_id: Version ID to deploy
            to_production: Whether to deploy to production
            rollback_version: Version to rollback to if deployment fails
            
        Returns:
            True if deployment successful
        """
        try:
            # Find the version
            target_version = None
            model_type = None
            
            for mtype, versions in self.registry.items():
                for version in versions:
                    if version.version_id == version_id:
                        target_version = version
                        model_type = mtype
                        break
                if target_version:
                    break
            
            if not target_version:
                fraud_logger.error(f"Version {version_id} not found")
                return False
            
            # Deactivate current active/production versions
            for version in self.registry[model_type]:
                if to_production:
                    if version.is_production:
                        version.is_production = False
                        version.rollback_version = version.version_id
                else:
                    if version.is_active:
                        version.is_active = False
            
            # Activate target version
            if to_production:
                target_version.is_production = True
                target_version.deployment_date = datetime.now().isoformat()
            else:
                target_version.is_active = True
            
            if rollback_version:
                target_version.rollback_version = rollback_version
            
            # Copy model to active location
            active_model_path = self.models_dir / model_type
            version_model_path = Path(target_version.model_path) / "model"
            
            if active_model_path.exists():
                shutil.rmtree(active_model_path)
            
            if version_model_path.is_dir():
                shutil.copytree(version_model_path, active_model_path)
            else:
                active_model_path.mkdir(exist_ok=True)
                shutil.copy2(Path(target_version.model_path) / "model.pkl", 
                           active_model_path / "model.pkl")
            
            # Save registry
            self._save_registry()
            
            deployment_type = "production" if to_production else "active"
            fraud_logger.info(f"Deployed {version_id} to {deployment_type}")
            
            return True
            
        except Exception as e:
            fraud_logger.error(f"Deployment failed for {version_id}: {e}")
            return False
    
    def rollback_version(self, model_type: str) -> bool:
        """
        Rollback to the previous version
        
        Args:
            model_type: Type of model to rollback
            
        Returns:
            True if rollback successful
        """
        try:
            # Find current production version
            current_version = self.get_production_version(model_type)
            if not current_version or not current_version.rollback_version:
                fraud_logger.error(f"No rollback version available for {model_type}")
                return False
            
            # Deploy the rollback version
            success = self.deploy_version(
                current_version.rollback_version,
                to_production=True
            )
            
            if success:
                fraud_logger.info(f"Rolled back {model_type} to {current_version.rollback_version}")
            
            return success
            
        except Exception as e:
            fraud_logger.error(f"Rollback failed for {model_type}: {e}")
            return False
    
    def create_ab_test(self, 
                      test_name: str,
                      model_type: str,
                      version_a: str,
                      version_b: str,
                      traffic_split: float = 0.5,
                      duration_days: int = 7,
                      success_metric: str = "accuracy",
                      min_sample_size: int = 100) -> str:
        """
        Create an A/B test between two model versions
        
        Args:
            test_name: Name of the test
            model_type: Type of model being tested
            version_a: Control version ID
            version_b: Test version ID
            traffic_split: Percentage of traffic for version B (0.0 to 1.0)
            duration_days: Duration of test in days
            success_metric: Metric to optimize for
            min_sample_size: Minimum sample size before making decisions
            
        Returns:
            Test ID
        """
        test_id = f"ab_test_{int(time.time())}"
        start_date = datetime.now()
        end_date = start_date.replace(day=start_date.day + duration_days)
        
        ab_test = ABTestConfig(
            test_id=test_id,
            test_name=test_name,
            model_a=version_a,
            model_b=version_b,
            traffic_split=traffic_split,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            success_metric=success_metric,
            min_sample_size=min_sample_size
        )
        
        self.ab_tests.append(ab_test)
        self._save_registry()
        
        fraud_logger.info(f"Created A/B test: {test_id} ({test_name})")
        return test_id
    
    def get_ab_test_assignment(self, test_id: str, user_id: str) -> str:
        """
        Get A/B test assignment for a user
        
        Args:
            test_id: Test ID
            user_id: User identifier for consistent assignment
            
        Returns:
            Version ID to use for this user
        """
        # Find the test
        test = None
        for ab_test in self.ab_tests:
            if ab_test.test_id == test_id and ab_test.is_active:
                test = ab_test
                break
        
        if not test:
            return test.model_a if test else ""
        
        # Consistent assignment based on user ID hash
        import hashlib
        hash_value = int(hashlib.md5(f"{test_id}_{user_id}".encode()).hexdigest(), 16)
        assignment_value = (hash_value % 100) / 100.0
        
        if assignment_value < test.traffic_split:
            return test.model_b
        else:
            return test.model_a
    
    def end_ab_test(self, test_id: str, winning_version: Optional[str] = None) -> bool:
        """
        End an A/B test and optionally deploy the winning version
        
        Args:
            test_id: Test ID to end
            winning_version: Version ID of the winner (optional)
            
        Returns:
            True if successful
        """
        try:
            # Find and deactivate the test
            test = None
            for ab_test in self.ab_tests:
                if ab_test.test_id == test_id:
                    test = ab_test
                    break
            
            if not test:
                fraud_logger.error(f"A/B test {test_id} not found")
                return False
            
            test.is_active = False
            test.end_date = datetime.now().isoformat()
            
            # Deploy winning version if specified
            if winning_version:
                # Find model type from version
                model_type = None
                for mtype, versions in self.registry.items():
                    for version in versions:
                        if version.version_id == winning_version:
                            model_type = mtype
                            break
                    if model_type:
                        break
                
                if model_type:
                    self.deploy_version(winning_version, to_production=True)
                    fraud_logger.info(f"Deployed winning version {winning_version} from A/B test {test_id}")
            
            self._save_registry()
            fraud_logger.info(f"Ended A/B test: {test_id}")
            
            return True
            
        except Exception as e:
            fraud_logger.error(f"Error ending A/B test {test_id}: {e}")
            return False
    
    def get_model_performance_comparison(self, model_type: str) -> Dict[str, Any]:
        """Get performance comparison across all versions of a model type"""
        versions = self.registry.get(model_type, [])
        
        if not versions:
            return {}
        
        comparison = {
            'model_type': model_type,
            'total_versions': len(versions),
            'versions': []
        }
        
        for version in sorted(versions, key=lambda v: v.created_at, reverse=True):
            version_info = {
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_at': version.created_at,
                'is_active': version.is_active,
                'is_production': version.is_production,
                'performance_metrics': version.performance_metrics,
                'notes': version.notes
            }
            comparison['versions'].append(version_info)
        
        # Find best performing version
        if versions and versions[0].performance_metrics:
            metric_name = list(versions[0].performance_metrics.keys())[0]
            best_version = max(versions, key=lambda v: v.performance_metrics.get(metric_name, 0))
            comparison['best_version'] = {
                'version_id': best_version.version_id,
                'metric': metric_name,
                'value': best_version.performance_metrics.get(metric_name, 0)
            }
        
        return comparison
    
    def cleanup_old_versions(self, model_type: str, keep_count: int = 5) -> int:
        """
        Clean up old model versions, keeping only the most recent ones
        
        Args:
            model_type: Type of model to clean up
            keep_count: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        versions = self.registry.get(model_type, [])
        
        if len(versions) <= keep_count:
            return 0
        
        # Sort by creation date (newest first)
        sorted_versions = sorted(versions, key=lambda v: v.created_at, reverse=True)
        
        # Keep active and production versions
        protected_versions = [v for v in sorted_versions if v.is_active or v.is_production]
        other_versions = [v for v in sorted_versions if not (v.is_active or v.is_production)]
        
        # Determine versions to delete
        to_keep = protected_versions + other_versions[:keep_count - len(protected_versions)]
        to_delete = [v for v in versions if v not in to_keep]
        
        deleted_count = 0
        for version in to_delete:
            try:
                # Delete version directory
                version_path = Path(version.model_path)
                if version_path.exists():
                    shutil.rmtree(version_path)
                
                # Remove from registry
                self.registry[model_type].remove(version)
                deleted_count += 1
                
                fraud_logger.info(f"Deleted old version: {version.version_id}")
                
            except Exception as e:
                fraud_logger.error(f"Error deleting version {version.version_id}: {e}")
        
        if deleted_count > 0:
            self._save_registry()
        
        return deleted_count
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get overall deployment status"""
        status = {
            'model_types': list(self.registry.keys()),
            'total_versions': sum(len(versions) for versions in self.registry.values()),
            'active_versions': {},
            'production_versions': {},
            'active_ab_tests': len([t for t in self.ab_tests if t.is_active])
        }
        
        for model_type in self.registry.keys():
            active_version = self.get_active_version(model_type)
            production_version = self.get_production_version(model_type)
            
            status['active_versions'][model_type] = active_version.version_id if active_version else None
            status['production_versions'][model_type] = production_version.version_id if production_version else None
        
        return status