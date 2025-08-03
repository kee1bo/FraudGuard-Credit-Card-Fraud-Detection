"""
Mapping Model Registry
Manages versioning, deployment, and A/B testing of feature mapping models.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import joblib
import numpy as np

from fraudguard.entity.feature_mapping_entity import MappingModelMetadata
from fraudguard.logger import fraud_logger


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_type: str
    version_number: str
    created_at: str
    created_by: str
    description: str
    performance_metrics: Dict[str, float]
    model_path: str
    is_active: bool = False
    is_champion: bool = False
    deployment_date: Optional[str] = None
    rollback_version: Optional[str] = None


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    test_name: str
    champion_version: str
    challenger_version: str
    traffic_split: float  # Percentage of traffic to challenger (0-100)
    start_date: str
    end_date: Optional[str]
    success_metrics: List[str]
    is_active: bool = True
    results: Optional[Dict[str, Any]] = None


class MappingModelRegistry:
    """Registry for managing feature mapping model versions"""
    
    def __init__(self, registry_path: str = "artifacts/mapping_model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry files
        self.versions_file = self.registry_path / "model_versions.json"
        self.ab_tests_file = self.registry_path / "ab_tests.json"
        self.deployment_log_file = self.registry_path / "deployment_log.json"
        
        # In-memory storage
        self.model_versions: Dict[str, ModelVersion] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.deployment_log: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load_registry_data()
        
        fraud_logger.info(f"Mapping model registry initialized at {registry_path}")
    
    def _load_registry_data(self):
        """Load existing registry data from files"""
        try:
            # Load model versions
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    versions_data = json.load(f)
                    self.model_versions = {
                        k: ModelVersion(**v) for k, v in versions_data.items()
                    }
                fraud_logger.info(f"Loaded {len(self.model_versions)} model versions")
            
            # Load A/B tests
            if self.ab_tests_file.exists():
                with open(self.ab_tests_file, 'r') as f:
                    ab_tests_data = json.load(f)
                    self.ab_tests = {
                        k: ABTestConfig(**v) for k, v in ab_tests_data.items()
                    }
                fraud_logger.info(f"Loaded {len(self.ab_tests)} A/B tests")
            
            # Load deployment log
            if self.deployment_log_file.exists():
                with open(self.deployment_log_file, 'r') as f:
                    self.deployment_log = json.load(f)
                fraud_logger.info(f"Loaded {len(self.deployment_log)} deployment log entries")
                
        except Exception as e:
            fraud_logger.error(f"Error loading registry data: {e}")
    
    def _save_registry_data(self):
        """Save registry data to files"""
        try:
            # Save model versions
            with open(self.versions_file, 'w') as f:
                versions_data = {k: asdict(v) for k, v in self.model_versions.items()}
                json.dump(versions_data, f, indent=2)
            
            # Save A/B tests
            with open(self.ab_tests_file, 'w') as f:
                ab_tests_data = {k: asdict(v) for k, v in self.ab_tests.items()}
                json.dump(ab_tests_data, f, indent=2)
            
            # Save deployment log
            with open(self.deployment_log_file, 'w') as f:
                json.dump(self.deployment_log, f, indent=2)
                
        except Exception as e:
            fraud_logger.error(f"Error saving registry data: {e}")
    
    def register_model(self, 
                      model_path: str,
                      model_type: str,
                      version_number: str,
                      description: str,
                      performance_metrics: Dict[str, float],
                      created_by: str = "system") -> str:
        """
        Register a new model version
        
        Args:
            model_path: Path to the trained model
            model_type: Type of model (random_forest, xgboost, etc.)
            version_number: Version number (e.g., "1.0.0")
            description: Description of the model
            performance_metrics: Performance metrics
            created_by: Who created the model
            
        Returns:
            Version ID of the registered model
        """
        try:
            # Generate version ID
            version_id = f"{model_type}_v{version_number}_{int(time.time())}"
            
            # Copy model to registry
            registry_model_path = self.registry_path / "models" / version_id
            registry_model_path.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            source_path = Path(model_path)
            if source_path.is_dir():
                shutil.copytree(source_path, registry_model_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, registry_model_path)
            
            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_type=model_type,
                version_number=version_number,
                created_at=datetime.now().isoformat(),
                created_by=created_by,
                description=description,
                performance_metrics=performance_metrics,
                model_path=str(registry_model_path)
            )
            
            # Store in registry
            self.model_versions[version_id] = model_version
            
            # Save to disk
            self._save_registry_data()
            
            # Log registration
            self._log_deployment_event({
                'event_type': 'model_registered',
                'version_id': version_id,
                'model_type': model_type,
                'version_number': version_number,
                'created_by': created_by,
                'timestamp': datetime.now().isoformat()
            })
            
            fraud_logger.info(f"Model registered: {version_id}")
            return version_id
            
        except Exception as e:
            fraud_logger.error(f"Error registering model: {e}")
            raise
    
    def deploy_model(self, 
                    version_id: str,
                    deployment_target: str = "production",
                    make_champion: bool = False) -> bool:
        """
        Deploy a model version
        
        Args:
            version_id: Version ID to deploy
            deployment_target: Deployment target (production, staging, etc.)
            make_champion: Whether to make this the champion model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if version_id not in self.model_versions:
                raise ValueError(f"Model version not found: {version_id}")
            
            model_version = self.model_versions[version_id]
            
            # Copy model to deployment location
            deployment_path = self.registry_path.parent / "feature_mapping" / "mappers" / model_version.model_type
            deployment_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove existing deployment
            if deployment_path.exists():
                shutil.rmtree(deployment_path)
            
            # Copy new model
            shutil.copytree(model_version.model_path, deployment_path)
            
            # Update model version status
            model_version.is_active = True
            model_version.deployment_date = datetime.now().isoformat()
            
            # Handle champion status
            if make_champion:
                # Remove champion status from other models of same type
                for other_version in self.model_versions.values():
                    if (other_version.model_type == model_version.model_type and 
                        other_version.version_id != version_id):
                        other_version.is_champion = False
                
                model_version.is_champion = True
            
            # Save changes
            self._save_registry_data()
            
            # Log deployment
            self._log_deployment_event({
                'event_type': 'model_deployed',
                'version_id': version_id,
                'model_type': model_version.model_type,
                'deployment_target': deployment_target,
                'make_champion': make_champion,
                'timestamp': datetime.now().isoformat()
            })
            
            fraud_logger.info(f"Model deployed: {version_id} to {deployment_target}")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Error deploying model {version_id}: {e}")
            return False
    
    def get_model_versions(self, model_type: Optional[str] = None) -> List[ModelVersion]:
        """
        Get all model versions, optionally filtered by type
        
        Args:
            model_type: Optional model type filter
            
        Returns:
            List of model versions
        """
        versions = list(self.model_versions.values())
        
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        return versions
    
    def get_champion_model(self, model_type: str) -> Optional[ModelVersion]:
        """
        Get the champion model for a given type
        
        Args:
            model_type: Model type to get champion for
            
        Returns:
            Champion model version or None
        """
        for version in self.model_versions.values():
            if version.model_type == model_type and version.is_champion:
                return version
        return None
    
    def _log_deployment_event(self, event: Dict[str, Any]):
        """Log a deployment event"""
        self.deployment_log.append(event)
        
        # Keep only last 1000 entries
        if len(self.deployment_log) > 1000:
            self.deployment_log = self.deployment_log[-1000:]


# Global registry instance
_mapping_model_registry = None


def get_mapping_model_registry() -> MappingModelRegistry:
    """Get the global mapping model registry instance"""
    global _mapping_model_registry
    if _mapping_model_registry is None:
        _mapping_model_registry = MappingModelRegistry()
    return _mapping_model_registry


if __name__ == "__main__":
    # Example usage
    registry = MappingModelRegistry()
    
    # Example: Register a model
    version_id = registry.register_model(
        model_path="artifacts/feature_mapping/mappers/random_forest",
        model_type="random_forest",
        version_number="1.0.0",
        description="Initial Random Forest mapper",
        performance_metrics={"mse": 0.15, "avg_correlation": 0.85},
        created_by="training_pipeline"
    )
    
    print(f"Registered model: {version_id}")
    
    # Deploy the model
    registry.deploy_model(version_id, make_champion=True)
    
    print("Model deployed as champion")