"""
Professional Model Registry for Version Control and Management
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

from .model_metadata import ModelMetadata, ModelVersion, PerformanceMetrics
from ..logger import fraud_logger
from ..exception import FraudGuardException


class ModelRegistry:
    """Professional model registry with SQLite backend"""
    
    def __init__(self, registry_path: str = "artifacts/model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.registry_path / "registry.db"
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self._init_database()
        fraud_logger.info(f"Model registry initialized at {self.registry_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    algorithm TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    description TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    version TEXT,
                    created_at TEXT,
                    is_active BOOLEAN,
                    deployment_stage TEXT,
                    performance_metrics TEXT,
                    training_config TEXT,
                    artifacts_path TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    version_id TEXT,
                    timestamp TEXT,
                    metrics TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id),
                    FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
                )
            """)
            
            conn.commit()
    
    def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model with comprehensive metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO models 
                    (model_id, name, algorithm, created_at, created_by, description, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.name,
                    metadata.algorithm,
                    metadata.created_at.isoformat(),
                    metadata.created_by,
                    metadata.description,
                    json.dumps(metadata.tags),
                    metadata.to_json()
                ))
                conn.commit()
            
            fraud_logger.info(f"Model registered: {metadata.name} ({metadata.model_id})")
            return metadata.model_id
            
        except Exception as e:
            fraud_logger.error(f"Failed to register model: {e}")
            raise FraudGuardException(f"Model registration failed: {e}")
    
    def register_version(self, model_version: ModelVersion, artifacts_path: str) -> str:
        """Register a new model version with artifacts"""
        try:
            # Copy artifacts to registry
            version_path = self.models_path / model_version.model_id / model_version.version
            version_path.mkdir(parents=True, exist_ok=True)
            
            if Path(artifacts_path).exists():
                shutil.copytree(artifacts_path, version_path, dirs_exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_versions
                    (version_id, model_id, version, created_at, is_active, deployment_stage,
                     performance_metrics, training_config, artifacts_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_version.version_id,
                    model_version.model_id,
                    model_version.version,
                    model_version.created_at.isoformat(),
                    model_version.is_active,
                    model_version.deployment_stage,
                    json.dumps(model_version.performance_metrics),
                    json.dumps(model_version.training_config),
                    str(version_path)
                ))
                conn.commit()
            
            fraud_logger.info(f"Model version registered: {model_version.version} for {model_version.model_id}")
            return model_version.version_id
            
        except Exception as e:
            fraud_logger.error(f"Failed to register model version: {e}")
            raise FraudGuardException(f"Model version registration failed: {e}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT metadata FROM models WHERE model_id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                if row:
                    return ModelMetadata.from_json(row[0])
                return None
                
        except Exception as e:
            fraud_logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get model version by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version_id, model_id, version, created_at, is_active, 
                           deployment_stage, performance_metrics, training_config, artifacts_path
                    FROM model_versions WHERE version_id = ?
                """, (version_id,))
                
                row = cursor.fetchone()
                if row:
                    return ModelVersion(
                        version_id=row[0],
                        model_id=row[1],
                        version=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        is_active=bool(row[4]),
                        deployment_stage=row[5],
                        performance_metrics=json.loads(row[6]) if row[6] else {},
                        training_config=json.loads(row[7]) if row[7] else {},
                        model_artifacts={'path': row[8]}
                    )
                return None
                
        except Exception as e:
            fraud_logger.error(f"Failed to get model version {version_id}: {e}")
            return None
    
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models"""
        try:
            models = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT metadata FROM models ORDER BY created_at DESC")
                
                for row in cursor.fetchall():
                    models.append(ModelMetadata.from_json(row[0]))
            
            return models
            
        except Exception as e:
            fraud_logger.error(f"Failed to list models: {e}")
            return []
    
    def list_model_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions for a model"""
        try:
            versions = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version_id, model_id, version, created_at, is_active,
                           deployment_stage, performance_metrics, training_config, artifacts_path
                    FROM model_versions 
                    WHERE model_id = ? 
                    ORDER BY created_at DESC
                """, (model_id,))
                
                for row in cursor.fetchall():
                    versions.append(ModelVersion(
                        version_id=row[0],
                        model_id=row[1],
                        version=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        is_active=bool(row[4]),
                        deployment_stage=row[5],
                        performance_metrics=json.loads(row[6]) if row[6] else {},
                        training_config=json.loads(row[7]) if row[7] else {},
                        model_artifacts={'path': row[8]}
                    ))
            
            return versions
            
        except Exception as e:
            fraud_logger.error(f"Failed to list model versions for {model_id}: {e}")
            return []
    
    def set_active_version(self, model_id: str, version_id: str) -> bool:
        """Set a specific version as active"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Deactivate all versions for this model
                conn.execute("""
                    UPDATE model_versions SET is_active = 0 WHERE model_id = ?
                """, (model_id,))
                
                # Activate the specified version
                conn.execute("""
                    UPDATE model_versions SET is_active = 1 WHERE version_id = ?
                """, (version_id,))
                
                conn.commit()
            
            fraud_logger.info(f"Set active version {version_id} for model {model_id}")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Failed to set active version: {e}")
            return False
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the active version for a model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version_id, model_id, version, created_at, is_active,
                           deployment_stage, performance_metrics, training_config, artifacts_path
                    FROM model_versions 
                    WHERE model_id = ? AND is_active = 1
                """, (model_id,))
                
                row = cursor.fetchone()
                if row:
                    return ModelVersion(
                        version_id=row[0],
                        model_id=row[1],
                        version=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        is_active=bool(row[4]),
                        deployment_stage=row[5],
                        performance_metrics=json.loads(row[6]) if row[6] else {},
                        training_config=json.loads(row[7]) if row[7] else {},
                        model_artifacts={'path': row[8]}
                    )
                return None
                
        except Exception as e:
            fraud_logger.error(f"Failed to get active version for {model_id}: {e}")
            return None
    
    def record_performance(self, model_id: str, version_id: str, metrics: PerformanceMetrics):
        """Record performance metrics for a model version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_history 
                    (model_id, version_id, timestamp, metrics)
                    VALUES (?, ?, ?, ?)
                """, (
                    model_id,
                    version_id,
                    metrics.timestamp.isoformat(),
                    json.dumps(metrics.to_dict())
                ))
                conn.commit()
            
            fraud_logger.debug(f"Performance recorded for {model_id}/{version_id}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to record performance: {e}")
    
    def get_performance_history(self, model_id: str, limit: int = 100) -> List[PerformanceMetrics]:
        """Get performance history for a model"""
        try:
            history = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT metrics FROM performance_history 
                    WHERE model_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (model_id, limit))
                
                for row in cursor.fetchall():
                    metrics_data = json.loads(row[0])
                    history.append(PerformanceMetrics.from_dict(metrics_data))
            
            return history
            
        except Exception as e:
            fraud_logger.error(f"Failed to get performance history for {model_id}: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions"""
        try:
            # Remove artifacts
            model_path = self.models_path / model_id
            if model_path.exists():
                shutil.rmtree(model_path)
            
            with sqlite3.connect(self.db_path) as conn:
                # Delete performance history
                conn.execute("DELETE FROM performance_history WHERE model_id = ?", (model_id,))
                
                # Delete versions
                conn.execute("DELETE FROM model_versions WHERE model_id = ?", (model_id,))
                
                # Delete model
                conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                
                conn.commit()
            
            fraud_logger.info(f"Model {model_id} deleted")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Failed to delete model {model_id}: {e}")
            return False