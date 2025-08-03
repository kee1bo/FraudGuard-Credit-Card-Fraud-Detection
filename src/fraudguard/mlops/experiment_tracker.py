"""
Professional Experiment Tracking System
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import shutil
import hashlib

from .model_metadata import ModelMetadata, ModelVersion
from ..logger import fraud_logger
from ..exception import FraudGuardException


class Experiment:
    """Represents a single ML experiment"""
    
    def __init__(self, experiment_id: str = None, name: str = "", description: str = ""):
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.status = "running"  # running, completed, failed, cancelled
        self.parameters = {}
        self.metrics = {}
        self.artifacts = {}
        self.tags = []
        self.parent_experiment = None
        self.child_experiments = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary"""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'tags': self.tags,
            'parent_experiment': self.parent_experiment,
            'child_experiments': self.child_experiments
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary"""
        exp = cls(
            experiment_id=data['experiment_id'],
            name=data['name'],
            description=data['description']
        )
        exp.created_at = datetime.fromisoformat(data['created_at'])
        exp.status = data['status']
        exp.parameters = data['parameters']
        exp.metrics = data['metrics']
        exp.artifacts = data['artifacts']
        exp.tags = data['tags']
        exp.parent_experiment = data.get('parent_experiment')
        exp.child_experiments = data.get('child_experiments', [])
        return exp


class ExperimentTracker:
    """Professional experiment tracking and versioning system"""
    
    def __init__(self, tracking_path: str = "artifacts/experiments"):
        self.tracking_path = Path(tracking_path)
        self.tracking_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.tracking_path / "experiments.db"
        self.artifacts_path = self.tracking_path / "artifacts"
        self.artifacts_path.mkdir(exist_ok=True)
        
        self._init_database()
        fraud_logger.info(f"Experiment tracker initialized at {self.tracking_path}")
    
    def _init_database(self):
        """Initialize SQLite database for experiment tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT,
                    status TEXT,
                    parameters TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    tags TEXT,
                    parent_experiment TEXT,
                    child_experiments TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    run_name TEXT,
                    created_at TEXT,
                    status TEXT,
                    parameters TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    model_artifacts TEXT,
                    duration_seconds REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    name TEXT,
                    experiment_ids TEXT,
                    created_at TEXT,
                    comparison_results TEXT
                )
            """)
            
            conn.commit()
    
    def create_experiment(self, name: str, description: str = "", 
                         parameters: Dict[str, Any] = None, 
                         tags: List[str] = None) -> Experiment:
        """Create a new experiment"""
        try:
            experiment = Experiment(name=name, description=description)
            experiment.parameters = parameters or {}
            experiment.tags = tags or []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experiments 
                    (experiment_id, name, description, created_at, status, parameters, 
                     metrics, artifacts, tags, parent_experiment, child_experiments)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment.experiment_id,
                    experiment.name,
                    experiment.description,
                    experiment.created_at.isoformat(),
                    experiment.status,
                    json.dumps(experiment.parameters),
                    json.dumps(experiment.metrics),
                    json.dumps(experiment.artifacts),
                    json.dumps(experiment.tags),
                    experiment.parent_experiment,
                    json.dumps(experiment.child_experiments)
                ))
                conn.commit()
            
            # Create experiment directory
            exp_dir = self.artifacts_path / experiment.experiment_id
            exp_dir.mkdir(exist_ok=True)
            
            fraud_logger.info(f"Experiment created: {experiment.name} ({experiment.experiment_id})")
            return experiment
            
        except Exception as e:
            fraud_logger.error(f"Failed to create experiment: {e}")
            raise FraudGuardException(f"Experiment creation failed: {e}")
    
    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]):
        """Log parameters for an experiment"""
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                raise FraudGuardException(f"Experiment {experiment_id} not found")
            
            experiment.parameters.update(parameters)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE experiments SET parameters = ? WHERE experiment_id = ?
                """, (json.dumps(experiment.parameters), experiment_id))
                conn.commit()
            
            fraud_logger.debug(f"Parameters logged for experiment {experiment_id}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to log parameters: {e}")
            raise FraudGuardException(f"Parameter logging failed: {e}")
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: int = None):
        """Log metrics for an experiment"""
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                raise FraudGuardException(f"Experiment {experiment_id} not found")
            
            # Handle step-based metrics
            if step is not None:
                for metric_name, value in metrics.items():
                    if metric_name not in experiment.metrics:
                        experiment.metrics[metric_name] = []
                    experiment.metrics[metric_name].append({'step': step, 'value': value})
            else:
                experiment.metrics.update(metrics)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE experiments SET metrics = ? WHERE experiment_id = ?
                """, (json.dumps(experiment.metrics), experiment_id))
                conn.commit()
            
            fraud_logger.debug(f"Metrics logged for experiment {experiment_id}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to log metrics: {e}")
            raise FraudGuardException(f"Metrics logging failed: {e}")
    
    def log_artifact(self, experiment_id: str, artifact_name: str, 
                    artifact_path: str, artifact_type: str = "file"):
        """Log an artifact for an experiment"""
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                raise FraudGuardException(f"Experiment {experiment_id} not found")
            
            # Copy artifact to experiment directory
            exp_artifacts_dir = self.artifacts_path / experiment_id / "artifacts"
            exp_artifacts_dir.mkdir(exist_ok=True)
            
            source_path = Path(artifact_path)
            if source_path.exists():
                if source_path.is_file():
                    dest_path = exp_artifacts_dir / source_path.name
                    shutil.copy2(source_path, dest_path)
                else:
                    dest_path = exp_artifacts_dir / source_path.name
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                
                # Calculate file hash for integrity
                file_hash = self._calculate_file_hash(dest_path)
                
                experiment.artifacts[artifact_name] = {
                    'path': str(dest_path),
                    'type': artifact_type,
                    'size_bytes': self._get_path_size(dest_path),
                    'hash': file_hash,
                    'logged_at': datetime.now().isoformat()
                }
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE experiments SET artifacts = ? WHERE experiment_id = ?
                    """, (json.dumps(experiment.artifacts), experiment_id))
                    conn.commit()
                
                fraud_logger.debug(f"Artifact {artifact_name} logged for experiment {experiment_id}")
            else:
                raise FraudGuardException(f"Artifact path does not exist: {artifact_path}")
                
        except Exception as e:
            fraud_logger.error(f"Failed to log artifact: {e}")
            raise FraudGuardException(f"Artifact logging failed: {e}")
    
    def finish_experiment(self, experiment_id: str, status: str = "completed"):
        """Mark experiment as finished"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE experiments SET status = ? WHERE experiment_id = ?
                """, (status, experiment_id))
                conn.commit()
            
            fraud_logger.info(f"Experiment {experiment_id} finished with status: {status}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to finish experiment: {e}")
            raise FraudGuardException(f"Experiment finish failed: {e}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT experiment_id, name, description, created_at, status, 
                           parameters, metrics, artifacts, tags, parent_experiment, child_experiments
                    FROM experiments WHERE experiment_id = ?
                """, (experiment_id,))
                
                row = cursor.fetchone()
                if row:
                    experiment = Experiment(
                        experiment_id=row[0],
                        name=row[1],
                        description=row[2]
                    )
                    experiment.created_at = datetime.fromisoformat(row[3])
                    experiment.status = row[4]
                    experiment.parameters = json.loads(row[5]) if row[5] else {}
                    experiment.metrics = json.loads(row[6]) if row[6] else {}
                    experiment.artifacts = json.loads(row[7]) if row[7] else {}
                    experiment.tags = json.loads(row[8]) if row[8] else []
                    experiment.parent_experiment = row[9]
                    experiment.child_experiments = json.loads(row[10]) if row[10] else []
                    
                    return experiment
                
                return None
                
        except Exception as e:
            fraud_logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    def list_experiments(self, limit: int = 50, status: str = None) -> List[Experiment]:
        """List experiments with optional filtering"""
        try:
            experiments = []
            
            query = """
                SELECT experiment_id, name, description, created_at, status, 
                       parameters, metrics, artifacts, tags, parent_experiment, child_experiments
                FROM experiments
            """
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    experiment = Experiment(
                        experiment_id=row[0],
                        name=row[1],
                        description=row[2]
                    )
                    experiment.created_at = datetime.fromisoformat(row[3])
                    experiment.status = row[4]
                    experiment.parameters = json.loads(row[5]) if row[5] else {}
                    experiment.metrics = json.loads(row[6]) if row[6] else {}
                    experiment.artifacts = json.loads(row[7]) if row[7] else {}
                    experiment.tags = json.loads(row[8]) if row[8] else []
                    experiment.parent_experiment = row[9]
                    experiment.child_experiments = json.loads(row[10]) if row[10] else []
                    
                    experiments.append(experiment)
            
            return experiments
            
        except Exception as e:
            fraud_logger.error(f"Failed to list experiments: {e}")
            return []
    
    def compare_experiments(self, experiment_ids: List[str], 
                          comparison_name: str = "") -> Dict[str, Any]:
        """Compare multiple experiments"""
        try:
            experiments = []
            for exp_id in experiment_ids:
                exp = self.get_experiment(exp_id)
                if exp:
                    experiments.append(exp)
            
            if len(experiments) < 2:
                raise FraudGuardException("At least 2 experiments required for comparison")
            
            comparison_results = {
                'comparison_id': str(uuid.uuid4()),
                'name': comparison_name or f"Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'created_at': datetime.now().isoformat(),
                'experiments': [],
                'parameter_comparison': {},
                'metric_comparison': {},
                'best_performing': {}
            }
            
            # Collect experiment data
            all_parameters = set()
            all_metrics = set()
            
            for exp in experiments:
                exp_data = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'status': exp.status,
                    'parameters': exp.parameters,
                    'metrics': exp.metrics
                }
                comparison_results['experiments'].append(exp_data)
                
                all_parameters.update(exp.parameters.keys())
                all_metrics.update(exp.metrics.keys())
            
            # Compare parameters
            for param in all_parameters:
                comparison_results['parameter_comparison'][param] = {}
                for exp in experiments:
                    comparison_results['parameter_comparison'][param][exp.experiment_id] = \
                        exp.parameters.get(param, None)
            
            # Compare metrics
            for metric in all_metrics:
                comparison_results['metric_comparison'][metric] = {}
                metric_values = []
                
                for exp in experiments:
                    value = exp.metrics.get(metric, None)
                    comparison_results['metric_comparison'][metric][exp.experiment_id] = value
                    if isinstance(value, (int, float)):
                        metric_values.append((exp.experiment_id, value))
                
                # Find best performing for this metric
                if metric_values:
                    if metric.lower() in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                        # Higher is better
                        best_exp_id, best_value = max(metric_values, key=lambda x: x[1])
                    elif metric.lower() in ['loss', 'error_rate', 'drift_score']:
                        # Lower is better
                        best_exp_id, best_value = min(metric_values, key=lambda x: x[1])
                    else:
                        # Default to higher is better
                        best_exp_id, best_value = max(metric_values, key=lambda x: x[1])
                    
                    comparison_results['best_performing'][metric] = {
                        'experiment_id': best_exp_id,
                        'value': best_value
                    }
            
            # Save comparison
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experiment_comparisons 
                    (comparison_id, name, experiment_ids, created_at, comparison_results)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    comparison_results['comparison_id'],
                    comparison_results['name'],
                    json.dumps(experiment_ids),
                    comparison_results['created_at'],
                    json.dumps(comparison_results)
                ))
                conn.commit()
            
            fraud_logger.info(f"Experiment comparison created: {comparison_results['comparison_id']}")
            return comparison_results
            
        except Exception as e:
            fraud_logger.error(f"Failed to compare experiments: {e}")
            raise FraudGuardException(f"Experiment comparison failed: {e}")
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its artifacts"""
        try:
            # Remove artifacts
            exp_dir = self.artifacts_path / experiment_id
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                conn.execute("DELETE FROM experiment_runs WHERE experiment_id = ?", (experiment_id,))
                conn.commit()
            
            fraud_logger.info(f"Experiment {experiment_id} deleted")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file or directory"""
        try:
            if file_path.is_file():
                hash_sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(chunk)
                return hash_sha256.hexdigest()
            else:
                # For directories, hash the structure and file contents
                hash_sha256 = hashlib.sha256()
                for file_path in sorted(file_path.rglob("*")):
                    if file_path.is_file():
                        hash_sha256.update(str(file_path.relative_to(file_path)).encode())
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_sha256.update(chunk)
                return hash_sha256.hexdigest()
        except Exception as e:
            fraud_logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return "unknown"
    
    def _get_path_size(self, path: Path) -> int:
        """Get total size of a file or directory"""
        try:
            if path.is_file():
                return path.stat().st_size
            else:
                return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        except Exception as e:
            fraud_logger.error(f"Failed to get size for {path}: {e}")
            return 0