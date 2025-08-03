"""
Professional Model Metadata Management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import json


@dataclass
class ModelMetadata:
    """Comprehensive model metadata for professional tracking"""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    algorithm: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    training_duration: Optional[timedelta] = None
    dataset_version: str = "unknown"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    deployment_status: str = "development"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    model_size_mb: float = 0.0
    training_samples: int = 0
    feature_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, timedelta):
                data[key] = value.total_seconds()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Handle timedelta conversion
        if 'training_duration' in data and isinstance(data['training_duration'], (int, float)):
            data['training_duration'] = timedelta(seconds=data['training_duration'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class PerformanceMetrics:
    """Real-time model performance tracking"""
    model_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    prediction_latency: float = 0.0  # milliseconds
    throughput: float = 0.0  # predictions per second
    drift_score: float = 0.0
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def is_degraded(self, thresholds: Dict[str, float]) -> bool:
        """Check if performance has degraded below thresholds"""
        checks = {
            'accuracy': self.accuracy < thresholds.get('accuracy', 0.95),
            'precision': self.precision < thresholds.get('precision', 0.90),
            'recall': self.recall < thresholds.get('recall', 0.85),
            'f1_score': self.f1_score < thresholds.get('f1_score', 0.88),
            'auc_roc': self.auc_roc < thresholds.get('auc_roc', 0.90),
            'drift_score': self.drift_score > thresholds.get('drift_score', 0.1),
            'error_rate': self.error_rate > thresholds.get('error_rate', 0.05)
        }
        return any(checks.values())
    
    def get_degradation_reasons(self, thresholds: Dict[str, float]) -> List[str]:
        """Get list of metrics that have degraded"""
        reasons = []
        
        if self.accuracy < thresholds.get('accuracy', 0.95):
            reasons.append(f"Accuracy below threshold: {self.accuracy:.3f} < {thresholds.get('accuracy', 0.95)}")
        
        if self.precision < thresholds.get('precision', 0.90):
            reasons.append(f"Precision below threshold: {self.precision:.3f} < {thresholds.get('precision', 0.90)}")
        
        if self.recall < thresholds.get('recall', 0.85):
            reasons.append(f"Recall below threshold: {self.recall:.3f} < {thresholds.get('recall', 0.85)}")
        
        if self.f1_score < thresholds.get('f1_score', 0.88):
            reasons.append(f"F1 Score below threshold: {self.f1_score:.3f} < {thresholds.get('f1_score', 0.88)}")
        
        if self.auc_roc < thresholds.get('auc_roc', 0.90):
            reasons.append(f"AUC-ROC below threshold: {self.auc_roc:.3f} < {thresholds.get('auc_roc', 0.90)}")
        
        if self.drift_score > thresholds.get('drift_score', 0.1):
            reasons.append(f"Data drift detected: {self.drift_score:.3f} > {thresholds.get('drift_score', 0.1)}")
        
        if self.error_rate > thresholds.get('error_rate', 0.05):
            reasons.append(f"Error rate too high: {self.error_rate:.3f} > {thresholds.get('error_rate', 0.05)}")
        
        return reasons


@dataclass
class ModelVersion:
    """Comprehensive model version tracking"""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    model_artifacts: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    data_version: str = ""
    code_version: str = ""
    environment_spec: Dict[str, str] = field(default_factory=dict)
    parent_version: Optional[str] = None
    is_active: bool = False
    deployment_stage: str = "development"  # development, staging, production
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)