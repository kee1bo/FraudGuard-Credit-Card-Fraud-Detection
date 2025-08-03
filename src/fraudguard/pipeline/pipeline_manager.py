"""
Singleton Pipeline Manager to avoid multiple model loading
"""

from .prediction_pipeline import PredictionPipeline
from ..logger import fraud_logger

class PipelineManager:
    """Singleton manager for prediction pipeline"""
    _instance = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineManager, cls).__new__(cls)
        return cls._instance
    
    def get_pipeline(self):
        """Get the prediction pipeline (initialize once)"""
        if self._pipeline is None:
            try:
                fraud_logger.info("Initializing prediction pipeline (singleton)...")
                self._pipeline = PredictionPipeline()
                fraud_logger.info("Prediction pipeline initialized successfully")
            except Exception as e:
                fraud_logger.error(f"Failed to initialize prediction pipeline: {e}")
                self._pipeline = None
        return self._pipeline
    
    def get_available_models(self):
        """Get available models"""
        pipeline = self.get_pipeline()
        if pipeline:
            return pipeline.get_available_models()
        return []

# Global instance
pipeline_manager = PipelineManager()