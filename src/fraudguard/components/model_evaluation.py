import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from fraudguard.utils.metrics import ModelMetrics
from fraudguard.utils.visualization import VisualizationUtils
from fraudguard.utils.common import save_json
from fraudguard.logger import fraud_logger
from fraudguard.constants.constants import *

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics_calculator = ModelMetrics(y_test, y_pred, y_pred_proba)
            metrics = metrics_calculator.calculate_all_metrics()
            
            # Generate visualizations
            cm_plot = VisualizationUtils.create_confusion_matrix_plot(
                confusion_matrix(y_test, y_pred)
            )
            
            # Store results
            evaluation_result = {
                'metrics': metrics,
                'confusion_matrix_plot': cm_plot,
                'model_name': model_name
            }
            
            self.evaluation_results[model_name] = evaluation_result
            
            fraud_logger.info(f"Evaluated {model_name}")
            return evaluation_result
            
        except Exception as e:
            fraud_logger.error(f"Error evaluating {model_name}: {e}")
            return None
    
    def compare_models(self, models, X_test, y_test):
        """Compare multiple models"""
        comparison_data = {}
        
        for model_name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, model_name)
            if result:
                comparison_data[model_name] = result['metrics']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Save comparison results
        save_json(comparison_data, REPORTS_DIR / "model_comparison.json")
        
        return comparison_df