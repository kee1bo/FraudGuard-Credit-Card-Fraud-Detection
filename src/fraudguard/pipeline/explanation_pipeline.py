import numpy as np
from typing import Dict, Any, List
from fraudguard.explainers.shap_explainer import SHAPExplainer
from fraudguard.explainers.lime_explainer import LIMEExplainer
from fraudguard.utils.common import load_object
from fraudguard.logger import fraud_logger

class ExplanationPipeline:
    """Pipeline for generating model explanations"""
    
    def __init__(self, model, X_background, feature_names):
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.explainers = {}
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize all available explainers"""
        try:
            # Initialize SHAP explainer
            self.explainers['shap'] = SHAPExplainer(self.model, self.X_background)
            
            # Initialize LIME explainer
            self.explainers['lime'] = LIMEExplainer(
                self.model, self.X_background, self.feature_names
            )
            
            fraud_logger.info("Initialized all explainers")
            
        except Exception as e:
            fraud_logger.error(f"Error initializing explainers: {e}")
    
    def explain_prediction(self, X_instance, explainer_type='shap'):
        """Generate explanation for a prediction"""
        try:
            if explainer_type not in self.explainers:
                raise ValueError(f"Explainer {explainer_type} not available")
            
            explainer = self.explainers[explainer_type]
            explanation = explainer.explain_instance(X_instance, feature_names=self.feature_names)
            plot = explainer.generate_plot(explanation)
            
            return {
                'explanation': explanation,
                'plot': plot,
                'explainer_type': explainer_type
            }
            
        except Exception as e:
            fraud_logger.error(f"Error generating explanation: {e}")
            return None
    
    def compare_explanations(self, X_instance):
        """Compare explanations from different explainers"""
        results = {}
        
        for explainer_type in self.explainers.keys():
            result = self.explain_prediction(X_instance, explainer_type)
            if result:
                results[explainer_type] = result
        
        return results

