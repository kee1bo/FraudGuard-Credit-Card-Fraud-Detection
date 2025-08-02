import lime
import lime.lime_tabular
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import io
import base64
from .base_explainer import BaseExplainer

class LIMEExplainer(BaseExplainer):
    """LIME-based explanation generator"""
    
    def __init__(self, model, X_training: np.ndarray, feature_names: List[str], **kwargs):
        super().__init__(model, **kwargs)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_training,
            feature_names=feature_names,
            class_names=['Normal', 'Fraud'],
            mode='classification'
        )
        self.feature_names = feature_names
        
    def explain_instance(self, X_instance: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate LIME explanation for a single instance"""
        # Generate explanation
        exp = self.explainer.explain_instance(
            X_instance, 
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Extract explanation data
        explanation = {
            'feature_importance': dict(exp.as_list()),
            'score': exp.score,
            'intercept': exp.intercept[1],
            'instance_values': X_instance.tolist(),
            'feature_names': self.feature_names
        }
        
        return explanation
    
    def generate_plot(self, explanation: Dict[str, Any]) -> str:
        """Generate LIME plot and return as base64 string"""
        feature_importance = explanation['feature_importance']
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(feature_importance.keys())
        values = list(feature_importance.values())
        
        colors = ['red' if v > 0 else 'blue' for v in values]
        bars = ax.barh(features, values, color=colors)
        
        ax.set_xlabel('Feature Importance')
        ax.set_title('LIME Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()