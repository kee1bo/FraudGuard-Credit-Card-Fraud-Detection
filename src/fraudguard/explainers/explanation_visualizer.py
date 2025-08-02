import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import io
import base64

class ExplanationVisualizer:
    """Create advanced visualizations for model explanations"""
    
    @staticmethod
    def create_shap_summary_plot(shap_values: np.ndarray, 
                                feature_values: np.ndarray,
                                feature_names: List[str]) -> str:
        """Create SHAP summary plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create summary plot data
        for i, feature_name in enumerate(feature_names[:15]):  # Top 15 features
            ax.scatter(shap_values[:, i], [i] * len(shap_values), 
                      c=feature_values[:, i], alpha=0.6, s=20, cmap='viridis')
        
        ax.set_yticks(range(len(feature_names[:15])))
        ax.set_yticklabels(feature_names[:15])
        ax.set_xlabel('SHAP Value')
        ax.set_title('SHAP Summary Plot')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    
    @staticmethod
    def create_explanation_dashboard(explanation_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a dashboard of explanation visualizations"""
        visualizations = {}
        
        # Feature importance plot
        if 'feature_importance' in explanation_data:
            importance = explanation_data['feature_importance']
            features = list(importance.keys())
            values = list(importance.values())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if v > 0 else 'blue' for v in values]
            ax.barh(features[:15], values[:15], color=colors)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            visualizations['feature_importance'] = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close()
        
        return visualizations