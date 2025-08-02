import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import io
import base64

class VisualizationUtils:
    """Utility class for creating visualizations"""
    
    @staticmethod
    def plot_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        return base64.b64encode(plot_data).decode()
    
    @staticmethod
    def create_confusion_matrix_plot(cm: np.ndarray, labels: List[str] = None) -> str:
        """Create confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if labels is None:
            labels = ['Normal', 'Fraud']
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return VisualizationUtils.plot_to_base64(fig)
    
    @staticmethod
    def create_feature_importance_plot(feature_names: List[str], 
                                     importance_values: List[float],
                                     top_n: int = 15) -> str:
        """Create feature importance bar plot"""
        # Sort features by importance
        sorted_indices = np.argsort(importance_values)[-top_n:]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_values = [importance_values[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(sorted_names)), sorted_values)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))
        
        plt.tight_layout()
        return VisualizationUtils.plot_to_base64(fig)