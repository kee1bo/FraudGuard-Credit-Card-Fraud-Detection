import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance across models"""
    
    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        self.models = models
        self.feature_names = feature_names
    
    def get_model_feature_importance(self, model_name: str) -> np.ndarray:
        """Get feature importance for a specific model"""
        model = self.models[model_name]
        
        if hasattr(model.model, 'feature_importances_'):
            return model.model.feature_importances_
        elif hasattr(model.model, 'coef_'):
            return np.abs(model.model.coef_[0])
        else:
            return np.zeros(len(self.feature_names))
    
    def compare_feature_importance(self) -> pd.DataFrame:
        """Compare feature importance across all models"""
        importance_data = {}
        
        for model_name in self.models.keys():
            importance_data[model_name] = self.get_model_feature_importance(model_name)
        
        df = pd.DataFrame(importance_data, index=self.feature_names)
        return df
    
    def plot_comparison(self, top_n: int = 15) -> str:
        """Create comparison plot of feature importance"""
        df = self.compare_feature_importance()
        
        # Get top features by average importance
        df['average'] = df.mean(axis=1)
        top_features = df.nlargest(top_n, 'average').drop('average', axis=1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importance Comparison')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.legend(title='Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to base64
        import io
        import base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()