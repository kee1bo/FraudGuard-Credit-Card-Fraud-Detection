"""
Professional Chart Generator for Reports
"""

from typing import Dict, List, Any, Optional
import json
import base64
from io import BytesIO

from ..logger import fraud_logger


class ChartGenerator:
    """Generate professional charts for reports"""
    
    def __init__(self):
        fraud_logger.info("Chart generator initialized")
    
    def generate_performance_chart_data(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart data for model performance visualization"""
        try:
            models = model_data.get('model_comparison', {})
            if not models:
                return {}
            
            chart_data = {
                'labels': [],
                'auc_scores': [],
                'f1_scores': [],
                'precision_scores': [],
                'recall_scores': []
            }
            
            for model_name, metrics in models.items():
                chart_data['labels'].append(model_name.replace('_', ' ').title())
                chart_data['auc_scores'].append(metrics.get('auc_roc', 0))
                chart_data['f1_scores'].append(metrics.get('f1_score', 0))
                chart_data['precision_scores'].append(metrics.get('precision', 0))
                chart_data['recall_scores'].append(metrics.get('recall', 0))
            
            return chart_data
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate chart data: {e}")
            return {}
    
    def generate_feature_importance_data(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feature importance chart data"""
        try:
            if not feature_data or 'top_features' not in feature_data:
                return {}
            
            top_features = feature_data['top_features']
            
            chart_data = {
                'features': [],
                'importance_values': [],
                'colors': []
            }
            
            # Get top 10 features
            sorted_features = sorted(
                top_features.items(),
                key=lambda x: x[1].get('mean_importance', 0),
                reverse=True
            )[:10]
            
            for feature, importance_data in sorted_features:
                chart_data['features'].append(feature)
                chart_data['importance_values'].append(importance_data.get('mean_importance', 0))
                chart_data['colors'].append('#486581')  # Professional blue
            
            return chart_data
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate feature importance data: {e}")
            return {}
    
    def create_chart_config(self, chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Chart.js configuration for professional charts"""
        try:
            base_config = {
                'type': chart_type,
                'data': data,
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {
                            'position': 'bottom',
                            'labels': {
                                'usePointStyle': True,
                                'padding': 20,
                                'font': {
                                    'family': 'Inter, sans-serif',
                                    'size': 12,
                                    'weight': '500'
                                },
                                'color': '#4a5568'
                            }
                        },
                        'tooltip': {
                            'backgroundColor': 'rgba(26, 32, 44, 0.95)',
                            'titleColor': '#ffffff',
                            'bodyColor': '#ffffff',
                            'borderColor': '#486581',
                            'borderWidth': 1,
                            'cornerRadius': 8
                        }
                    },
                    'animation': {
                        'duration': 750,
                        'easing': 'easeInOutCubic'
                    }
                }
            }
            
            return base_config
            
        except Exception as e:
            fraud_logger.error(f"Failed to create chart config: {e}")
            return {}