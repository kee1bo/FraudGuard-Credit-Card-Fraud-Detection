"""
Professional Report Generator
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from ..logger import fraud_logger


class ReportGenerator:
    """Generate professional reports from model data"""
    
    def __init__(self):
        fraud_logger.info("Report generator initialized")
    
    def generate_model_summary(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model performance summary"""
        try:
            summary = {
                'total_models': len(model_data.get('model_comparison', {})),
                'best_model': model_data.get('training_metadata', {}).get('best_model', 'Unknown'),
                'best_auc': model_data.get('training_metadata', {}).get('best_auc', 0),
                'average_auc': model_data.get('performance_summary', {}).get('average_auc', 0),
                'generated_at': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate model summary: {e}")
            return {}
    
    def generate_insights(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from model performance"""
        try:
            insights = {
                'performance_tier': 'Unknown',
                'strengths': [],
                'recommendations': [],
                'risk_assessment': 'Medium'
            }
            
            models = model_data.get('model_comparison', {})
            if not models:
                return insights
            
            # Analyze performance
            auc_scores = [m.get('auc_roc', 0) for m in models.values()]
            avg_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0
            
            if avg_auc >= 0.9:
                insights['performance_tier'] = 'Excellent'
                insights['risk_assessment'] = 'Low'
            elif avg_auc >= 0.8:
                insights['performance_tier'] = 'Good'
                insights['risk_assessment'] = 'Medium'
            elif avg_auc >= 0.7:
                insights['performance_tier'] = 'Fair'
                insights['risk_assessment'] = 'Medium-High'
            else:
                insights['performance_tier'] = 'Needs Improvement'
                insights['risk_assessment'] = 'High'
            
            # Generate strengths and recommendations
            if len(models) >= 5:
                insights['strengths'].append('Comprehensive model diversity')
            if avg_auc >= 0.8:
                insights['strengths'].append('Strong predictive performance')
            
            if avg_auc < 0.8:
                insights['recommendations'].append('Consider feature engineering improvements')
            
            insights['recommendations'].append('Implement continuous monitoring')
            insights['recommendations'].append('Schedule periodic retraining')
            
            return insights
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate insights: {e}")
            return {}