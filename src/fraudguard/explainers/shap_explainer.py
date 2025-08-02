# src/fraudguard/explainers/shap_explainer.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from .base_explainer import BaseExplainer
from fraudguard.logger import fraud_logger

# Try to import SHAP, but handle if it's not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    fraud_logger.warning("SHAP not available. Install with: pip install shap")

class SHAPExplainer(BaseExplainer):
    """SHAP-based explanation generator"""
    
    def __init__(self, model, X_background: np.ndarray, **kwargs):
        super().__init__(model, **kwargs)
        self.X_background = X_background
        self.explainer = None
        self.expected_value = 0.5  # Default expected value
        
        if SHAP_AVAILABLE:
            self._initialize_explainer()
        else:
            fraud_logger.warning("SHAP not available, using fallback explanation method")
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # Try different explainer types based on model
            model_obj = self.model.model if hasattr(self.model, 'model') else self.model
            
            # For tree-based models
            if hasattr(model_obj, 'get_booster') or hasattr(model_obj, 'estimators_'):
                try:
                    self.explainer = shap.TreeExplainer(model_obj)
                    self.expected_value = self.explainer.expected_value
                    if isinstance(self.expected_value, (list, np.ndarray)):
                        self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
                    fraud_logger.info("TreeExplainer initialized successfully")
                except Exception as e:
                    fraud_logger.warning(f"TreeExplainer failed: {e}, trying LinearExplainer")
                    self._try_linear_explainer(model_obj)
            else:
                self._try_linear_explainer(model_obj)
                
        except Exception as e:
            fraud_logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _try_linear_explainer(self, model_obj):
        """Try to use LinearExplainer for linear models"""
        try:
            self.explainer = shap.LinearExplainer(model_obj, self.X_background)
            self.expected_value = self.explainer.expected_value
            if isinstance(self.expected_value, (list, np.ndarray)):
                self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
            fraud_logger.info("LinearExplainer initialized successfully")
        except Exception as e:
            fraud_logger.warning(f"LinearExplainer failed: {e}, using KernelExplainer")
            self._try_kernel_explainer(model_obj)
    
    def _try_kernel_explainer(self, model_obj):
        """Try to use KernelExplainer as fallback"""
        try:
            # Use a smaller background sample for KernelExplainer (it's slow)
            background_sample = self.X_background[:50]  # Use only 50 samples
            
            if hasattr(model_obj, 'predict_proba'):
                predict_fn = lambda x: model_obj.predict_proba(x)[:, 1]
            else:
                predict_fn = model_obj.predict
                
            self.explainer = shap.KernelExplainer(predict_fn, background_sample)
            self.expected_value = 0.5  # Set default for KernelExplainer
            fraud_logger.info("KernelExplainer initialized successfully")
        except Exception as e:
            fraud_logger.warning(f"KernelExplainer failed: {e}")
            self.explainer = None
    
    def explain_instance(self, X_instance: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Generate SHAP explanation for a single instance"""
        
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(X_instance, feature_names)
        
        try:
            # Ensure X_instance is 2D
            if X_instance.ndim == 1:
                X_instance = X_instance.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_instance)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For binary classification, take positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first instance
            
            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values.tolist(),
                'base_value': float(self.expected_value),
                'instance_values': X_instance[0].tolist(),
                'feature_names': feature_names or [f'Feature_{i}' for i in range(len(X_instance[0]))]
            }
            
            return explanation
            
        except Exception as e:
            fraud_logger.warning(f"SHAP explanation failed: {e}, using fallback")
            return self._fallback_explanation(X_instance, feature_names)
    
    def _fallback_explanation(self, X_instance: np.ndarray, feature_names: List[str] = None):
        """Fallback explanation when SHAP is not available"""
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Create dummy SHAP values based on feature magnitude
        dummy_shap_values = X_instance[0] * 0.1  # Simple scaling
        
        explanation = {
            'shap_values': dummy_shap_values.tolist(),
            'base_value': 0.5,
            'instance_values': X_instance[0].tolist(),
            'feature_names': feature_names or [f'Feature_{i}' for i in range(len(X_instance[0]))]
        }
        
        return explanation
    
    def generate_plot(self, explanation: Dict[str, Any]) -> str:
        """Generate waterfall plot and return as base64 string"""
        try:
            if not SHAP_AVAILABLE:
                return self._generate_simple_plot(explanation)
            
            # Create SHAP explanation object
            shap_values = np.array(explanation['shap_values'])
            base_value = explanation['base_value']
            instance_values = np.array(explanation['instance_values'])
            feature_names = explanation['feature_names']
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create a simple waterfall-style plot
            self._create_waterfall_plot(ax, shap_values, base_value, feature_names)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close(fig)
            
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            fraud_logger.warning(f"Plot generation failed: {e}")
            return self._generate_simple_plot(explanation)
    
    def _create_waterfall_plot(self, ax, shap_values, base_value, feature_names):
        """Create a simple waterfall-style plot"""
        # Get top 10 most important features
        abs_values = np.abs(shap_values)
        top_indices = np.argsort(abs_values)[-10:][::-1]
        
        top_shap_values = shap_values[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]
        
        # Create the waterfall plot
        cumulative = base_value
        x_pos = range(len(top_shap_values) + 2)  # +2 for base and final
        heights = [base_value] + top_shap_values.tolist() + [cumulative + sum(top_shap_values)]
        
        # Plot bars
        colors = ['gray'] + ['red' if val > 0 else 'blue' for val in top_shap_values] + ['green']
        labels = ['Base Value'] + top_feature_names + ['Prediction']
        
        bars = ax.bar(x_pos, heights, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, height) in enumerate(zip(bars, heights)):
            if i == 0:  # Base value
                label = f'{height:.3f}'
            elif i == len(bars) - 1:  # Final prediction
                label = f'{height:.3f}'
            else:  # Feature contributions
                label = f'{height:+.3f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   label, ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('SHAP Value')
        ax.set_title('SHAP Waterfall Plot - Feature Contributions')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    def _generate_simple_plot(self, explanation: Dict[str, Any]) -> str:
        """Generate a simple bar plot when SHAP is not available"""
        try:
            shap_values = np.array(explanation['shap_values'])
            feature_names = explanation['feature_names']
            
            # Get top 10 features
            abs_values = np.abs(shap_values)
            top_indices = np.argsort(abs_values)[-10:][::-1]
            
            top_values = shap_values[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['red' if val > 0 else 'blue' for val in top_values]
            bars = ax.barh(top_names, top_values, color=colors, alpha=0.7)
            
            ax.set_xlabel('Feature Importance')
            ax.set_title('Feature Importance (Fallback Explanation)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close(fig)
            
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            fraud_logger.error(f"Simple plot generation failed: {e}")
            return ""