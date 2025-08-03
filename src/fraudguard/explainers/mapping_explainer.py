"""
Mapping Explainer
Provides SHAP-based explanations for feature mapping operations,
showing how user inputs contribute to PCA component estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
import io

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingExplanation, MerchantCategory, 
    LocationRisk, SpendingPattern
)
from fraudguard.logger import fraud_logger


class MappingExplainer:
    """Explains feature mapping operations using SHAP and custom interpretability methods"""
    
    def __init__(self, feature_mapper, background_data: Optional[np.ndarray] = None):
        self.feature_mapper = feature_mapper
        self.background_data = background_data
        self.shap_explainer = None
        
        # Feature names for interpretable inputs
        self.input_feature_names = [
            'transaction_amount',
            'merchant_category', 
            'location_risk',
            'spending_pattern',
            'hour_sin',
            'hour_cos', 
            'day_of_week',
            'is_weekend'
        ]
        
        # PCA component names
        self.pca_feature_names = [f'V{i}' for i in range(1, 29)]
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE and hasattr(feature_mapper, 'model'):
            self._initialize_shap_explainer()
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for the feature mapper"""
        try:
            if self.background_data is None:
                # Create synthetic background data
                self.background_data = self._create_synthetic_background_data()
            
            # Initialize appropriate SHAP explainer based on model type
            model_name = getattr(self.feature_mapper, 'model_name', '').lower()
            
            if 'random_forest' in model_name or 'xgboost' in model_name:
                # Tree-based explainer
                self.shap_explainer = shap.TreeExplainer(
                    self.feature_mapper.model,
                    self.background_data
                )
            elif 'neural_network' in model_name:
                # Deep explainer for neural networks
                self.shap_explainer = shap.DeepExplainer(
                    self.feature_mapper.model,
                    self.background_data
                )
            else:
                # Kernel explainer as fallback
                def model_predict(X):
                    return self.feature_mapper.predict(X)
                
                self.shap_explainer = shap.KernelExplainer(
                    model_predict,
                    self.background_data
                )
            
            fraud_logger.info(f"SHAP explainer initialized for {model_name}")
            
        except Exception as e:
            fraud_logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _create_synthetic_background_data(self, n_samples: int = 100) -> np.ndarray:
        """Create synthetic background data for SHAP explainer"""
        # Create diverse background samples
        background_samples = []
        
        for _ in range(n_samples):
            # Random transaction amount (log-normal distribution)
            amount = np.random.lognormal(mean=4, sigma=1.5)  # ~$50-500 typical range
            amount = np.clip(amount, 1, 10000)
            
            # Random merchant category
            merchant_encoded = np.random.randint(0, len(MerchantCategory))
            
            # Random location risk
            location_encoded = np.random.randint(0, len(LocationRisk))
            
            # Random spending pattern
            spending_encoded = np.random.randint(0, len(SpendingPattern))
            
            # Random time features
            hour = np.random.randint(0, 24)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_of_week = np.random.randint(0, 7)
            is_weekend = int(day_of_week >= 5)
            
            sample = [
                amount, merchant_encoded, location_encoded, spending_encoded,
                hour_sin, hour_cos, day_of_week, is_weekend
            ]
            background_samples.append(sample)
        
        return np.array(background_samples)
    
    def explain_mapping(self, user_input: UserTransactionInput) -> MappingExplanation:
        """
        Generate comprehensive explanation for feature mapping
        
        Args:
            user_input: UserTransactionInput object
            
        Returns:
            MappingExplanation object with detailed explanations
        """
        try:
            # Convert user input to feature array
            input_features = self.feature_mapper._convert_user_input_to_features(user_input)
            
            # Get PCA estimates
            pca_estimates = self.feature_mapper.predict_single(user_input)
            pca_dict = {f'V{i+1}': float(pca_estimates[i]) for i in range(len(pca_estimates))}
            
            # Get SHAP explanations if available
            if self.shap_explainer is not None:
                input_contributions = self._get_shap_contributions(input_features)
            else:
                input_contributions = self._get_fallback_contributions(input_features)
            
            # Calculate confidence intervals (simplified)
            confidence_intervals = self._calculate_confidence_intervals(pca_estimates)
            
            # Generate business interpretation
            business_interpretation = self._generate_business_interpretation(
                user_input, input_contributions, pca_estimates
            )
            
            return MappingExplanation(
                input_contributions=input_contributions,
                pca_estimates=pca_dict,
                confidence_intervals=confidence_intervals,
                business_interpretation=business_interpretation,
                mapping_method=self.feature_mapper.model_name
            )
            
        except Exception as e:
            fraud_logger.error(f"Error generating mapping explanation: {e}")
            # Return basic explanation
            return self._create_basic_explanation(user_input)
    
    def _get_shap_contributions(self, input_features: np.ndarray) -> Dict[str, float]:
        """Get SHAP-based input contributions"""
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(input_features.reshape(1, -1))
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-output case - average across outputs
                shap_values = np.mean(shap_values, axis=0)
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Convert to contributions dictionary
            contributions = {}
            for i, feature_name in enumerate(self.input_feature_names):
                if i < len(shap_values):
                    contributions[feature_name] = float(shap_values[i])
                else:
                    contributions[feature_name] = 0.0
            
            # Normalize contributions to sum to 1
            total_abs_contribution = sum(abs(v) for v in contributions.values())
            if total_abs_contribution > 0:
                contributions = {
                    k: abs(v) / total_abs_contribution 
                    for k, v in contributions.items()
                }
            
            return contributions
            
        except Exception as e:
            fraud_logger.warning(f"Error getting SHAP contributions: {e}")
            return self._get_fallback_contributions(input_features)
    
    def _get_fallback_contributions(self, input_features: np.ndarray) -> Dict[str, float]:
        """Get fallback contributions when SHAP is not available"""
        # Use feature importance if available
        if hasattr(self.feature_mapper, 'get_feature_importance'):
            try:
                importance = self.feature_mapper.get_feature_importance()
                if importance:
                    return importance
            except Exception as e:
                fraud_logger.warning(f"Could not get feature importance: {e}")
        
        # Default equal contributions
        n_features = len(self.input_feature_names)
        return {name: 1.0/n_features for name in self.input_feature_names}
    
    def _calculate_confidence_intervals(self, pca_estimates: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for PCA estimates"""
        confidence_intervals = {}
        
        # Get uncertainty if available
        if hasattr(self.feature_mapper, 'predict_with_uncertainty'):
            try:
                input_features = np.random.randn(1, len(self.input_feature_names))  # Dummy input
                _, uncertainties = self.feature_mapper.predict_with_uncertainty(input_features)
                uncertainties = uncertainties[0]
            except Exception:
                uncertainties = np.ones(len(pca_estimates)) * 0.5  # Default uncertainty
        else:
            uncertainties = np.ones(len(pca_estimates)) * 0.5  # Default uncertainty
        
        # Calculate intervals
        for i, estimate in enumerate(pca_estimates):
            uncertainty = uncertainties[i] if i < len(uncertainties) else 0.5
            lower_bound = float(estimate - uncertainty)
            upper_bound = float(estimate + uncertainty)
            confidence_intervals[f'V{i+1}'] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _generate_business_interpretation(self, 
                                        user_input: UserTransactionInput,
                                        contributions: Dict[str, float],
                                        pca_estimates: np.ndarray) -> str:
        """Generate business-friendly interpretation"""
        interpretation_parts = []
        
        # Find most important contributing factors
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_contributions[:3]
        
        interpretation_parts.append("Key mapping factors:")
        
        for factor, contribution in top_factors:
            contribution_pct = contribution * 100
            
            if factor == 'transaction_amount':
                amount = user_input.transaction_amount
                if amount > 1000:
                    interpretation_parts.append(
                        f"• High transaction amount (${amount:.2f}) contributes {contribution_pct:.1f}% to the risk profile"
                    )
                else:
                    interpretation_parts.append(
                        f"• Transaction amount (${amount:.2f}) contributes {contribution_pct:.1f}% to the analysis"
                    )
            
            elif factor == 'merchant_category':
                merchant = user_input.merchant_category.value.replace('_', ' ').title()
                interpretation_parts.append(
                    f"• {merchant} merchant type contributes {contribution_pct:.1f}% to the pattern recognition"
                )
            
            elif factor == 'location_risk':
                location = user_input.location_risk.value.replace('_', ' ')
                if location != 'normal':
                    interpretation_parts.append(
                        f"• Unusual location ({location}) contributes {contribution_pct:.1f}% to the risk assessment"
                    )
                else:
                    interpretation_parts.append(
                        f"• Normal location contributes {contribution_pct:.1f}% to the confidence score"
                    )
            
            elif factor == 'spending_pattern':
                pattern = user_input.spending_pattern.value.replace('_', ' ')
                if pattern in ['suspicious', 'much_higher']:
                    interpretation_parts.append(
                        f"• {pattern.title()} spending pattern contributes {contribution_pct:.1f}% to the fraud indicators"
                    )
                else:
                    interpretation_parts.append(
                        f"• {pattern.title()} spending pattern contributes {contribution_pct:.1f}% to the analysis"
                    )
        
        # Add PCA mapping summary
        high_risk_components = sum(1 for estimate in pca_estimates if abs(estimate) > 2)
        if high_risk_components > 5:
            interpretation_parts.append(
                f"The mapping generated {high_risk_components} high-magnitude PCA components, indicating unusual transaction characteristics."
            )
        else:
            interpretation_parts.append(
                "The mapping generated PCA components within normal ranges, suggesting typical transaction patterns."
            )
        
        return " ".join(interpretation_parts)
    
    def _create_basic_explanation(self, user_input: UserTransactionInput) -> MappingExplanation:
        """Create basic explanation when advanced methods fail"""
        # Basic contributions based on input characteristics
        contributions = {
            'transaction_amount': 0.3,
            'merchant_category': 0.25,
            'location_risk': 0.2,
            'spending_pattern': 0.15,
            'hour_sin': 0.05,
            'hour_cos': 0.05,
            'day_of_week': 0.0,
            'is_weekend': 0.0
        }
        
        # Basic PCA estimates (zeros)
        pca_estimates = {f'V{i}': 0.0 for i in range(1, 29)}
        
        # Basic confidence intervals
        confidence_intervals = {f'V{i}': (-1.0, 1.0) for i in range(1, 29)}
        
        # Basic interpretation
        business_interpretation = (
            f"Basic mapping analysis for ${user_input.transaction_amount:.2f} "
            f"{user_input.merchant_category.value} transaction. "
            f"Advanced explanations are not available."
        )
        
        return MappingExplanation(
            input_contributions=contributions,
            pca_estimates=pca_estimates,
            confidence_intervals=confidence_intervals,
            business_interpretation=business_interpretation,
            mapping_method="basic"
        )
    
    def create_visualization(self, 
                           explanation: MappingExplanation,
                           save_path: Optional[str] = None) -> Optional[str]:
        """
        Create visualization of the mapping explanation
        
        Args:
            explanation: MappingExplanation object
            save_path: Optional path to save the plot
            
        Returns:
            Base64 encoded plot image or None
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Input contributions bar chart
            contributions = explanation.input_contributions
            features = list(contributions.keys())
            values = list(contributions.values())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            bars = ax1.bar(features, values, color=colors)
            ax1.set_title('Input Feature Contributions', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Contribution')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 2. Top PCA components
            pca_estimates = explanation.pca_estimates
            # Get top 10 components by absolute value
            sorted_pca = sorted(pca_estimates.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            
            pca_names = [item[0] for item in sorted_pca]
            pca_values = [item[1] for item in sorted_pca]
            
            colors = ['red' if v < 0 else 'blue' for v in pca_values]
            bars = ax2.bar(pca_names, pca_values, color=colors, alpha=0.7)
            ax2.set_title('Top 10 PCA Component Estimates', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Estimated Value')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Confidence intervals for top components
            top_5_pca = sorted_pca[:5]
            pca_names_top5 = [item[0] for item in top_5_pca]
            pca_values_top5 = [item[1] for item in top_5_pca]
            
            confidence_intervals = explanation.confidence_intervals
            lower_bounds = [confidence_intervals[name][0] for name in pca_names_top5]
            upper_bounds = [confidence_intervals[name][1] for name in pca_names_top5]
            errors = [[val - lower for val, lower in zip(pca_values_top5, lower_bounds)],
                     [upper - val for val, upper in zip(pca_values_top5, upper_bounds)]]
            
            ax3.errorbar(pca_names_top5, pca_values_top5, yerr=errors, 
                        fmt='o', capsize=5, capthick=2, markersize=8)
            ax3.set_title('Top 5 PCA Components with Confidence Intervals', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Estimated Value')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Feature contribution pie chart
            # Only show top 5 contributions for clarity
            top_contributions = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5])
            other_contribution = sum(contributions.values()) - sum(top_contributions.values())
            if other_contribution > 0:
                top_contributions['Others'] = other_contribution
            
            wedges, texts, autotexts = ax4.pie(
                top_contributions.values(), 
                labels=top_contributions.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            ax4.set_title('Feature Contribution Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save or return as base64
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                return plot_data
                
        except Exception as e:
            fraud_logger.error(f"Error creating visualization: {e}")
            return None
    
    def generate_detailed_report(self, 
                               user_input: UserTransactionInput,
                               explanation: MappingExplanation) -> Dict[str, Any]:
        """Generate detailed explanation report"""
        report = {
            'summary': {
                'transaction_amount': user_input.transaction_amount,
                'merchant_category': user_input.merchant_category.value,
                'location_risk': user_input.location_risk.value,
                'spending_pattern': user_input.spending_pattern.value,
                'mapping_method': explanation.mapping_method
            },
            'input_analysis': self._analyze_inputs(user_input, explanation.input_contributions),
            'pca_analysis': self._analyze_pca_components(explanation.pca_estimates),
            'confidence_analysis': self._analyze_confidence(explanation.confidence_intervals),
            'business_interpretation': explanation.business_interpretation,
            'recommendations': self._generate_recommendations(user_input, explanation)
        }
        
        return report
    
    def _analyze_inputs(self, user_input: UserTransactionInput, contributions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze input contributions"""
        analysis = {}
        
        for feature, contribution in contributions.items():
            if feature == 'transaction_amount':
                analysis[feature] = {
                    'value': user_input.transaction_amount,
                    'contribution': contribution,
                    'interpretation': self._interpret_amount_contribution(
                        user_input.transaction_amount, contribution
                    )
                }
            elif feature == 'merchant_category':
                analysis[feature] = {
                    'value': user_input.merchant_category.value,
                    'contribution': contribution,
                    'interpretation': f"Merchant type contributes {contribution*100:.1f}% to the mapping"
                }
            # Add other features...
        
        return analysis
    
    def _interpret_amount_contribution(self, amount: float, contribution: float) -> str:
        """Interpret amount contribution"""
        if contribution > 0.3:
            if amount > 1000:
                return f"High amount (${amount:.2f}) is a major factor in fraud risk assessment"
            else:
                return f"Amount (${amount:.2f}) significantly influences the transaction profile"
        else:
            return f"Amount (${amount:.2f}) has moderate influence on the analysis"
    
    def _analyze_pca_components(self, pca_estimates: Dict[str, float]) -> Dict[str, Any]:
        """Analyze PCA component estimates"""
        values = list(pca_estimates.values())
        
        analysis = {
            'total_components': len(values),
            'high_magnitude_count': sum(1 for v in values if abs(v) > 2),
            'mean_absolute_value': np.mean([abs(v) for v in values]),
            'max_component': max(pca_estimates.items(), key=lambda x: abs(x[1])),
            'distribution_summary': {
                'positive_components': sum(1 for v in values if v > 0),
                'negative_components': sum(1 for v in values if v < 0),
                'near_zero_components': sum(1 for v in values if abs(v) < 0.5)
            }
        }
        
        return analysis
    
    def _analyze_confidence(self, confidence_intervals: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze confidence intervals"""
        interval_widths = [upper - lower for lower, upper in confidence_intervals.values()]
        
        analysis = {
            'average_interval_width': np.mean(interval_widths),
            'max_interval_width': max(interval_widths),
            'min_interval_width': min(interval_widths),
            'high_uncertainty_components': [
                comp for comp, (lower, upper) in confidence_intervals.items()
                if (upper - lower) > 2.0
            ]
        }
        
        return analysis
    
    def _generate_recommendations(self, 
                                user_input: UserTransactionInput,
                                explanation: MappingExplanation) -> List[str]:
        """Generate recommendations based on the explanation"""
        recommendations = []
        
        # Check for high uncertainty
        high_uncertainty_count = sum(
            1 for lower, upper in explanation.confidence_intervals.values()
            if (upper - lower) > 2.0
        )
        
        if high_uncertainty_count > 10:
            recommendations.append(
                "High uncertainty detected in mapping - consider using ensemble approach"
            )
        
        # Check input consistency
        if user_input.spending_pattern == SpendingPattern.SUSPICIOUS:
            recommendations.append(
                "Suspicious spending pattern detected - additional verification recommended"
            )
        
        if user_input.location_risk != LocationRisk.NORMAL:
            recommendations.append(
                "Unusual location detected - consider additional location verification"
            )
        
        # Check amount vs merchant consistency
        amount = user_input.transaction_amount
        merchant = user_input.merchant_category
        
        if merchant == MerchantCategory.GROCERY and amount > 500:
            recommendations.append(
                "High amount for grocery transaction - verify transaction details"
            )
        elif merchant == MerchantCategory.ATM_WITHDRAWAL and amount > 1000:
            recommendations.append(
                "High ATM withdrawal amount - additional authentication recommended"
            )
        
        if not recommendations:
            recommendations.append("Transaction mapping appears normal - no specific recommendations")
        
        return recommendations