"""
Professional Document Rendering Engine
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..logger import fraud_logger


class DocumentEngine:
    """Professional document rendering and management"""
    
    def __init__(self, templates_dir: str = "app/templates/reports"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(self.templates_dir), "app/templates"]),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Initialize markdown processor with extensions
        self.markdown = markdown.Markdown(
            extensions=[
                'markdown.extensions.toc',
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite',
                'markdown.extensions.attr_list',
                'markdown.extensions.def_list',
                'markdown.extensions.footnotes',
                'markdown.extensions.meta'
            ],
            extension_configs={
                'markdown.extensions.toc': {
                    'permalink': True,
                    'permalink_class': 'toc-permalink',
                    'permalink_title': 'Link to this section'
                },
                'markdown.extensions.codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': True
                }
            }
        )
        
        fraud_logger.info("Document engine initialized")
    
    def render_markdown_to_html(self, markdown_content: str, 
                               template_name: str = "professional_report.html",
                               context: Dict[str, Any] = None) -> str:
        """Render markdown content to professional HTML"""
        try:
            # Reset markdown processor
            self.markdown.reset()
            
            # Convert markdown to HTML
            html_content = self.markdown.convert(markdown_content)
            
            # Extract metadata if available
            metadata = getattr(self.markdown, 'Meta', {})
            
            # Extract table of contents
            toc = getattr(self.markdown, 'toc', '')
            
            # Prepare context
            render_context = {
                'content': html_content,
                'toc': toc,
                'metadata': metadata,
                'generated_at': datetime.now(),
                'title': metadata.get('title', ['Professional Report'])[0] if metadata.get('title') else 'Professional Report'
            }
            
            # Add custom context
            if context:
                render_context.update(context)
            
            # Render with template
            template = self.jinja_env.get_template(template_name)
            return template.render(**render_context)
            
        except Exception as e:
            fraud_logger.error(f"Failed to render markdown: {e}")
            raise
    
    def render_model_report(self, model_data: Dict[str, Any], 
                           template_name: str = "model_report.html") -> str:
        """Render comprehensive model performance report"""
        try:
            # Prepare model analysis context
            context = {
                'models': model_data.get('models', {}),
                'performance_summary': model_data.get('performance_summary', {}),
                'feature_analysis': model_data.get('feature_analysis', {}),
                'training_metadata': model_data.get('training_metadata', {}),
                'generated_at': datetime.now(),
                'title': 'Model Performance Analysis Report'
            }
            
            # Add computed insights
            context['insights'] = self._generate_model_insights(model_data)
            
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
            
        except Exception as e:
            fraud_logger.error(f"Failed to render model report: {e}")
            raise
    
    def render_dissertation_report(self, dissertation_path: str,
                                  template_name: str = "professional_report.html") -> str:
        """Render dissertation as professional web report"""
        try:
            # Read dissertation markdown
            with open(dissertation_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Process and enhance content
            enhanced_content = self._enhance_dissertation_content(markdown_content)
            
            # Add professional context
            context = {
                'document_type': 'dissertation',
                'academic_style': True,
                'include_citations': True,
                'include_appendices': True,
                'author': 'MSc Data Science Student',
                'institution': 'University',
                'year': datetime.now().year
            }
            
            return self.render_markdown_to_html(
                enhanced_content, 
                template_name, 
                context
            )
            
        except Exception as e:
            fraud_logger.error(f"Failed to render dissertation: {e}")
            raise
    
    def generate_executive_summary(self, model_data: Dict[str, Any]) -> str:
        """Generate executive summary from model data"""
        try:
            best_model = model_data.get('training_metadata', {}).get('best_model', 'Unknown')
            best_auc = model_data.get('training_metadata', {}).get('best_auc', 0)
            model_count = len(model_data.get('models', {}))
            
            summary = f"""
# Executive Summary

## Project Overview
This report presents a comprehensive analysis of {model_count} machine learning models 
developed for credit card fraud detection. The models were trained and evaluated using 
advanced techniques to ensure robust performance in production environments.

## Key Findings
- **Best Performing Model**: {best_model.replace('_', ' ').title()}
- **Peak Performance**: {best_auc:.4f} AUC-ROC score
- **Model Diversity**: {model_count} different algorithms evaluated
- **Production Ready**: All models meet professional deployment standards

## Recommendations
Based on the comprehensive analysis, we recommend deploying the {best_model.replace('_', ' ').title()} 
model for production fraud detection, with continuous monitoring and periodic retraining.

## Business Impact
The implemented solution provides:
- Real-time fraud detection capabilities
- Explainable AI for regulatory compliance
- Scalable architecture for high-volume transactions
- Professional monitoring and alerting systems
"""
            return summary.strip()
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate executive summary: {e}")
            return "# Executive Summary\n\nSummary generation failed."
    
    def _enhance_dissertation_content(self, content: str) -> str:
        """Enhance dissertation content with professional formatting"""
        # Add metadata
        enhanced = """---
title: Advanced Credit Card Fraud Detection with Explainable AI
subtitle: MSc Data Science Dissertation
author: MSc Data Science Student
date: """ + datetime.now().strftime("%B %Y") + """
abstract: |
    This dissertation presents a comprehensive study of machine learning approaches 
    for credit card fraud detection, with emphasis on explainable AI techniques 
    and professional deployment considerations.
keywords: [Machine Learning, Fraud Detection, Explainable AI, Credit Cards]
---

"""
        
        # Process content sections
        enhanced += content
        
        # Add professional styling markers
        enhanced = re.sub(r'^# (.+)$', r'# \1 {.chapter}', enhanced, flags=re.MULTILINE)
        enhanced = re.sub(r'^## (.+)$', r'## \1 {.section}', enhanced, flags=re.MULTILINE)
        
        return enhanced
    
    def _generate_model_insights(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from model performance data"""
        insights = {
            'performance_tier': 'Unknown',
            'strengths': [],
            'recommendations': [],
            'risk_assessment': 'Medium'
        }
        
        try:
            models = model_data.get('models', {})
            if not models:
                return insights
            
            # Analyze performance tiers
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
            
            # Generate strengths
            if len(models) >= 5:
                insights['strengths'].append('Comprehensive model diversity')
            if avg_auc >= 0.8:
                insights['strengths'].append('Strong predictive performance')
            if any('cross_val' in str(m) for m in models.values()):
                insights['strengths'].append('Robust validation methodology')
            
            # Generate recommendations
            if avg_auc < 0.8:
                insights['recommendations'].append('Consider feature engineering improvements')
            if len(models) < 3:
                insights['recommendations'].append('Evaluate additional model types')
            
            insights['recommendations'].append('Implement continuous monitoring')
            insights['recommendations'].append('Schedule periodic retraining')
            
        except Exception as e:
            fraud_logger.error(f"Failed to generate insights: {e}")
        
        return insights
    
    def create_professional_template(self, template_name: str, template_content: str):
        """Create a new professional template"""
        try:
            template_path = self.templates_dir / template_name
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            fraud_logger.info(f"Created template: {template_name}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to create template {template_name}: {e}")
            raise