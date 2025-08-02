import sys
import json
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from flask import Blueprint, render_template, request, jsonify
from fraudguard.pipeline.prediction_pipeline import PredictionPipeline

dashboard_bp = Blueprint('dashboard', __name__)

# Initialize prediction pipeline for metrics
try:
    prediction_pipeline = PredictionPipeline()
    available_models = prediction_pipeline.get_available_models()
except Exception as e:
    print(f"Warning: Could not initialize prediction pipeline: {e}")
    prediction_pipeline = None
    available_models = []

def load_model_metrics():
    """Load model metrics from the comparison report"""
    try:
        metrics_file = Path("artifacts/reports/model_comparison.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Warning: Could not load model metrics: {e}")
        return {}

@dashboard_bp.route('/')
def dashboard():
    """Model performance dashboard"""
    # Get metrics from the comparison report
    all_metrics = load_model_metrics()
    
    # If we have models from pipeline but no metrics, create empty entries
    if available_models and not all_metrics:
        all_metrics = {model: {"error": "Metrics not available"} for model in available_models}
    
    return render_template('dashboard-redesigned.html', 
                         models=available_models,
                         metrics=all_metrics)

@dashboard_bp.route('/comparison')
def comparison():
    """Model comparison page"""
    # Load metrics from the comparison report
    all_metrics = load_model_metrics()
    
    # Use models from metrics if pipeline models are empty
    models_to_use = list(all_metrics.keys()) if all_metrics else available_models
    
    # If no metrics loaded, try to provide default empty structure
    if not all_metrics and models_to_use:
        all_metrics = {}
        for model in models_to_use:
            all_metrics[model] = {
                "error": "Metrics not available - please run training first",
                "roc_auc_score": 0,
                "classification_report": {
                    "1": {
                        "precision": 0,
                        "recall": 0,
                        "f1-score": 0
                    }
                }
            }
    
    print(f"Debug - Available models: {models_to_use}")
    print(f"Debug - Metrics keys: {list(all_metrics.keys()) if all_metrics else 'None'}")
    
    return render_template('comparison.html', 
                         models=models_to_use,
                         metrics=all_metrics)