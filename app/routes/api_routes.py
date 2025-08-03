import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from flask import Blueprint, request, jsonify
from fraudguard.pipeline.pipeline_manager import pipeline_manager

api_bp = Blueprint('api', __name__)

@api_bp.route('/models')
def get_available_models():
    """API endpoint to get available models"""
    if prediction_pipeline:
        return jsonify({"models": prediction_pipeline.get_available_models()})
    else:
        return jsonify({"error": "Prediction pipeline not available"}), 500

@api_bp.route('/model_metrics/<model_type>')
def get_model_metrics(model_type):
    """API endpoint for model metrics"""
    try:
        if not prediction_pipeline:
            return jsonify({"error": "Prediction pipeline not available"}), 500
            
        metrics = prediction_pipeline.get_model_metrics(model_type)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_bp.route('/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not prediction_pipeline:
            return jsonify({"error": "Prediction pipeline not available"}), 500
            
        data = request.get_json()
        result = prediction_pipeline.predict_single_transaction(
            data['transaction_data'],
            model_type=data.get('model_type', 'xgboost'),
            include_explanation=data.get('include_explanation', True)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    status = "healthy" if prediction_pipeline else "unhealthy"
    return jsonify({"status": status})