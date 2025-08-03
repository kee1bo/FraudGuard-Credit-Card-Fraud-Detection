import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from flask import Blueprint, render_template, request, jsonify, flash
from fraudguard.pipeline.pipeline_manager import pipeline_manager

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/', methods=['GET', 'POST'])
def predict():
    """Simplified prediction page"""
    if request.method == 'GET':
        available_models = pipeline_manager.get_available_models()
        return render_template('prediction_simple.html', models=available_models)
    
    # Handle POST request (for backward compatibility)
    try:
        prediction_pipeline = pipeline_manager.get_pipeline()
        if not prediction_pipeline:
            raise Exception("Prediction pipeline not available")
            
        # Get form data
        transaction_data = {
            'Time': float(request.form.get('time', 0)),
            'Amount': float(request.form.get('amount', 0)),
        }
        
        # Add V1-V28 features (zeros for simplified interface)
        for i in range(1, 29):
            transaction_data[f'V{i}'] = float(request.form.get(f'v{i}', 0))
        
        model_type = request.form.get('model_type', 'random_forest')
        include_explanation = request.form.get('include_explanation') == 'on'
        
        # Make prediction
        result = prediction_pipeline.predict_single_transaction(
            transaction_data, 
            model_type=model_type,
            include_explanation=include_explanation
        )
        
        return render_template('results.html', result=result)
        
    except Exception as e:
        flash(f"Prediction error: {str(e)}", 'error')
        available_models = pipeline_manager.get_available_models()
        return render_template('prediction_simple.html', models=available_models)

@prediction_bp.route('/api', methods=['POST'])
def predict_api():
    """API endpoint for simplified predictions"""
    try:
        prediction_pipeline = pipeline_manager.get_pipeline()
        if not prediction_pipeline:
            return jsonify({'error': 'Prediction pipeline not available'}), 500
            
        # Get JSON data
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        features = data['features']
        model_type = data.get('model_type', 'random_forest')
        
        # Make prediction
        result = prediction_pipeline.predict_single_transaction(
            features, 
            model_type=model_type,
            include_explanation=False
        )
        
        # Format response
        response = {
            'prediction': int(result.get('prediction', 0)),
            'fraud_probability': float(result.get('fraud_probability', 0)),
            'model_used': result.get('model_type', model_type),
            'confidence': float(1 - result.get('fraud_probability', 0)),
            'risk_level': 'HIGH' if result.get('fraud_probability', 0) > 0.5 else 'LOW'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/advanced')
def predict_advanced():
    """Advanced prediction page with all features (for testing)"""
    available_models = pipeline_manager.get_available_models()
    return render_template('prediction.html', models=available_models)

@prediction_bp.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction page"""
    if request.method == 'GET':
        available_models = pipeline_manager.get_available_models()
        return render_template('batch_prediction.html', models=available_models)
    
    # Handle batch predictions
    try:
        # Implementation for batch predictions
        flash('Batch prediction feature coming soon!', 'info')
        return render_template('batch_prediction.html', models=available_models)
        
    except Exception as e:
        flash(f"Batch prediction error: {str(e)}", 'error')
        available_models = pipeline_manager.get_available_models()
        return render_template('batch_prediction.html', models=available_models)
