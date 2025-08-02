import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from flask import Blueprint, render_template, request, jsonify, flash
from fraudguard.pipeline.prediction_pipeline import PredictionPipeline

prediction_bp = Blueprint('prediction', __name__)

# Initialize prediction pipeline
try:
    prediction_pipeline = PredictionPipeline()
    available_models = prediction_pipeline.get_available_models()
except Exception as e:
    print(f"Warning: Could not initialize prediction pipeline: {e}")
    prediction_pipeline = None
    available_models = []

@prediction_bp.route('/', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('prediction.html', models=available_models)
    
    # Handle POST request
    try:
        if not prediction_pipeline:
            raise Exception("Prediction pipeline not available")
            
        # Get form data
        transaction_data = {
            'Time': float(request.form.get('time', 0)),
            'Amount': float(request.form.get('amount', 0)),
        }
        
        # Add V1-V28 features
        for i in range(1, 29):
            transaction_data[f'V{i}'] = float(request.form.get(f'v{i}', 0))
        
        model_type = request.form.get('model_type', 'xgboost')
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
        return render_template('prediction.html', models=available_models)

@prediction_bp.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction page"""
    if request.method == 'GET':
        return render_template('batch_prediction.html', models=available_models)
    
    # Handle batch predictions
    try:
        # Implementation for batch predictions
        flash('Batch prediction feature coming soon!', 'info')
        return render_template('batch_prediction.html', models=available_models)
        
    except Exception as e:
        flash(f"Batch prediction error: {str(e)}", 'error')
        return render_template('batch_prediction.html', models=available_models)
