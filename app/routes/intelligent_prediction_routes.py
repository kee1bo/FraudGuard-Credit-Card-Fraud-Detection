"""
Intelligent Prediction Routes
Flask routes for the intelligent feature mapping prediction system.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from flask import Blueprint, render_template, request, jsonify, flash
import traceback

try:
    from fraudguard.pipeline.intelligent_prediction_pipeline import IntelligentPredictionPipeline
    from fraudguard.entity.feature_mapping_entity import (
        MerchantCategory, LocationRisk, SpendingPattern
    )
    INTELLIGENT_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Intelligent prediction pipeline not available: {e}")
    INTELLIGENT_PIPELINE_AVAILABLE = False

intelligent_bp = Blueprint('intelligent', __name__)

# Global pipeline instance
intelligent_pipeline = None

def get_intelligent_pipeline():
    """Get or create intelligent prediction pipeline"""
    global intelligent_pipeline
    
    if not INTELLIGENT_PIPELINE_AVAILABLE:
        return None
    
    if intelligent_pipeline is None:
        try:
            intelligent_pipeline = IntelligentPredictionPipeline()
        except Exception as e:
            print(f"Error initializing intelligent pipeline: {e}")
            return None
    
    return intelligent_pipeline

@intelligent_bp.route('/', methods=['GET'])
def intelligent_prediction_form():
    """Display the intelligent prediction form"""
    return render_template('intelligent_prediction.html')

@intelligent_bp.route('/predict', methods=['POST'])
def predict_intelligent():
    """Handle intelligent prediction requests"""
    try:
        pipeline = get_intelligent_pipeline()
        if not pipeline:
            return jsonify({
                'error': 'Intelligent prediction pipeline not available. Please ensure all components are properly installed and trained.'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'transaction_amount', 'merchant_category', 'hour_of_day', 
            'day_of_week', 'location_risk', 'spending_pattern'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Make prediction
        result = pipeline.predict_intelligent(
            input_data=data,
            include_explanation=True
        )
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"Error in intelligent prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@intelligent_bp.route('/validate', methods=['POST'])
def validate_input():
    """Validate input data and provide suggestions"""
    try:
        pipeline = get_intelligent_pipeline()
        if not pipeline:
            return jsonify({
                'error': 'Intelligent prediction pipeline not available'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input
        validation_result = pipeline.validate_input_data(data)
        
        return jsonify(validation_result)
        
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        print(f"Error in input validation: {e}")
        return jsonify({'error': error_msg}), 500

@intelligent_bp.route('/suggestions', methods=['POST'])
def get_suggestions():
    """Get intelligent suggestions based on partial input"""
    try:
        pipeline = get_intelligent_pipeline()
        if not pipeline:
            return jsonify({
                'error': 'Intelligent prediction pipeline not available'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        suggestions = {}
        
        # Get merchant suggestions based on amount
        if 'transaction_amount' in data:
            try:
                amount = float(data['transaction_amount'])
                if amount > 0:
                    merchant_suggestions = pipeline.input_validator.get_merchant_suggestions(amount)
                    suggestions['merchant_categories'] = [m.value for m in merchant_suggestions]
            except (ValueError, TypeError):
                pass
        
        # Get time suggestions based on merchant
        if 'merchant_category' in data:
            try:
                merchant = MerchantCategory(data['merchant_category'])
                time_suggestions = pipeline.input_validator.get_time_suggestions(merchant)
                suggestions['time_context'] = time_suggestions
            except ValueError:
                pass
        
        # Get risk suggestions
        if 'merchant_category' in data and 'transaction_amount' in data:
            try:
                merchant = MerchantCategory(data['merchant_category'])
                amount = float(data['transaction_amount'])
                risk_suggestions = pipeline.input_validator.get_risk_level_suggestions(merchant, amount)
                suggestions.update(risk_suggestions)
            except (ValueError, TypeError):
                pass
        
        return jsonify({
            'suggestions': suggestions,
            'status': 'success'
        })
        
    except Exception as e:
        error_msg = f"Suggestions error: {str(e)}"
        print(f"Error getting suggestions: {e}")
        return jsonify({'error': error_msg}), 500

@intelligent_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the intelligent prediction system"""
    try:
        pipeline = get_intelligent_pipeline()
        if not pipeline:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Intelligent prediction pipeline not available'
            }), 503
        
        health_status = pipeline.health_check()
        
        status_code = 200
        if health_status['status'] == 'unhealthy':
            status_code = 503
        elif health_status['status'] == 'degraded':
            status_code = 206
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@intelligent_bp.route('/stats', methods=['GET'])
def get_statistics():
    """Get pipeline statistics and information"""
    try:
        pipeline = get_intelligent_pipeline()
        if not pipeline:
            return jsonify({
                'error': 'Intelligent prediction pipeline not available'
            }), 500
        
        stats = pipeline.get_mapping_statistics()
        
        # Add enum information for frontend
        stats['enums'] = {
            'merchant_categories': [category.value for category in MerchantCategory],
            'location_risks': [risk.value for risk in LocationRisk],
            'spending_patterns': [pattern.value for pattern in SpendingPattern]
        }
        
        return jsonify(stats)
        
    except Exception as e:
        error_msg = f"Statistics error: {str(e)}"
        print(f"Error getting statistics: {e}")
        return jsonify({'error': error_msg}), 500

@intelligent_bp.route('/examples', methods=['GET'])
def get_examples():
    """Get example transaction data for testing"""
    examples = {
        'normal_grocery': {
            'transaction_amount': 45.67,
            'merchant_category': 'grocery',
            'hour_of_day': 14,
            'day_of_week': 2,
            'location_risk': 'normal',
            'spending_pattern': 'typical',
            'description': 'Normal grocery shopping on Wednesday afternoon'
        },
        'evening_restaurant': {
            'transaction_amount': 85.30,
            'merchant_category': 'restaurant',
            'hour_of_day': 19,
            'day_of_week': 5,
            'location_risk': 'normal',
            'spending_pattern': 'typical',
            'description': 'Dinner at restaurant on Friday evening'
        },
        'suspicious_online': {
            'transaction_amount': 1500.00,
            'merchant_category': 'online',
            'hour_of_day': 2,
            'day_of_week': 1,
            'location_risk': 'slightly_unusual',
            'spending_pattern': 'suspicious',
            'description': 'High-value online purchase at 2 AM'
        },
        'foreign_travel': {
            'transaction_amount': 2200.00,
            'merchant_category': 'travel',
            'hour_of_day': 10,
            'day_of_week': 0,
            'location_risk': 'foreign',
            'spending_pattern': 'much_higher',
            'description': 'Travel booking in foreign country'
        },
        'late_night_atm': {
            'transaction_amount': 300.00,
            'merchant_category': 'atm',
            'hour_of_day': 23,
            'day_of_week': 6,
            'location_risk': 'slightly_unusual',
            'spending_pattern': 'slightly_higher',
            'description': 'ATM withdrawal late at night on Sunday'
        },
        'high_value_department': {
            'transaction_amount': 800.00,
            'merchant_category': 'department_store',
            'hour_of_day': 15,
            'day_of_week': 5,
            'location_risk': 'normal',
            'spending_pattern': 'much_higher',
            'description': 'High-value department store purchase'
        }
    }
    
    return jsonify(examples)

# Error handlers
@intelligent_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@intelligent_bp.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({'error': 'Method not allowed'}), 405

@intelligent_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500