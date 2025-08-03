"""
ML Operations Routes for Professional Model Management
"""

from flask import Blueprint, jsonify, request, render_template
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from fraudguard.mlops import MappingModelRegistry, get_mapping_model_registry, DeploymentManager, get_deployment_manager
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException

mlops_bp = Blueprint('mlops', __name__)

# Initialize ML Model Registry and Deployment Manager
try:
    model_registry = get_mapping_model_registry()
    deployment_manager = get_deployment_manager()
    fraud_logger.info("ML Operations routes initialized")
except Exception as e:
    fraud_logger.error(f"Failed to initialize ML Operations: {e}")
    model_registry = None
    deployment_manager = None


@mlops_bp.route('/models')
def list_models():
    """List all registered mapping models"""
    try:
        if not model_registry:
            return jsonify({'error': 'Model Registry not available'}), 500
        
        models = model_registry.get_model_versions()
        models_data = [
            {
                'version_id': model.version_id,
                'model_type': model.model_type,
                'version_number': model.version_number,
                'created_at': model.created_at,
                'is_active': model.is_active,
                'is_champion': model.is_champion,
                'description': model.description,
                'performance_metrics': model.performance_metrics,
                'deployment_date': model.deployment_date
            }
            for model in models
        ]
        
        return jsonify({
            'models': models_data,
            'count': len(models_data)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to list models: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<version_id>')
def get_model_details(version_id):
    """Get detailed information about a specific model version"""
    try:
        if not model_registry:
            return jsonify({'error': 'Model Registry not available'}), 500
        
        if version_id not in model_registry.model_versions:
            return jsonify({'error': 'Model version not found'}), 404
        
        model = model_registry.model_versions[version_id]
        model_data = {
            'version_id': model.version_id,
            'model_type': model.model_type,
            'version_number': model.version_number,
            'created_at': model.created_at,
            'created_by': model.created_by,
            'description': model.description,
            'performance_metrics': model.performance_metrics,
            'model_path': model.model_path,
            'is_active': model.is_active,
            'is_champion': model.is_champion,
            'deployment_date': model.deployment_date,
            'rollback_version': model.rollback_version
        }
        
        return jsonify(model_data)
        
    except Exception as e:
        fraud_logger.error(f"Failed to get model details for {version_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/types/<model_type>')
def list_model_type_versions(model_type):
    """List all versions for a specific model type"""
    try:
        if not model_registry:
            return jsonify({'error': 'Model Registry not available'}), 500
        
        versions = model_registry.get_model_versions(model_type)
        versions_data = [
            {
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_at': version.created_at,
                'is_active': version.is_active,
                'is_champion': version.is_champion,
                'performance_metrics': version.performance_metrics,
                'deployment_date': version.deployment_date
            }
            for version in versions
        ]
        
        return jsonify({
            'model_type': model_type,
            'versions': versions_data,
            'count': len(versions_data)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to list versions for {model_type}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/deployment/status')
def get_deployment_status():
    """Get current deployment status"""
    try:
        if not deployment_manager:
            return jsonify({'error': 'Deployment Manager not available'}), 500
        
        status = deployment_manager.get_deployment_status()
        return jsonify(status)
        
    except Exception as e:
        fraud_logger.error(f"Failed to get deployment status: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<version_id>/deploy', methods=['POST'])
def deploy_model(version_id):
    """Deploy model version"""
    try:
        if not deployment_manager:
            return jsonify({'error': 'Deployment Manager not available'}), 500
        
        data = request.get_json() or {}
        make_champion = data.get('make_champion', False)
        run_health_checks = data.get('run_health_checks', True)
        
        result = deployment_manager.deploy_model_with_checks(
            version_id=version_id,
            make_champion=make_champion,
            run_health_checks=run_health_checks
        )
        
        if result['success']:
            return jsonify({
                'message': f'Model {version_id} deployed successfully',
                'version_id': version_id,
                'deployment_result': result
            })
        else:
            return jsonify({
                'error': 'Deployment failed',
                'details': result
            }), 500
            
    except Exception as e:
        fraud_logger.error(f"Failed to deploy model {version_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/deployment/history')
def get_deployment_history():
    """Get deployment history"""
    try:
        if not deployment_manager:
            return jsonify({'error': 'Deployment Manager not available'}), 500
        
        limit = request.args.get('limit', 10, type=int)
        history = deployment_manager.get_deployment_history(limit)
        
        return jsonify({
            'deployment_history': history,
            'count': len(history)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to get deployment history: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/performance/<model_type>')
def get_model_performance_comparison(model_type):
    """Get performance comparison for model type"""
    try:
        if not model_registry:
            return jsonify({'error': 'Model Registry not available'}), 500
        
        comparison = model_registry.get_model_performance_comparison(model_type)
        return jsonify(comparison)
        
    except Exception as e:
        fraud_logger.error(f"Failed to get performance comparison for {model_type}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/monitoring/dashboard')
def monitoring_dashboard():
    """Render ML monitoring dashboard"""
    try:
        if not model_registry:
            return render_template('error.html', 
                                 error_message='Model Registry not available'), 500
        
        # Get deployment status
        deployment_status = deployment_manager.get_deployment_status() if deployment_manager else {}
        
        # Get all model versions
        models = model_registry.get_model_versions()
        
        # Group by model type
        model_types = {}
        for model in models:
            if model.model_type not in model_types:
                model_types[model.model_type] = []
            model_types[model.model_type].append(model)
        
        return render_template('mlops/monitoring_dashboard.html',
                             model_types=model_types,
                             deployment_status=deployment_status,
                             total_models=len(models))
        
    except Exception as e:
        fraud_logger.error(f"Failed to render monitoring dashboard: {e}")
        return render_template('error.html', 
                             error_message=f'Dashboard error: {str(e)}'), 500


@mlops_bp.route('/monitoring/alerts')
def get_alerts():
    """Get recent performance alerts"""
    try:
        # Simple alert system based on model performance
        alerts = []
        
        if model_registry:
            models = model_registry.get_model_versions()
            for model in models:
                if model.is_champion and model.performance_metrics:
                    # Check for performance issues
                    mse = model.performance_metrics.get('mse', 0)
                    correlation = model.performance_metrics.get('avg_correlation', 1)
                    
                    if mse > 0.5:
                        alerts.append({
                            'alert_id': f'alert_mse_{model.version_id}',
                            'model_id': model.version_id,
                            'model_type': model.model_type,
                            'severity': 'medium',
                            'message': f'MSE above threshold: {mse:.3f} > 0.5',
                            'timestamp': datetime.now().isoformat(),
                            'status': 'active'
                        })
                    
                    if correlation < 0.7:
                        alerts.append({
                            'alert_id': f'alert_corr_{model.version_id}',
                            'model_id': model.version_id,
                            'model_type': model.model_type,
                            'severity': 'high',
                            'message': f'Correlation below threshold: {correlation:.3f} < 0.7',
                            'timestamp': datetime.now().isoformat(),
                            'status': 'active'
                        })
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to get alerts: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/health')
def health_check():
    """Health check endpoint for ML operations"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_registry_available': model_registry is not None,
            'deployment_manager_available': deployment_manager is not None,
            'registry_accessible': False,
            'total_models': 0
        }
        
        if model_registry:
            try:
                # Test registry access
                models = model_registry.get_model_versions()
                health_status['registry_accessible'] = True
                health_status['total_models'] = len(models)
                
                # Count active models
                active_models = [m for m in models if m.is_active]
                health_status['active_models'] = len(active_models)
                
                # Count champion models
                champion_models = [m for m in models if m.is_champion]
                health_status['champion_models'] = len(champion_models)
                
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['error'] = str(e)
        
        if deployment_manager:
            try:
                # Test deployment manager
                deployment_status = deployment_manager.get_deployment_status()
                health_status['deployment_accessible'] = True
                
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['deployment_error'] = str(e)
        
        if not model_registry and not deployment_manager:
            health_status['status'] = 'unhealthy'
            health_status['error'] = 'ML services not initialized'
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        fraud_logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


# Error handlers
@mlops_bp.errorhandler(FraudGuardException)
def handle_fraudguard_exception(e):
    """Handle FraudGuard specific exceptions"""
    fraud_logger.error(f"FraudGuard exception in MLOps: {e}")
    return jsonify({'error': str(e)}), 400


@mlops_bp.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404


@mlops_bp.errorhandler(500)
def handle_internal_error(e):
    """Handle 500 errors"""
    fraud_logger.error(f"Internal error in MLOps: {e}")
    return jsonify({'error': 'Internal server error'}), 500