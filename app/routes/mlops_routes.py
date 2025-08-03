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

from fraudguard.mlops import MLModelManager, ModelMetadata, PerformanceMetrics, ExperimentTracker
from fraudguard.logger import fraud_logger
from fraudguard.exception import FraudGuardException

mlops_bp = Blueprint('mlops', __name__)

# Initialize ML Model Manager and Experiment Tracker
try:
    ml_manager = MLModelManager()
    experiment_tracker = ExperimentTracker()
    fraud_logger.info("ML Operations routes initialized")
except Exception as e:
    fraud_logger.error(f"Failed to initialize ML Manager: {e}")
    ml_manager = None
    experiment_tracker = None


@mlops_bp.route('/models')
def list_models():
    """List all registered models"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        models = ml_manager.registry.list_models()
        models_data = [
            {
                'model_id': model.model_id,
                'name': model.name,
                'algorithm': model.algorithm,
                'created_at': model.created_at.isoformat(),
                'deployment_status': model.deployment_status,
                'tags': model.tags,
                'description': model.description,
                'model_size_mb': model.model_size_mb
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


@mlops_bp.route('/models/<model_id>')
def get_model_details(model_id):
    """Get detailed information about a specific model"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        status = ml_manager.get_model_status(model_id)
        return jsonify(status)
        
    except Exception as e:
        fraud_logger.error(f"Failed to get model details for {model_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<model_id>/versions')
def list_model_versions(model_id):
    """List all versions for a specific model"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        versions = ml_manager.registry.list_model_versions(model_id)
        versions_data = [
            {
                'version_id': version.version_id,
                'version': version.version,
                'created_at': version.created_at.isoformat(),
                'is_active': version.is_active,
                'deployment_stage': version.deployment_stage,
                'performance_metrics': version.performance_metrics
            }
            for version in versions
        ]
        
        return jsonify({
            'versions': versions_data,
            'count': len(versions_data)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to list versions for {model_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<model_id>/performance')
def get_performance_history(model_id):
    """Get performance history for a model"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        limit = request.args.get('limit', 50, type=int)
        history = ml_manager.registry.get_performance_history(model_id, limit)
        
        history_data = [
            {
                'timestamp': metrics.timestamp.isoformat(),
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'auc_roc': metrics.auc_roc,
                'prediction_latency': metrics.prediction_latency,
                'throughput': metrics.throughput,
                'drift_score': metrics.drift_score,
                'error_rate': metrics.error_rate
            }
            for metrics in history
        ]
        
        return jsonify({
            'performance_history': history_data,
            'count': len(history_data)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to get performance history for {model_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<model_id>/deploy', methods=['POST'])
def deploy_model(model_id):
    """Deploy model to specified stage"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        data = request.get_json()
        stage = data.get('stage', 'production')
        
        success = ml_manager.deploy_model(model_id, stage)
        
        if success:
            return jsonify({
                'message': f'Model {model_id} deployed to {stage}',
                'model_id': model_id,
                'stage': stage
            })
        else:
            return jsonify({'error': 'Deployment failed'}), 500
            
    except Exception as e:
        fraud_logger.error(f"Failed to deploy model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<model_id>/rollback', methods=['POST'])
def rollback_model(model_id):
    """Rollback model to previous version"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        data = request.get_json() or {}
        target_version = data.get('target_version')
        
        success = ml_manager.rollback_model(model_id, target_version)
        
        if success:
            return jsonify({
                'message': f'Model {model_id} rolled back successfully',
                'model_id': model_id,
                'target_version': target_version
            })
        else:
            return jsonify({'error': 'Rollback failed'}), 500
            
    except Exception as e:
        fraud_logger.error(f"Failed to rollback model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/models/<model_id>/retrain', methods=['POST'])
def trigger_retraining(model_id):
    """Trigger model retraining"""
    try:
        if not ml_manager:
            return jsonify({'error': 'ML Manager not available'}), 500
        
        data = request.get_json() or {}
        reason = data.get('reason', 'Manual retraining request')
        
        job_id = ml_manager.trigger_retraining(model_id, reason)
        
        return jsonify({
            'message': f'Retraining triggered for model {model_id}',
            'model_id': model_id,
            'job_id': job_id,
            'reason': reason
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to trigger retraining for {model_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/monitoring/dashboard')
def monitoring_dashboard():
    """Render ML monitoring dashboard"""
    try:
        if not ml_manager:
            return render_template('error.html', 
                                 error_message='ML Manager not available'), 500
        
        # Get all models and their status
        models = ml_manager.registry.list_models()
        model_statuses = []
        
        for model in models:
            status = ml_manager.get_model_status(model.model_id)
            model_statuses.append(status)
        
        return render_template('mlops/monitoring_dashboard.html',
                             models=model_statuses,
                             total_models=len(models))
        
    except Exception as e:
        fraud_logger.error(f"Failed to render monitoring dashboard: {e}")
        return render_template('error.html', 
                             error_message=f'Dashboard error: {str(e)}'), 500


@mlops_bp.route('/monitoring/alerts')
def get_alerts():
    """Get recent performance alerts"""
    try:
        # This would typically fetch from a database or alert system
        # For now, return mock data
        alerts = [
            {
                'alert_id': 'alert_001',
                'model_id': 'model_123',
                'model_name': 'Random Forest Classifier',
                'severity': 'medium',
                'message': 'F1 Score below threshold: 0.87 < 0.90',
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
        ]
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to get alerts: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments')
def list_experiments():
    """List all experiments"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status')
        
        experiments = experiment_tracker.list_experiments(limit=limit, status=status)
        experiments_data = [
            {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'description': exp.description,
                'created_at': exp.created_at.isoformat(),
                'status': exp.status,
                'parameters': exp.parameters,
                'metrics': exp.metrics,
                'tags': exp.tags
            }
            for exp in experiments
        ]
        
        return jsonify({
            'experiments': experiments_data,
            'count': len(experiments_data)
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to list experiments: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments', methods=['POST'])
def create_experiment():
    """Create a new experiment"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        data = request.get_json()
        name = data.get('name', '')
        description = data.get('description', '')
        parameters = data.get('parameters', {})
        tags = data.get('tags', [])
        
        if not name:
            return jsonify({'error': 'Experiment name is required'}), 400
        
        experiment = experiment_tracker.create_experiment(
            name=name,
            description=description,
            parameters=parameters,
            tags=tags
        )
        
        return jsonify({
            'experiment_id': experiment.experiment_id,
            'name': experiment.name,
            'status': experiment.status,
            'created_at': experiment.created_at.isoformat()
        }), 201
        
    except Exception as e:
        fraud_logger.error(f"Failed to create experiment: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments/<experiment_id>')
def get_experiment(experiment_id):
    """Get experiment details"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        experiment = experiment_tracker.get_experiment(experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        return jsonify({
            'experiment_id': experiment.experiment_id,
            'name': experiment.name,
            'description': experiment.description,
            'created_at': experiment.created_at.isoformat(),
            'status': experiment.status,
            'parameters': experiment.parameters,
            'metrics': experiment.metrics,
            'artifacts': experiment.artifacts,
            'tags': experiment.tags
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to get experiment {experiment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments/<experiment_id>/metrics', methods=['POST'])
def log_experiment_metrics(experiment_id):
    """Log metrics for an experiment"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        data = request.get_json()
        metrics = data.get('metrics', {})
        step = data.get('step')
        
        experiment_tracker.log_metrics(experiment_id, metrics, step)
        
        return jsonify({
            'message': 'Metrics logged successfully',
            'experiment_id': experiment_id,
            'metrics': metrics
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to log metrics for {experiment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments/<experiment_id>/parameters', methods=['POST'])
def log_experiment_parameters(experiment_id):
    """Log parameters for an experiment"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        data = request.get_json()
        parameters = data.get('parameters', {})
        
        experiment_tracker.log_parameters(experiment_id, parameters)
        
        return jsonify({
            'message': 'Parameters logged successfully',
            'experiment_id': experiment_id,
            'parameters': parameters
        })
        
    except Exception as e:
        fraud_logger.error(f"Failed to log parameters for {experiment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments/compare', methods=['POST'])
def compare_experiments():
    """Compare multiple experiments"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        data = request.get_json()
        experiment_ids = data.get('experiment_ids', [])
        comparison_name = data.get('name', '')
        
        if len(experiment_ids) < 2:
            return jsonify({'error': 'At least 2 experiments required for comparison'}), 400
        
        comparison_results = experiment_tracker.compare_experiments(
            experiment_ids=experiment_ids,
            comparison_name=comparison_name
        )
        
        return jsonify(comparison_results)
        
    except Exception as e:
        fraud_logger.error(f"Failed to compare experiments: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/experiments/<experiment_id>', methods=['DELETE'])
def delete_experiment(experiment_id):
    """Delete an experiment"""
    try:
        if not experiment_tracker:
            return jsonify({'error': 'Experiment tracker not available'}), 500
        
        success = experiment_tracker.delete_experiment(experiment_id)
        
        if success:
            return jsonify({
                'message': f'Experiment {experiment_id} deleted successfully'
            })
        else:
            return jsonify({'error': 'Failed to delete experiment'}), 500
            
    except Exception as e:
        fraud_logger.error(f"Failed to delete experiment {experiment_id}: {e}")
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/health')
def health_check():
    """Health check endpoint for ML operations"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'ml_manager_available': ml_manager is not None,
            'experiment_tracker_available': experiment_tracker is not None,
            'registry_accessible': False,
            'monitor_active': False
        }
        
        if ml_manager:
            try:
                # Test registry access
                models = ml_manager.registry.list_models()
                health_status['registry_accessible'] = True
                health_status['total_models'] = len(models)
                
                # Test monitor
                health_status['monitor_active'] = ml_manager.monitor is not None
                
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['error'] = str(e)
        
        if experiment_tracker:
            try:
                # Test experiment tracker
                experiments = experiment_tracker.list_experiments(limit=1)
                health_status['experiment_tracker_accessible'] = True
                
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['experiment_error'] = str(e)
        
        if not ml_manager and not experiment_tracker:
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