import sys
import os
import json
from pathlib import Path
from flask import Flask

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def create_app():
    """Flask application factory pattern for FraudGuard application"""
    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.jinja_env.filters['tojson'] = json.dumps
    
    from app.routes.main_routes import main_bp
    from app.routes.prediction_routes import prediction_bp
    from app.routes.dashboard_routes import dashboard_bp
    from app.routes.api_routes import api_bp
    from app.routes.mlops_routes import mlops_bp
    from app.routes.documentation_routes import documentation_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp, url_prefix='/predict')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(mlops_bp, url_prefix='/mlops')
    app.register_blueprint(documentation_bp, url_prefix='/docs')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)