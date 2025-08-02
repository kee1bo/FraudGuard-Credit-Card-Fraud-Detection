import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from app.main import create_app
    from fraudguard.constants.constants import FLASK_CONFIG
    from fraudguard.logger import fraud_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Run the Flask application"""
    try:
        fraud_logger.info("Starting FraudGuard Web Application...")
        
        # Check if artifacts exist
        artifacts_path = Path("artifacts")
        if not artifacts_path.exists():
            fraud_logger.warning("Artifacts directory not found. Please run training first:")
            fraud_logger.warning("python main.py")
            
        app = create_app()
        app.run(
            host=FLASK_CONFIG['HOST'],
            port=FLASK_CONFIG['PORT'],
            debug=FLASK_CONFIG['DEBUG']
        )
        
    except Exception as e:
        fraud_logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()