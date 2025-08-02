import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from fraudguard.pipeline.training_pipeline import TrainingPipeline
    from fraudguard.logger import fraud_logger
    from fraudguard.exception import FraudGuardException
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main training pipeline execution"""
    try:
        fraud_logger.info("Starting FraudGuard Training Pipeline...")
        
        # Initialize and run training pipeline
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        
        fraud_logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        fraud_logger.error(f"Training pipeline failed: {e}")
        raise FraudGuardException(str(e), sys)

if __name__ == "__main__":
    main()