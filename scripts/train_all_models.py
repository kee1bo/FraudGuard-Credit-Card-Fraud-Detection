#!/usr/bin/env python3
'''
Training Script for All Fraud Detection Models
This script trains all available models and saves them with their metrics.
'''

import sys
import os
from pathlib import Path
import argparse
import time

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from fraudguard.pipeline.training_pipeline import TrainingPipeline
    from fraudguard.logger import fraud_logger
    from fraudguard.exception import FraudGuardException
    from fraudguard.constants.constants import AVAILABLE_MODELS
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and all dependencies are installed")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--models', nargs='+', choices=AVAILABLE_MODELS + ['all'],
                       default=['all'], help='Models to train')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with reduced parameters')
    
    return parser.parse_args()

def setup_logging(verbose=False):
    """Setup logging configuration"""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def train_models(models_to_train, config_path, quick_mode=False):
    """Train specified models"""
    start_time = time.time()
    
    try:
        fraud_logger.info("=" * 60)
        fraud_logger.info("STARTING FRAUDGUARD TRAINING PIPELINE")
        fraud_logger.info("=" * 60)
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(config_path)
        
        # Modify config for quick training if requested
        if quick_mode:
            fraud_logger.info("Quick mode enabled - reducing training parameters")
            pipeline.config['models']['hyperparameter_tuning'] = False
            pipeline.config['models']['cross_validation'] = False
        
        # Filter models if specific ones requested
        if 'all' not in models_to_train:
            fraud_logger.info(f"Training specific models: {models_to_train}")
            pipeline.config['models']['specific_models'] = models_to_train
            pipeline.config['models']['train_all'] = False
        
        # Run the training pipeline
        results = pipeline.run_pipeline()
        
        # Print results summary
        print_results_summary(results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        fraud_logger.info("=" * 60)
        fraud_logger.info(f"TRAINING COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
        fraud_logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        fraud_logger.error(f"Training failed: {e}")
        raise FraudGuardException(str(e), sys)

def print_results_summary(results):
    """Print a summary of training results"""
    print("\n" + "=" * 80)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 80)
    
    if 'model_results' in results:
        model_results = results['model_results']
        
        # Sort models by ROC AUC score
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get('roc_auc_score', 0),
            reverse=True
        )
        
        print(f"{'Model':<20} {'ROC AUC':<10} {'Precision':<12} {'Recall':<10} {'F1 Score':<10}")
        print("-" * 70)
        
        for model_name, metrics in sorted_models:
            roc_auc = metrics.get('roc_auc_score', 0)
            fraud_metrics = metrics.get('classification_report', {}).get('1', {})
            precision = fraud_metrics.get('precision', 0)
            recall = fraud_metrics.get('recall', 0)
            f1_score = fraud_metrics.get('f1-score', 0)
            
            print(f"{model_name:<20} {roc_auc:<10.4f} {precision:<12.4f} {recall:<10.4f} {f1_score:<10.4f}")
        
        # Best model
        if 'best_model' in results:
            print(f"\nBest Model: {results['best_model']}")
            best_score = model_results[results['best_model']].get('roc_auc_score', 0)
            print(f"Best ROC AUC Score: {best_score:.4f}")
    
    print("\nModel artifacts saved to: artifacts/models/")
    print("Logs saved to: logs/")
    print("=" * 80)

def validate_environment():
    """Validate that the environment is set up correctly"""
    required_dirs = ['artifacts', 'logs', 'src']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            fraud_logger.warning(f"Directory {dir_name} does not exist, creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if we can import required modules
    try:
        import sklearn
        import pandas
        import numpy
        fraud_logger.info("All required packages are available")
    except ImportError as e:
        fraud_logger.error(f"Missing required package: {e}")
        return False
    
    return True

def main():
    """Main training script"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    print("FraudGuard AI - Model Training Script")
    print("====================================")
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please check your setup.")
        sys.exit(1)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        fraud_logger.warning(f"Config file {config_path} not found, using defaults")
    
    # Train models
    try:
        results = train_models(args.models, str(config_path), args.quick)
        print(f"\nTraining completed successfully!")
        print(f"Models saved to: artifacts/models/")
        print(f"To start the web application, run: python run_app.py")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()