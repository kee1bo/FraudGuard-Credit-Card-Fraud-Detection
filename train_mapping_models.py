#!/usr/bin/env python3
"""
Train Feature Mapping Models
Script to train the feature mapping models for the intelligent prediction system.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fraudguard.pipeline.mapping_training_pipeline import train_mapping_models
from fraudguard.logger import fraud_logger


def main():
    """Main training function"""
    fraud_logger.info("Starting feature mapping model training...")
    
    # Check if ULB dataset exists
    dataset_path = "data/creditcard.csv"
    dataset_exists = Path(dataset_path).exists()
    
    if dataset_exists:
        fraud_logger.info(f"Found ULB dataset at {dataset_path}")
        # Train with real data (using smaller sample for faster training)
        result = train_mapping_models(
            dataset_path=dataset_path,
            sample_size=5000,  # Use 5000 samples for faster training
            create_demo=False
        )
    else:
        fraud_logger.warning(f"ULB dataset not found at {dataset_path}")
        fraud_logger.info("Creating demo mappers with synthetic data for testing...")
        # Create demo mappers for testing
        result = train_mapping_models(
            dataset_path=dataset_path,
            create_demo=True
        )
    
    # Print results
    if result['success']:
        fraud_logger.info("✅ Training completed successfully!")
        fraud_logger.info(f"Trained models: {result['trained_models']}")
        
        if result.get('type') == 'demo':
            fraud_logger.info("📝 Note: Demo mappers created with synthetic data")
            fraud_logger.info("   For production use, please provide the ULB dataset at data/creditcard.csv")
        else:
            fraud_logger.info(f"Training time: {result.get('training_time', 0):.2f} seconds")
            
        fraud_logger.info("\n🚀 You can now use the intelligent prediction interface!")
        fraud_logger.info("   Navigate to http://localhost:5000/intelligent/ to test the system")
        
    else:
        fraud_logger.error("❌ Training failed!")
        fraud_logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
        if 'suggestion' in result:
            fraud_logger.info(f"💡 Suggestion: {result['suggestion']}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()