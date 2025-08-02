#!/usr/bin/env python3
'''
Script to generate explanations for fraud detection models
'''

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from fraudguard.pipeline.prediction_pipeline import PredictionPipeline
    from fraudguard.pipeline.explanation_pipeline import ExplanationPipeline
    from fraudguard.models.model_factory import ModelFactory
    from fraudguard.utils.common import load_object, save_json
    from fraudguard.logger import fraud_logger
    from fraudguard.constants.constants import ARTIFACTS_DIR
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate model explanations')
    parser.add_argument('--models', nargs='+', 
                       help='Models to generate explanations for')
    parser.add_argument('--output', type=str, default='explanations.json',
                       help='Output file for explanations')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of sample explanations to generate')
    
    return parser.parse_args()

def generate_sample_transactions(n_samples=100):
    """Generate sample transactions for explanation"""
    np.random.seed(42)
    
    # Generate realistic transaction data
    samples = []
    for i in range(n_samples):
        # Create varied transaction patterns
        if i < n_samples // 3:
            # Normal transactions
            amount = np.random.normal(150, 50)
            time = np.random.uniform(28800, 64800)  # 8 AM to 6 PM
        elif i < 2 * n_samples // 3:
            # Suspicious transactions
            amount = np.random.normal(800, 200)
            time = np.random.uniform(0, 21600)  # Midnight to 6 AM
        else:
            # High-risk transactions
            amount = np.random.normal(2000, 500)
            time = np.random.uniform(79200, 86400)  # 10 PM to Midnight
        
        # Generate PCA features
        v_features = np.random.randn(28) * 2
        
        transaction = {
            'Time': max(0, time),
            'Amount': max(1, amount),
        }
        
        # Add V1-V28 features
        for j in range(1, 29):
            transaction[f'V{j}'] = v_features[j-1]
        
        samples.append(transaction)
    
    return samples

def generate_explanations_for_model(model_name, samples):
    """Generate explanations for a specific model"""
    try:
        fraud_logger.info(f"Generating explanations for {model_name}...")
        
        # Load the prediction pipeline
        prediction_pipeline = PredictionPipeline()
        
        if model_name not in prediction_pipeline.get_available_models():
            fraud_logger.warning(f"Model {model_name} not available")
            return None
        
        explanations = []
        
        for i, sample in enumerate(samples):
            try:
                # Get prediction with explanation
                result = prediction_pipeline.predict_single_transaction(
                    sample, 
                    model_type=model_name,
                    include_explanation=True
                )
                
                # Store explanation data
                explanation_data = {
                    'sample_id': i,
                    'transaction': sample,
                    'prediction': result['prediction'],
                    'probability_fraud': result['probability_fraud'],
                    'risk_score': result['risk_score'],
                    'explanation': result.get('explanation'),
                    'model': model_name
                }
                
                explanations.append(explanation_data)
                
                if (i + 1) % 20 == 0:
                    fraud_logger.info(f"Generated {i + 1} explanations for {model_name}")
                    
            except Exception as e:
                fraud_logger.error(f"Error generating explanation {i} for {model_name}: {e}")
                continue
        
        fraud_logger.info(f"Generated {len(explanations)} explanations for {model_name}")
        return explanations
        
    except Exception as e:
        fraud_logger.error(f"Error generating explanations for {model_name}: {e}")
        return None

def analyze_explanations(explanations):
    """Analyze generated explanations"""
    if not explanations:
        return {}
    
    analysis = {
        'total_explanations': len(explanations),
        'fraud_predictions': sum(1 for exp in explanations if exp['prediction'] == 1),
        'normal_predictions': sum(1 for exp in explanations if exp['prediction'] == 0),
        'average_risk_score': np.mean([exp['risk_score'] for exp in explanations]),
        'risk_score_distribution': {}
    }
    
    # Risk score distribution
    risk_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for low, high in risk_ranges:
        count = sum(1 for exp in explanations 
                   if low <= exp['risk_score'] < high)
        analysis['risk_score_distribution'][f'{low}-{high}%'] = count
    
    # Feature importance analysis (if available)
    if explanations[0].get('explanation'):
        feature_impacts = []
        for exp in explanations:
            if exp.get('explanation') and exp['explanation'].get('shap_values'):
                feature_impacts.append(exp['explanation']['shap_values'])
        
        if feature_impacts:
            feature_impacts = np.array(feature_impacts)
            analysis['average_feature_importance'] = {
                f'V{i+1}': float(np.mean(np.abs(feature_impacts[:, i]))) 
                for i in range(min(28, feature_impacts.shape[1]))
            }
            
            # Add Time and Amount if present
            if feature_impacts.shape[1] > 28:
                analysis['average_feature_importance']['Time'] = float(np.mean(np.abs(feature_impacts[:, 28])))
            if feature_impacts.shape[1] > 29:
                analysis['average_feature_importance']['Amount'] = float(np.mean(np.abs(feature_impacts[:, 29])))
    
    return analysis

def main():
    """Main explanation generation script"""
    args = parse_arguments()
    
    print("FraudGuard AI - Explanation Generation Script")
    print("===========================================")
    
    # Generate sample transactions
    fraud_logger.info(f"Generating {args.samples} sample transactions...")
    samples = generate_sample_transactions(args.samples)
    
    # Determine which models to process
    try:
        prediction_pipeline = PredictionPipeline()
        available_models = prediction_pipeline.get_available_models()
        
        if args.models:
            models_to_process = [m for m in args.models if m in available_models]
        else:
            models_to_process = available_models
            
        if not models_to_process:
            print("No valid models specified or available")
            sys.exit(1)
            
        fraud_logger.info(f"Processing models: {models_to_process}")
        
    except Exception as e:
        fraud_logger.error(f"Error initializing prediction pipeline: {e}")
        sys.exit(1)
    
    # Generate explanations for each model
    all_explanations = {}
    
    for model_name in models_to_process:
        explanations = generate_explanations_for_model(model_name, samples)
        if explanations:
            all_explanations[model_name] = {
                'explanations': explanations,
                'analysis': analyze_explanations(explanations)
            }
    
    # Save results
    output_path = Path(args.output)
    save_json(all_explanations, str(output_path))
    
    # Print summary
    print("\nExplanation Generation Summary:")
    print("=" * 50)
    for model_name, data in all_explanations.items():
        analysis = data['analysis']
        print(f"\nModel: {model_name}")
        print(f"  Total explanations: {analysis['total_explanations']}")
        print(f"  Fraud predictions: {analysis['fraud_predictions']}")
        print(f"  Normal predictions: {analysis['normal_predictions']}")
        print(f"  Average risk score: {analysis['average_risk_score']:.2f}%")
    
    print(f"\nExplanations saved to: {output_path}")

if __name__ == "__main__":
    main()