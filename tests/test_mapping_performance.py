"""
Performance Benchmarking for Feature Mapping Models
Specific performance tests for individual mapping components and models.
"""

import pytest
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
import json
import psutil
import gc
from memory_profiler import profile

# Import mapping components
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper
from fraudguard.models.ensemble_mapper import EnsembleMapper
from fraudguard.components.feature_assembler import FeatureVectorAssembler
from fraudguard.components.confidence_scorer import ConfidenceScorer
from fraudguard.entity.feature_mapping_entity import UserTransactionInput
from fraudguard.logger import fraud_logger

# Try to import neural network mapper
try:
    from fraudguard.models.neural_network_mapper import NeuralNetworkMapper
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NeuralNetworkMapper = None
    NEURAL_NETWORK_AVAILABLE = False


class MappingPerformanceBenchmark:
    """Performance benchmarking suite for feature mapping models"""
    
    def __init__(self):
        self.benchmark_results = {
            'model_performance': {},
            'memory_usage': {},
            'throughput_tests': {},
            'scalability_tests': {}
        }
        
        # Test data sizes
        self.test_sizes = [1, 10, 100, 1000, 5000]
        
        # Generate test data
        self.test_data = self._generate_benchmark_data()
        
    def _generate_benchmark_data(self) -> Dict[str, np.ndarray]:
        """Generate benchmark data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate interpretable features (5 features)
        max_size = max(self.test_sizes)
        X_data = np.random.rand(max_size, 5)
        
        # Simulate realistic ranges
        X_data[:, 0] *= 1000  # transaction_amount: 0-1000
        X_data[:, 1] = np.random.choice([0, 1, 2, 3, 4], max_size)  # merchant_category (encoded)
        X_data[:, 2] *= 1.0   # location_risk_score: 0-1
        X_data[:, 3] *= 168   # time_since_last_transaction: 0-168 hours
        X_data[:, 4] *= 1.0   # spending_pattern_score: 0-1
        
        # Generate corresponding PCA features (28 features)
        y_data = np.random.randn(max_size, 28)
        
        return {
            'X': X_data,
            'y': y_data
        }
    
    def benchmark_model_training_performance(self) -> Dict[str, Any]:
        """Benchmark training performance for different models"""
        training_results = {
            'models_tested': [],
            'training_times': {},
            'memory_usage': {},
            'model_sizes': {}
        }
        
        fraud_logger.info("Benchmarking model training performance...")
        
        # Test different data sizes
        for size in [100, 500, 1000]:
            X_train = self.test_data['X'][:size]
            y_train = self.test_data['y'][:size]
            
            # Test Random Forest
            training_results = self._benchmark_single_model_training(
                "RandomForest", RandomForestMapper, X_train, y_train, training_results, size
            )
            
            # Test XGBoost
            training_results = self._benchmark_single_model_training(
                "XGBoost", XGBoostMapper, X_train, y_train, training_results, size
            )
            
            # Test Neural Network if available
            if NEURAL_NETWORK_AVAILABLE:
                training_results = self._benchmark_single_model_training(
                    "NeuralNetwork", NeuralNetworkMapper, X_train, y_train, training_results, size
                )
        
        self.benchmark_results['model_performance']['training'] = training_results
        return training_results
    
    def _benchmark_single_model_training(self, model_name: str, model_class, X_train, y_train, results, size):
        """Benchmark training for a single model"""
        try:
            fraud_logger.info(f"Training {model_name} with {size} samples...")
            
            # Monitor memory before training
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and train model
            start_time = time.time()
            
            if model_name == "NeuralNetwork":
                model = model_class(epochs=10, hidden_layers=[32])  # Reduced for benchmarking
            else:
                model = model_class()
            
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Monitor memory after training
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Store results
            key = f"{model_name}_{size}"
            if model_name not in results['training_times']:
                results['training_times'][model_name] = {}
                results['memory_usage'][model_name] = {}
            
            results['training_times'][model_name][size] = training_time
            results['memory_usage'][model_name][size] = memory_used
            
            if model_name not in results['models_tested']:
                results['models_tested'].append(model_name)
            
            fraud_logger.info(f"{model_name} training completed: {training_time:.2f}s, {memory_used:.1f}MB")
            
            # Clean up
            del model
            gc.collect()
            
        except Exception as e:
            fraud_logger.error(f"Error benchmarking {model_name}: {e}")
        
        return results
    
    def benchmark_prediction_performance(self) -> Dict[str, Any]:
        """Benchmark prediction performance for trained models"""
        prediction_results = {
            'models_tested': [],
            'prediction_times': {},
            'throughput': {},
            'batch_performance': {}
        }
        
        fraud_logger.info("Benchmarking prediction performance...")
        
        # Create and train models for testing
        X_train = self.test_data['X'][:1000]
        y_train = self.test_data['y'][:1000]
        
        models = {}
        
        # Train Random Forest
        try:
            rf_model = RandomForestMapper(n_estimators=50)  # Reduced for benchmarking
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model
        except Exception as e:
            fraud_logger.error(f"Failed to train RandomForest for benchmarking: {e}")
        
        # Train XGBoost
        try:
            xgb_model = XGBoostMapper(n_estimators=50)
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
        except Exception as e:
            fraud_logger.error(f"Failed to train XGBoost for benchmarking: {e}")
        
        # Test prediction performance
        for model_name, model in models.items():
            prediction_results = self._benchmark_single_model_prediction(
                model_name, model, prediction_results
            )
        
        self.benchmark_results['model_performance']['prediction'] = prediction_results
        return prediction_results
    
    def _benchmark_single_model_prediction(self, model_name: str, model, results):
        """Benchmark prediction for a single model"""
        try:
            results['models_tested'].append(model_name)
            results['prediction_times'][model_name] = {}
            results['throughput'][model_name] = {}
            results['batch_performance'][model_name] = {}
            
            # Test different batch sizes
            for batch_size in [1, 10, 100, 1000]:
                if batch_size > len(self.test_data['X']):\n                    continue
                
                X_test = self.test_data['X'][:batch_size]
                
                # Warm up
                _ = model.predict(X_test[:1])
                
                # Benchmark prediction time
                start_time = time.time()
                predictions = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                # Calculate throughput (predictions per second)
                throughput = batch_size / prediction_time if prediction_time > 0 else 0
                
                # Store results
                results['prediction_times'][model_name][batch_size] = prediction_time
                results['throughput'][model_name][batch_size] = throughput
                results['batch_performance'][model_name][batch_size] = {
                    'time_per_prediction_ms': (prediction_time / batch_size) * 1000,
                    'predictions_per_second': throughput
                }
                
                fraud_logger.info(f"{model_name} batch {batch_size}: {prediction_time:.4f}s, {throughput:.1f} pred/s")
        
        except Exception as e:
            fraud_logger.error(f"Error benchmarking {model_name} prediction: {e}")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        memory_results = {
            'baseline_memory': 0,
            'model_memory_usage': {},
            'prediction_memory_usage': {},
            'memory_leaks_detected': False
        }
        
        fraud_logger.info("Benchmarking memory usage...")
        
        process = psutil.Process()
        memory_results['baseline_memory'] = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory usage for model loading and prediction
        try:
            # Test Random Forest memory usage
            memory_before = process.memory_info().rss / 1024 / 1024
            
            rf_model = RandomForestMapper(n_estimators=100)
            X_train = self.test_data['X'][:1000]
            y_train = self.test_data['y'][:1000]
            rf_model.fit(X_train, y_train)
            
            memory_after_training = process.memory_info().rss / 1024 / 1024
            
            # Make predictions
            X_test = self.test_data['X'][:100]
            predictions = rf_model.predict(X_test)
            
            memory_after_prediction = process.memory_info().rss / 1024 / 1024
            
            memory_results['model_memory_usage']['RandomForest'] = {
                'training_memory_mb': memory_after_training - memory_before,
                'prediction_memory_mb': memory_after_prediction - memory_after_training,
                'total_memory_mb': memory_after_prediction - memory_before
            }
            
            # Clean up and check for memory leaks
            del rf_model, predictions
            gc.collect()
            
            memory_after_cleanup = process.memory_info().rss / 1024 / 1024
            memory_leak = memory_after_cleanup - memory_before
            
            if memory_leak > 10:  # More than 10MB not released
                memory_results['memory_leaks_detected'] = True
                fraud_logger.warning(f"Potential memory leak detected: {memory_leak:.1f}MB not released")
            
        except Exception as e:
            fraud_logger.error(f"Error in memory benchmarking: {e}")
        
        self.benchmark_results['memory_usage'] = memory_results
        return memory_results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with increasing data sizes"""
        scalability_results = {
            'data_sizes_tested': self.test_sizes,
            'scaling_performance': {},
            'performance_degradation': {}
        }
        
        fraud_logger.info("Benchmarking scalability...")
        
        # Create a simple model for scalability testing
        try:
            model = RandomForestMapper(n_estimators=50)
            
            # Train on medium dataset
            X_train = self.test_data['X'][:1000]
            y_train = self.test_data['y'][:1000]
            model.fit(X_train, y_train)
            
            # Test prediction performance at different scales
            baseline_time = None
            
            for size in self.test_sizes:
                if size > len(self.test_data['X']):
                    continue
                
                X_test = self.test_data['X'][:size]
                
                # Measure prediction time
                start_time = time.time()
                predictions = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                time_per_prediction = prediction_time / size if size > 0 else 0
                
                if baseline_time is None:
                    baseline_time = time_per_prediction
                
                degradation_factor = time_per_prediction / baseline_time if baseline_time > 0 else 1
                
                scalability_results['scaling_performance'][size] = {
                    'total_time_s': prediction_time,
                    'time_per_prediction_ms': time_per_prediction * 1000,
                    'throughput_per_second': 1 / time_per_prediction if time_per_prediction > 0 else 0
                }
                
                scalability_results['performance_degradation'][size] = degradation_factor
                
                fraud_logger.info(f"Size {size}: {time_per_prediction*1000:.2f}ms per prediction, degradation: {degradation_factor:.2f}x")
        
        except Exception as e:
            fraud_logger.error(f"Error in scalability benchmarking: {e}")
        
        self.benchmark_results['scalability_tests'] = scalability_results
        return scalability_results
    
    def benchmark_feature_assembler_performance(self) -> Dict[str, Any]:
        """Benchmark performance of feature vector assembly"""
        assembler_results = {
            'assembly_times': {},
            'validation_times': {},
            'total_pipeline_times': {}
        }
        
        fraud_logger.info("Benchmarking feature assembler performance...")
        
        try:
            assembler = FeatureVectorAssembler()
            
            # Test different batch sizes
            for batch_size in [1, 10, 100, 1000]:
                if batch_size > len(self.test_data['X']):
                    continue
                
                # Create test inputs
                test_inputs = []
                for i in range(batch_size):
                    user_input = UserTransactionInput(
                        transaction_amount=float(self.test_data['X'][i, 0]),
                        merchant_category="grocery_store",
                        location_risk_score=float(self.test_data['X'][i, 2]),
                        time_since_last_transaction=float(self.test_data['X'][i, 3]),
                        spending_pattern_score=float(self.test_data['X'][i, 4])
                    )
                    test_inputs.append(user_input)
                
                # Benchmark assembly time
                mapped_features = self.test_data['y'][:batch_size]
                
                start_time = time.time()
                for i, user_input in enumerate(test_inputs):
                    feature_vector = assembler.assemble_feature_vector(
                        user_input, mapped_features[i]
                    )
                assembly_time = time.time() - start_time
                
                assembler_results['assembly_times'][batch_size] = assembly_time
                assembler_results['total_pipeline_times'][batch_size] = assembly_time
                
                fraud_logger.info(f"Feature assembly batch {batch_size}: {assembly_time:.4f}s")
        
        except Exception as e:
            fraud_logger.error(f"Error benchmarking feature assembler: {e}")
        
        return assembler_results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite"""
        fraud_logger.info("Starting comprehensive performance benchmark...")
        
        try:
            # Run all benchmark categories
            self.benchmark_model_training_performance()
            self.benchmark_prediction_performance()
            self.benchmark_memory_usage()
            self.benchmark_scalability()
            self.benchmark_feature_assembler_performance()
            
            # Generate performance summary
            summary = self._generate_performance_summary()
            
            fraud_logger.info("Comprehensive benchmark completed")
            return summary
            
        except Exception as e:
            fraud_logger.error(f"Benchmark suite failed: {e}")
            return {'error': str(e)}
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance benchmark summary"""
        summary = {
            'benchmark_completed': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            # Analyze prediction performance
            if 'prediction' in self.benchmark_results['model_performance']:
                pred_results = self.benchmark_results['model_performance']['prediction']
                
                # Find best performing model for single predictions
                best_single_prediction = None
                best_time = float('inf')
                
                for model_name in pred_results.get('models_tested', []):
                    if model_name in pred_results.get('batch_performance', {}):
                        if 1 in pred_results['batch_performance'][model_name]:
                            time_ms = pred_results['batch_performance'][model_name][1]['time_per_prediction_ms']
                            if time_ms < best_time:
                                best_time = time_ms
                                best_single_prediction = model_name
                
                summary['performance_metrics']['best_single_prediction_model'] = best_single_prediction
                summary['performance_metrics']['best_single_prediction_time_ms'] = best_time
                
                # Check if performance requirements are met
                if best_time <= 10:  # 10ms target for mapping
                    summary['recommendations'].append('Mapping performance meets requirements')
                else:
                    summary['recommendations'].append(f'Mapping performance needs improvement: {best_time:.2f}ms > 10ms target')
            
            # Analyze memory usage
            if 'model_memory_usage' in self.benchmark_results['memory_usage']:
                memory_results = self.benchmark_results['memory_usage']['model_memory_usage']
                
                total_memory = 0
                for model_name, memory_info in memory_results.items():
                    total_memory += memory_info.get('total_memory_mb', 0)
                
                summary['performance_metrics']['total_memory_usage_mb'] = total_memory
                
                if total_memory > 500:  # 500MB threshold
                    summary['recommendations'].append(f'High memory usage detected: {total_memory:.1f}MB')
                else:
                    summary['recommendations'].append('Memory usage within acceptable limits')
            
            # Analyze scalability
            if 'scaling_performance' in self.benchmark_results['scalability_tests']:
                scaling_results = self.benchmark_results['scalability_tests']['scaling_performance']
                
                # Check if throughput scales linearly
                throughputs = [info['throughput_per_second'] for info in scaling_results.values()]
                if throughputs:
                    max_throughput = max(throughputs)
                    summary['performance_metrics']['max_throughput_per_second'] = max_throughput
                    
                    if max_throughput >= 100:  # 100 predictions per second
                        summary['recommendations'].append('Throughput meets scalability requirements')
                    else:
                        summary['recommendations'].append(f'Throughput may need improvement: {max_throughput:.1f} pred/s')
            
        except Exception as e:
            summary['error'] = f"Error generating summary: {e}"
        
        return summary
    
    def save_benchmark_results(self, output_path: str = "test_results/mapping_performance_benchmark.json"):
        """Save benchmark results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)
            
            fraud_logger.info(f"Benchmark results saved to {output_path}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to save benchmark results: {e}")


# Benchmark runner functions
def run_training_benchmark():
    """Run training performance benchmark"""
    benchmark = MappingPerformanceBenchmark()
    return benchmark.benchmark_model_training_performance()


def run_prediction_benchmark():
    """Run prediction performance benchmark"""
    benchmark = MappingPerformanceBenchmark()
    return benchmark.benchmark_prediction_performance()


def run_memory_benchmark():
    """Run memory usage benchmark"""
    benchmark = MappingPerformanceBenchmark()
    return benchmark.benchmark_memory_usage()


def run_scalability_benchmark():
    """Run scalability benchmark"""
    benchmark = MappingPerformanceBenchmark()
    return benchmark.benchmark_scalability()


def run_all_benchmarks():
    """Run comprehensive benchmark suite"""
    benchmark = MappingPerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.save_benchmark_results()
    return results


if __name__ == "__main__":
    # Run comprehensive benchmark suite
    print("Starting Feature Mapping Performance Benchmark...")
    print("=" * 80)
    
    results = run_all_benchmarks()
    
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))
    
    if 'recommendations' in results:
        print("\nPerformance Recommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")