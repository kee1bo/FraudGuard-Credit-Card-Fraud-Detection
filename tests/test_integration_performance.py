"""
Integration and Performance Tests for Intelligent Feature Mapping System
Tests the complete pipeline from user input to fraud prediction with performance validation.
"""

import pytest
import time
import asyncio
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import statistics
import json

# Import system components
from fraudguard.entity.feature_mapping_entity import UserTransactionInput, MappingResult
from fraudguard.pipeline.intelligent_prediction_pipeline import IntelligentPredictionPipeline
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper
from fraudguard.models.ensemble_mapper import EnsembleMapper
from fraudguard.components.feature_assembler import FeatureVectorAssembler
from fraudguard.components.confidence_scorer import ConfidenceScorer
from fraudguard.components.mapping_quality_monitor import MappingQualityMonitor
from fraudguard.logger import fraud_logger


class IntegrationTestSuite:
    """Comprehensive integration test suite for the intelligent mapping system"""
    
    def __init__(self):
        self.pipeline = None
        self.test_results = {
            'integration_tests': {},
            'performance_tests': {},
            'accuracy_tests': {},
            'load_tests': {}
        }
        
        # Test data
        self.sample_inputs = self._generate_test_inputs()
        self.ulb_reference_data = None
        
    def _generate_test_inputs(self) -> List[UserTransactionInput]:
        """Generate diverse test inputs for comprehensive testing"""
        test_cases = [
            # Normal transaction cases
            UserTransactionInput(
                transaction_amount=50.0,
                merchant_category="grocery_store",
                location_risk_score=0.1,
                time_since_last_transaction=2.5,
                spending_pattern_score=0.3
            ),
            UserTransactionInput(
                transaction_amount=1200.0,
                merchant_category="electronics_store",
                location_risk_score=0.2,
                time_since_last_transaction=24.0,
                spending_pattern_score=0.6
            ),
            UserTransactionInput(
                transaction_amount=25.0,
                merchant_category="gas_station",
                location_risk_score=0.05,
                time_since_last_transaction=1.0,
                spending_pattern_score=0.2
            ),
            
            # Edge cases
            UserTransactionInput(
                transaction_amount=0.01,  # Minimum amount
                merchant_category="online_retail",
                location_risk_score=0.0,
                time_since_last_transaction=0.1,
                spending_pattern_score=0.0
            ),
            UserTransactionInput(
                transaction_amount=10000.0,  # High amount
                merchant_category="luxury_goods",
                location_risk_score=0.9,
                time_since_last_transaction=168.0,  # 1 week
                spending_pattern_score=0.95
            ),
            
            # Suspicious patterns
            UserTransactionInput(
                transaction_amount=500.0,
                merchant_category="atm_withdrawal",
                location_risk_score=0.8,
                time_since_last_transaction=0.5,
                spending_pattern_score=0.9
            ),
            UserTransactionInput(
                transaction_amount=2500.0,
                merchant_category="online_gambling",
                location_risk_score=0.7,
                time_since_last_transaction=0.2,
                spending_pattern_score=0.85
            ),
            
            # International/travel scenarios
            UserTransactionInput(
                transaction_amount=150.0,
                merchant_category="restaurant",
                location_risk_score=0.6,  # Different country
                time_since_last_transaction=12.0,
                spending_pattern_score=0.4
            ),
        ]
        
        return test_cases
    
    def setup_test_environment(self) -> bool:
        """Set up the test environment with required models and data"""
        try:
            fraud_logger.info("Setting up integration test environment...")
            
            # Initialize the intelligent prediction pipeline
            self.pipeline = IntelligentPredictionPipeline()
            
            # Check if models are available
            models_available = self._check_model_availability()
            if not models_available:
                fraud_logger.warning("Some models not available, creating mock models for testing")
                self._create_mock_models()
            
            # Load reference data if available
            self._load_reference_data()
            
            fraud_logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Failed to setup test environment: {e}")
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if required models are available"""
        try:
            # Try to load a sample mapping model
            mapper_path = Path("artifacts/feature_mapping/mappers/random_forest")
            return mapper_path.exists()
        except Exception:
            return False
    
    def _create_mock_models(self):
        """Create mock models for testing when real models aren't available"""
        try:
            # Create mock Random Forest mapper
            mock_rf = RandomForestMapper(n_estimators=10)  # Small for testing
            
            # Generate mock training data
            X_mock = np.random.rand(100, 5)  # 100 samples, 5 interpretable features
            y_mock = np.random.rand(100, 28)  # 28 PCA components
            
            mock_rf.fit(X_mock, y_mock)
            
            # Save mock model
            mock_path = Path("artifacts/feature_mapping/mappers/random_forest")
            mock_path.mkdir(parents=True, exist_ok=True)
            mock_rf.save_model(str(mock_path))
            
            fraud_logger.info("Created mock models for testing")
            
        except Exception as e:
            fraud_logger.error(f"Failed to create mock models: {e}")
    
    def _load_reference_data(self):
        """Load reference data for accuracy validation"""
        try:
            ulb_path = Path("data/creditcard.csv")
            if ulb_path.exists():
                # Load a sample of ULB data for comparison
                df = pd.read_csv(ulb_path)
                self.ulb_reference_data = df.sample(n=min(1000, len(df)), random_state=42)
                fraud_logger.info(f"Loaded {len(self.ulb_reference_data)} reference samples")
            else:
                fraud_logger.warning("ULB reference data not available")
        except Exception as e:
            fraud_logger.error(f"Failed to load reference data: {e}")
    
    def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete pipeline from user input to fraud prediction"""
        test_results = {
            'test_name': 'end_to_end_pipeline',
            'total_tests': len(self.sample_inputs),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_response_time': 0,
            'response_times': [],
            'prediction_results': [],
            'errors': []
        }
        
        fraud_logger.info("Starting end-to-end pipeline tests...")
        
        for i, test_input in enumerate(self.sample_inputs):
            try:
                start_time = time.time()
                
                # Make prediction through the intelligent pipeline
                result = self.pipeline.predict_with_mapping(test_input)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Validate result structure
                self._validate_prediction_result(result)
                
                test_results['successful_predictions'] += 1
                test_results['response_times'].append(response_time)
                test_results['prediction_results'].append({
                    'input_index': i,
                    'response_time_ms': response_time,
                    'fraud_probability': result.fraud_probability,
                    'confidence_score': result.confidence_score,
                    'mapping_quality': result.mapping_result.quality_score if result.mapping_result else None
                })
                
                fraud_logger.info(f"Test {i+1}/{len(self.sample_inputs)}: Success ({response_time:.2f}ms)")
                
            except Exception as e:
                test_results['failed_predictions'] += 1
                test_results['errors'].append({
                    'input_index': i,
                    'error': str(e),
                    'input_data': test_input.__dict__
                })
                fraud_logger.error(f"Test {i+1}/{len(self.sample_inputs)}: Failed - {e}")
        
        # Calculate statistics
        if test_results['response_times']:
            test_results['average_response_time'] = statistics.mean(test_results['response_times'])
            test_results['median_response_time'] = statistics.median(test_results['response_times'])
            test_results['max_response_time'] = max(test_results['response_times'])
            test_results['min_response_time'] = min(test_results['response_times'])
        
        test_results['success_rate'] = (test_results['successful_predictions'] / test_results['total_tests']) * 100
        
        self.test_results['integration_tests']['end_to_end'] = test_results
        
        fraud_logger.info(f"End-to-end tests completed: {test_results['success_rate']:.1f}% success rate")
        return test_results
    
    def _validate_prediction_result(self, result):
        """Validate the structure and content of prediction results"""
        # Check required attributes
        required_attrs = ['fraud_probability', 'confidence_score', 'prediction_explanation']
        for attr in required_attrs:
            if not hasattr(result, attr):
                raise ValueError(f"Missing required attribute: {attr}")
        
        # Validate value ranges
        if not (0 <= result.fraud_probability <= 1):
            raise ValueError(f"Invalid fraud probability: {result.fraud_probability}")
        
        if not (0 <= result.confidence_score <= 1):
            raise ValueError(f"Invalid confidence score: {result.confidence_score}")
        
        # Check mapping result if present
        if hasattr(result, 'mapping_result') and result.mapping_result:
            if not hasattr(result.mapping_result, 'mapped_features'):
                raise ValueError("Missing mapped_features in mapping result")
            
            if len(result.mapping_result.mapped_features) != 28:
                raise ValueError(f"Expected 28 mapped features, got {len(result.mapping_result.mapped_features)}")
    
    def test_performance_requirements(self) -> Dict[str, Any]:
        """Test that the system meets performance requirements"""
        performance_results = {
            'test_name': 'performance_requirements',
            'target_response_time_ms': 50,
            'tests_passed': 0,
            'tests_failed': 0,
            'response_times': [],
            'performance_summary': {}
        }
        
        fraud_logger.info("Testing performance requirements...")
        
        # Test multiple iterations for statistical significance
        num_iterations = 20
        
        for i in range(num_iterations):
            # Use a representative test case
            test_input = self.sample_inputs[0]
            
            start_time = time.time()
            try:
                result = self.pipeline.predict_with_mapping(test_input)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000
                performance_results['response_times'].append(response_time)
                
                if response_time <= performance_results['target_response_time_ms']:
                    performance_results['tests_passed'] += 1
                else:
                    performance_results['tests_failed'] += 1
                
            except Exception as e:
                performance_results['tests_failed'] += 1
                fraud_logger.error(f"Performance test iteration {i+1} failed: {e}")
        
        # Calculate performance statistics
        if performance_results['response_times']:
            times = performance_results['response_times']
            performance_results['performance_summary'] = {
                'average_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
                'max_ms': max(times),
                'min_ms': min(times),
                'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0
            }
        
        performance_results['pass_rate'] = (performance_results['tests_passed'] / num_iterations) * 100
        performance_results['meets_requirements'] = performance_results['pass_rate'] >= 95  # 95% of requests should meet target
        
        self.test_results['performance_tests']['response_time'] = performance_results
        
        fraud_logger.info(f"Performance tests completed: {performance_results['pass_rate']:.1f}% pass rate")
        return performance_results
    
    def test_concurrent_load(self, num_concurrent_requests: int = 10) -> Dict[str, Any]:
        """Test system performance under concurrent load"""
        load_results = {
            'test_name': 'concurrent_load',
            'concurrent_requests': num_concurrent_requests,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time_seconds': 0,
            'requests_per_second': 0,
            'response_times': [],
            'errors': []
        }
        
        fraud_logger.info(f"Testing concurrent load with {num_concurrent_requests} requests...")
        
        def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single request and return timing info"""
            try:
                test_input = self.sample_inputs[request_id % len(self.sample_inputs)]
                start_time = time.time()
                
                result = self.pipeline.predict_with_mapping(test_input)
                
                end_time = time.time()
                return {
                    'request_id': request_id,
                    'success': True,
                    'response_time': (end_time - start_time) * 1000,
                    'fraud_probability': result.fraud_probability
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'response_time': None
                }
        
        # Execute concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        load_results['total_time_seconds'] = end_time - start_time
        
        # Process results
        for result in results:
            if result['success']:
                load_results['successful_requests'] += 1
                if result['response_time']:
                    load_results['response_times'].append(result['response_time'])
            else:
                load_results['failed_requests'] += 1
                load_results['errors'].append(result)
        
        # Calculate throughput
        if load_results['total_time_seconds'] > 0:
            load_results['requests_per_second'] = num_concurrent_requests / load_results['total_time_seconds']
        
        # Calculate response time statistics
        if load_results['response_times']:
            times = load_results['response_times']
            load_results['response_time_stats'] = {
                'average_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'p95_ms': np.percentile(times, 95),
                'max_ms': max(times),
                'min_ms': min(times)
            }
        
        load_results['success_rate'] = (load_results['successful_requests'] / num_concurrent_requests) * 100
        
        self.test_results['load_tests']['concurrent'] = load_results
        
        fraud_logger.info(f"Load test completed: {load_results['success_rate']:.1f}% success rate, {load_results['requests_per_second']:.2f} RPS")
        return load_results
    
    def test_mapping_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of feature mapping against reference data"""
        accuracy_results = {
            'test_name': 'mapping_accuracy',
            'tests_performed': 0,
            'correlation_scores': [],
            'distribution_similarities': [],
            'fraud_detection_accuracies': [],
            'overall_accuracy_score': 0
        }
        
        if not self.ulb_reference_data is not None:
            fraud_logger.warning("No reference data available for accuracy testing")
            accuracy_results['error'] = "No reference data available"
            return accuracy_results
        
        fraud_logger.info("Testing mapping accuracy against reference data...")
        
        try:
            # Sample reference transactions for testing
            test_sample = self.ulb_reference_data.sample(n=min(50, len(self.ulb_reference_data)))
            
            for _, row in test_sample.iterrows():
                try:
                    # Create user input from reference data (reverse engineering)
                    user_input = self._create_user_input_from_reference(row)
                    
                    # Get mapping result
                    result = self.pipeline.predict_with_mapping(user_input)
                    
                    if result.mapping_result:
                        # Compare mapped features with actual features
                        actual_features = row[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']].values
                        
                        mapped_features = np.array(result.mapping_result.mapped_features)
                        
                        # Calculate correlation
                        correlation = np.corrcoef(actual_features, mapped_features)[0, 1]
                        if not np.isnan(correlation):\n                            accuracy_results['correlation_scores'].append(correlation)
                        
                        accuracy_results['tests_performed'] += 1
                    
                except Exception as e:
                    fraud_logger.error(f"Error in accuracy test: {e}")
                    continue
            
            # Calculate overall accuracy metrics
            if accuracy_results['correlation_scores']:
                accuracy_results['overall_accuracy_score'] = statistics.mean(accuracy_results['correlation_scores'])
                accuracy_results['accuracy_std'] = statistics.stdev(accuracy_results['correlation_scores'])
                accuracy_results['min_correlation'] = min(accuracy_results['correlation_scores'])
                accuracy_results['max_correlation'] = max(accuracy_results['correlation_scores'])
            
            self.test_results['accuracy_tests']['mapping'] = accuracy_results
            
        except Exception as e:
            accuracy_results['error'] = str(e)
            fraud_logger.error(f"Accuracy testing failed: {e}")
        
        fraud_logger.info(f"Accuracy tests completed: {accuracy_results['tests_performed']} tests performed")
        return accuracy_results
    
    def _create_user_input_from_reference(self, row) -> UserTransactionInput:
        """Create user input from reference ULB data row"""
        # This is a simplified reverse engineering - in practice, this would be more sophisticated
        amount = row['Amount']
        time_val = row['Time']
        
        # Map to interpretable features (simplified)
        merchant_categories = ["grocery_store", "gas_station", "restaurant", "online_retail", "electronics_store"]
        merchant_category = np.random.choice(merchant_categories)
        
        # Derive other features from amount and time patterns
        location_risk = min(0.9, amount / 1000.0 * 0.1)  # Higher amounts = slightly higher risk
        time_since_last = max(0.1, (time_val % 86400) / 3600)  # Convert to hours
        spending_pattern = min(0.9, amount / 500.0 * 0.2)  # Pattern based on amount
        
        return UserTransactionInput(
            transaction_amount=amount,
            merchant_category=merchant_category,
            location_risk_score=location_risk,
            time_since_last_transaction=time_since_last,
            spending_pattern_score=spending_pattern
        )
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all integration and performance tests"""
        fraud_logger.info("Starting comprehensive test suite...")
        
        # Setup test environment
        if not self.setup_test_environment():
            return {'error': 'Failed to setup test environment'}
        
        # Run all test categories
        try:
            # Integration tests
            self.test_end_to_end_pipeline()
            
            # Performance tests
            self.test_performance_requirements()
            
            # Load tests
            self.test_concurrent_load(num_concurrent_requests=5)  # Reduced for CI/CD
            
            # Accuracy tests
            self.test_mapping_accuracy()
            
            # Generate summary
            summary = self._generate_test_summary()
            
            fraud_logger.info("Comprehensive test suite completed")
            return summary
            
        except Exception as e:
            fraud_logger.error(f"Test suite failed: {e}")
            return {'error': str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'test_suite_completed': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': 'PASS',
            'test_categories': {},
            'recommendations': []
        }
        
        # Analyze integration tests
        if 'end_to_end' in self.test_results['integration_tests']:
            integration_result = self.test_results['integration_tests']['end_to_end']
            summary['test_categories']['integration'] = {
                'status': 'PASS' if integration_result['success_rate'] >= 90 else 'FAIL',
                'success_rate': integration_result['success_rate'],
                'average_response_time': integration_result.get('average_response_time', 0)
            }
            
            if integration_result['success_rate'] < 90:
                summary['overall_status'] = 'FAIL'
                summary['recommendations'].append('Integration test success rate below 90%')
        
        # Analyze performance tests
        if 'response_time' in self.test_results['performance_tests']:
            perf_result = self.test_results['performance_tests']['response_time']
            summary['test_categories']['performance'] = {
                'status': 'PASS' if perf_result['meets_requirements'] else 'FAIL',
                'pass_rate': perf_result['pass_rate'],
                'average_response_time': perf_result['performance_summary'].get('average_ms', 0)
            }
            
            if not perf_result['meets_requirements']:
                summary['overall_status'] = 'FAIL'
                summary['recommendations'].append('Performance requirements not met')
        
        # Analyze load tests
        if 'concurrent' in self.test_results['load_tests']:
            load_result = self.test_results['load_tests']['concurrent']
            summary['test_categories']['load'] = {
                'status': 'PASS' if load_result['success_rate'] >= 95 else 'FAIL',
                'success_rate': load_result['success_rate'],
                'requests_per_second': load_result['requests_per_second']
            }
            
            if load_result['success_rate'] < 95:
                summary['overall_status'] = 'FAIL'
                summary['recommendations'].append('Load test success rate below 95%')
        
        # Analyze accuracy tests
        if 'mapping' in self.test_results['accuracy_tests']:
            accuracy_result = self.test_results['accuracy_tests']['mapping']
            if 'overall_accuracy_score' in accuracy_result:
                summary['test_categories']['accuracy'] = {
                    'status': 'PASS' if accuracy_result['overall_accuracy_score'] >= 0.7 else 'FAIL',
                    'accuracy_score': accuracy_result['overall_accuracy_score'],
                    'tests_performed': accuracy_result['tests_performed']
                }
                
                if accuracy_result['overall_accuracy_score'] < 0.7:
                    summary['overall_status'] = 'FAIL'
                    summary['recommendations'].append('Mapping accuracy below acceptable threshold')
        
        # Add general recommendations
        if summary['overall_status'] == 'PASS':
            summary['recommendations'].append('All tests passed - system ready for production')
        else:
            summary['recommendations'].append('Review failed tests before production deployment')
        
        return summary
    
    def save_test_results(self, output_path: str = "test_results/integration_performance_results.json"):
        """Save test results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            fraud_logger.info(f"Test results saved to {output_path}")
            
        except Exception as e:
            fraud_logger.error(f"Failed to save test results: {e}")


# Test runner functions
def run_integration_tests():
    """Run integration tests"""
    test_suite = IntegrationTestSuite()
    return test_suite.test_end_to_end_pipeline()


def run_performance_tests():
    """Run performance tests"""
    test_suite = IntegrationTestSuite()
    test_suite.setup_test_environment()
    return test_suite.test_performance_requirements()


def run_load_tests(concurrent_requests: int = 10):
    """Run load tests"""
    test_suite = IntegrationTestSuite()
    test_suite.setup_test_environment()
    return test_suite.test_concurrent_load(concurrent_requests)


def run_all_tests():
    """Run comprehensive test suite"""
    test_suite = IntegrationTestSuite()
    results = test_suite.run_comprehensive_test_suite()
    test_suite.save_test_results()
    return results


if __name__ == "__main__":
    # Run comprehensive test suite
    print("Starting Intelligent Feature Mapping Integration & Performance Tests...")
    print("=" * 80)
    
    results = run_all_tests()
    
    print("\nTest Results Summary:")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))
    
    if results.get('overall_status') == 'PASS':
        print("\n✅ All tests passed! System ready for production.")
    else:
        print("\n❌ Some tests failed. Review results before deployment.")
        if 'recommendations' in results:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")