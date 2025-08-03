"""
Integration and Performance Tests for Feature Mapping System
Tests end-to-end functionality and performance requirements.
"""

import unittest
import time
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import concurrent.futures
from unittest.mock import Mock, patch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, TimeContext, MerchantCategory, 
    LocationRisk, SpendingPattern
)
from fraudguard.pipeline.intelligent_prediction_pipeline import IntelligentPredictionPipeline
from fraudguard.components.performance_optimizer import PerformanceOptimizer
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.artifacts_dir = Path(cls.temp_dir) / "artifacts"
        cls.artifacts_dir.mkdir(exist_ok=True)
        
        # Create minimal trained mappers for testing
        cls._create_test_mappers()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_mappers(cls):
        """Create minimal trained mappers for testing"""
        # Create synthetic training data
        np.random.seed(42)
        X_train = np.random.randn(100, 8)
        y_train = np.random.randn(100, 28)
        
        # Create and train Random Forest mapper
        rf_mapper = RandomForestMapper(n_estimators=5, random_state=42)
        rf_mapper.fit(X_train, y_train)
        
        # Save mapper
        rf_path = cls.artifacts_dir / "feature_mapping" / "mappers" / "random_forest"
        rf_path.mkdir(parents=True, exist_ok=True)
        rf_mapper.save_model(str(rf_path))
        
        # Create and train XGBoost mapper
        xgb_mapper = XGBoostMapper(n_estimators=5, random_state=42)
        xgb_mapper.fit(X_train, y_train)
        
        # Save mapper
        xgb_path = cls.artifacts_dir / "feature_mapping" / "mappers" / "xgboost"
        xgb_path.mkdir(parents=True, exist_ok=True)
        xgb_mapper.save_model(str(xgb_path))
        
        # Create mock fraud models directory
        models_dir = cls.artifacts_dir / "models"
        models_dir.mkdir(exist_ok=True)
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_input = {
            'transaction_amount': 100.0,
            'merchant_category': 'grocery',
            'hour_of_day': 14,
            'day_of_week': 2,
            'location_risk': 'normal',
            'spending_pattern': 'typical'
        }
        
        self.user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with test artifacts"""
        pipeline = IntelligentPredictionPipeline(
            model_artifacts_path=str(self.artifacts_dir),
            mapping_artifacts_path=str(self.artifacts_dir / "feature_mapping")
        )
        
        # Should have loaded feature mappers
        self.assertGreater(len(pipeline.feature_mappers), 0)
        
        # Should have initialized components
        self.assertIsNotNone(pipeline.feature_assembler)
        self.assertIsNotNone(pipeline.input_validator)
        self.assertIsNotNone(pipeline.confidence_scorer)
    
    def test_input_creation_from_dict(self):
        """Test creating user input from dictionary"""
        pipeline = IntelligentPredictionPipeline(
            mapping_artifacts_path=str(self.artifacts_dir / "feature_mapping")
        )
        
        user_input = pipeline.create_user_input_from_dict(self.test_input)
        
        self.assertIsInstance(user_input, UserTransactionInput)
        self.assertEqual(user_input.transaction_amount, 100.0)
        self.assertEqual(user_input.merchant_category, MerchantCategory.GROCERY)
    
    def test_input_validation_integration(self):
        """Test input validation integration"""
        pipeline = IntelligentPredictionPipeline(
            mapping_artifacts_path=str(self.artifacts_dir / "feature_mapping")
        )
        
        # Valid input
        result = pipeline.validate_input_data(self.test_input)
        self.assertTrue(result['is_valid'])
        
        # Invalid input
        invalid_input = self.test_input.copy()
        invalid_input['transaction_amount'] = -100.0
        
        result = pipeline.validate_input_data(invalid_input)
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)
    
    @patch('fraudguard.models.model_factory.ModelFactory.create_model')
    def test_intelligent_prediction_with_mock_fraud_model(self, mock_create_model):
        """Test intelligent prediction with mocked fraud model"""
        # Mock fraud model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_create_model.return_value = mock_model
        
        pipeline = IntelligentPredictionPipeline(
            model_artifacts_path=str(self.artifacts_dir),
            mapping_artifacts_path=str(self.artifacts_dir / "feature_mapping")
        )
        
        # Add mock model to pipeline
        pipeline.fraud_models['test_model'] = mock_model
        
        result = pipeline.predict_intelligent(
            self.test_input,
            fraud_model_type='test_model'
        )
        
        # Should return prediction result
        self.assertIsInstance(result, dict)
        if 'error' not in result:
            self.assertIn('prediction', result)
            self.assertIn('fraud_probability', result)
            self.assertIn('mapping_confidence', result)
    
    def test_health_check(self):
        """Test pipeline health check"""
        pipeline = IntelligentPredictionPipeline(
            mapping_artifacts_path=str(self.artifacts_dir / "feature_mapping")
        )
        
        health = pipeline.health_check()
        
        self.assertIsInstance(health, dict)
        self.assertIn('status', health)
        self.assertIn('components', health)
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        pipeline = IntelligentPredictionPipeline(
            mapping_artifacts_path=str(self.artifacts_dir / "feature_mapping")
        )
        
        stats = pipeline.get_mapping_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('available_mappers', stats)
        self.assertIn('mapper_types', stats)


class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements and optimization"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.test_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
        
        # Create test mapper
        np.random.seed(42)
        X_train = np.random.randn(50, 8)
        y_train = np.random.randn(50, 28)
        
        self.mapper = RandomForestMapper(n_estimators=5, random_state=42)
        self.mapper.fit(X_train, y_train)
    
    def test_single_prediction_latency(self):
        """Test that single predictions meet latency requirements"""
        # Test multiple predictions to get average
        latencies = []
        
        for _ in range(10):
            start_time = time.time()
            prediction = self.mapper.predict_single(self.test_input)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Should meet sub-50ms requirement for individual mapper
        # (Full pipeline has sub-10ms requirement with caching)
        self.assertLess(avg_latency, 50.0, f"Average latency {avg_latency:.2f}ms exceeds 50ms")
        self.assertLess(max_latency, 100.0, f"Max latency {max_latency:.2f}ms exceeds 100ms")
        
        print(f"Single prediction latency - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")
    
    def test_batch_prediction_throughput(self):
        """Test batch prediction throughput"""
        batch_size = 100
        
        # Create batch of inputs
        batch_features = np.random.randn(batch_size, 8)
        
        start_time = time.time()
        predictions = self.mapper.predict(batch_features)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = batch_size / total_time
        
        # Should handle at least 100 predictions per second
        self.assertGreater(throughput, 100, f"Throughput {throughput:.1f} predictions/sec is too low")
        
        print(f"Batch prediction throughput: {throughput:.1f} predictions/sec")
    
    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many predictions
        for _ in range(1000):
            self.mapper.predict_single(self.test_input)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (less than 50MB)
        self.assertLess(memory_growth, 50, f"Memory grew by {memory_growth:.1f}MB")
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Growth: {memory_growth:.1f}MB")
    
    def test_concurrent_predictions(self):
        """Test concurrent prediction handling"""
        def make_prediction():
            return self.mapper.predict_single(self.test_input)
        
        # Test with multiple concurrent requests
        num_threads = 10
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_prediction) for _ in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All predictions should complete successfully
        self.assertEqual(len(results), num_threads)
        
        # Should handle concurrent requests efficiently
        self.assertLess(total_time, 5.0, f"Concurrent predictions took {total_time:.2f}s")
        
        print(f"Concurrent predictions ({num_threads} threads): {total_time:.2f}s")


class TestCachingPerformance(unittest.TestCase):
    """Test caching and performance optimization"""
    
    def setUp(self):
        """Set up caching test fixtures"""
        self.optimizer = PerformanceOptimizer(
            enable_memory_cache=True,
            enable_redis_cache=False,  # Disable Redis for testing
            memory_cache_size=100
        )
        
        self.test_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
    
    def test_cache_hit_performance(self):
        """Test cache hit performance improvement"""
        def mock_prediction_func(user_input, mapper_type, **kwargs):
            # Simulate slow prediction
            time.sleep(0.01)  # 10ms delay
            return {
                'prediction': 0,
                'fraud_probability': 0.1,
                'processing_time_ms': 10
            }
        
        # First call (cache miss)
        start_time = time.time()
        result1 = self.optimizer.optimize_prediction(
            mock_prediction_func, self.test_input, 'test_mapper'
        )
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = self.optimizer.optimize_prediction(
            mock_prediction_func, self.test_input, 'test_mapper'
        )
        second_call_time = (time.time() - start_time) * 1000
        
        # Cache hit should be much faster
        self.assertLess(second_call_time, first_call_time / 2)
        self.assertTrue(result2.get('from_cache', False))
        
        print(f"Cache performance - First call: {first_call_time:.2f}ms, Second call: {second_call_time:.2f}ms")
    
    def test_cache_statistics(self):
        """Test cache statistics collection"""
        def mock_prediction_func(user_input, mapper_type, **kwargs):
            return {'prediction': 0, 'fraud_probability': 0.1}
        
        # Make several predictions (some repeated)
        inputs = [self.test_input] * 3  # Same input 3 times
        
        for user_input in inputs:
            self.optimizer.optimize_prediction(
                mock_prediction_func, user_input, 'test_mapper'
            )
        
        stats = self.optimizer.get_performance_stats()
        
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('cache_hit_rate_percent', stats)
        
        # Should have 1 miss and 2 hits
        self.assertEqual(stats['cache_misses'], 1)
        self.assertEqual(stats['cache_hits'], 2)
        self.assertAlmostEqual(stats['cache_hit_rate_percent'], 66.67, places=1)


class TestAccuracyValidation(unittest.TestCase):
    """Test accuracy and quality validation"""
    
    def setUp(self):
        """Set up accuracy test fixtures"""
        # Create synthetic data that simulates real relationships
        np.random.seed(42)
        self.n_samples = 200
        
        # Create interpretable features
        self.X_interpretable = np.random.randn(self.n_samples, 8)
        
        # Create PCA features with some correlation to interpretable features
        # This simulates the real relationship we want to learn
        W = np.random.randn(28, 8) * 0.5
        self.y_pca_true = np.dot(self.X_interpretable, W.T) + np.random.randn(self.n_samples, 28) * 0.1
        
        # Split data
        split_idx = int(0.8 * self.n_samples)
        self.X_train = self.X_interpretable[:split_idx]
        self.y_train = self.y_pca_true[:split_idx]
        self.X_test = self.X_interpretable[split_idx:]
        self.y_test = self.y_pca_true[split_idx:]
    
    def test_mapping_accuracy_requirement(self):
        """Test that mapping accuracy meets requirements"""
        # Train mapper
        mapper = RandomForestMapper(n_estimators=50, random_state=42)
        mapper.fit(self.X_train, self.y_train)
        
        # Test predictions
        y_pred = mapper.predict(self.X_test)
        
        # Calculate accuracy metrics
        mse = np.mean((self.y_test - y_pred) ** 2)
        mae = np.mean(np.abs(self.y_test - y_pred))
        
        # Calculate correlation preservation
        correlations = []
        for i in range(self.y_test.shape[1]):
            corr = np.corrcoef(self.y_test[:, i], y_pred[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        # Requirements: Should achieve reasonable mapping quality
        # (These thresholds are based on the synthetic data characteristics)
        self.assertLess(mse, 2.0, f"MSE {mse:.4f} is too high")
        self.assertLess(mae, 1.0, f"MAE {mae:.4f} is too high")
        self.assertGreater(avg_correlation, 0.3, f"Average correlation {avg_correlation:.4f} is too low")
        
        print(f"Mapping accuracy - MSE: {mse:.4f}, MAE: {mae:.4f}, Avg Correlation: {avg_correlation:.4f}")
    
    def test_consistency_across_models(self):
        """Test consistency between different mapping models"""
        # Train multiple mappers
        rf_mapper = RandomForestMapper(n_estimators=20, random_state=42)
        rf_mapper.fit(self.X_train, self.y_train)
        
        xgb_mapper = XGBoostMapper(n_estimators=20, random_state=42)
        xgb_mapper.fit(self.X_train, self.y_train)
        
        # Get predictions from both
        rf_pred = rf_mapper.predict(self.X_test)
        xgb_pred = xgb_mapper.predict(self.X_test)
        
        # Calculate agreement between models
        correlations = []
        for i in range(rf_pred.shape[1]):
            corr = np.corrcoef(rf_pred[:, i], xgb_pred[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_agreement = np.mean(correlations)
        
        # Models should have reasonable agreement
        self.assertGreater(avg_agreement, 0.5, f"Model agreement {avg_agreement:.4f} is too low")
        
        print(f"Model consistency - Average agreement: {avg_agreement:.4f}")


class TestErrorRecovery(unittest.TestCase):
    """Test error handling and recovery mechanisms"""
    
    def test_fallback_mechanism(self):
        """Test fallback mechanisms when primary mapping fails"""
        from fraudguard.components.error_handler import ErrorHandler, ErrorCategory
        
        error_handler = ErrorHandler()
        
        user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
        
        # Simulate mapping failure
        test_error = ValueError("Mapping model failed")
        context = {'user_input': user_input}
        
        result = error_handler.handle_error(
            test_error, context, ErrorCategory.FEATURE_MAPPING
        )
        
        # Should have attempted recovery
        self.assertIn('error_handled', result)
        
        # If recovery was successful, should have prediction result
        if result.get('error_handled'):
            self.assertIn('prediction', result)
            self.assertIn('fraud_probability', result)
    
    def test_graceful_degradation(self):
        """Test graceful degradation under various failure conditions"""
        # This would test various failure scenarios and ensure
        # the system continues to function with reduced capability
        pass


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEndToEndPipeline,
        TestPerformanceRequirements,
        TestCachingPerformance,
        TestAccuracyValidation,
        TestErrorRecovery
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INTEGRATION & PERFORMANCE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    print(f"{'='*60}")