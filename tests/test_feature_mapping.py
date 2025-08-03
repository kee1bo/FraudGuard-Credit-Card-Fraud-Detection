"""
Comprehensive Unit Tests for Feature Mapping System
Tests all components of the intelligent feature mapping pipeline.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, TimeContext, MerchantCategory, 
    LocationRisk, SpendingPattern, MappingExplanation, QualityMetrics
)
from fraudguard.models.random_forest_mapper import RandomForestMapper
from fraudguard.models.xgboost_mapper import XGBoostMapper
from fraudguard.components.feature_assembler import FeatureVectorAssembler
from fraudguard.components.input_validator import InputValidator
from fraudguard.components.confidence_scorer import ConfidenceScorer
from fraudguard.components.mapping_quality_monitor import MappingQualityMonitor
from fraudguard.components.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class TestFeatureMappingEntities(unittest.TestCase):
    """Test feature mapping entity classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.time_context = TimeContext(
            hour_of_day=14,
            day_of_week=2,
            is_weekend=False
        )
        
        self.user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=self.time_context,
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
    
    def test_time_context_creation(self):
        """Test TimeContext creation and validation"""
        # Valid time context
        time_ctx = TimeContext(hour_of_day=15, day_of_week=3, is_weekend=False)
        self.assertEqual(time_ctx.hour_of_day, 15)
        self.assertEqual(time_ctx.day_of_week, 3)
        self.assertFalse(time_ctx.is_weekend)
        
        # Invalid hour
        with self.assertRaises(ValueError):
            TimeContext(hour_of_day=25, day_of_week=3, is_weekend=False)
        
        # Invalid day
        with self.assertRaises(ValueError):
            TimeContext(hour_of_day=15, day_of_week=8, is_weekend=False)
    
    def test_user_transaction_input_creation(self):
        """Test UserTransactionInput creation and validation"""
        # Valid input
        self.assertEqual(self.user_input.transaction_amount, 100.0)
        self.assertEqual(self.user_input.merchant_category, MerchantCategory.GROCERY)
        
        # Invalid amount
        with self.assertRaises(ValueError):
            UserTransactionInput(
                transaction_amount=-50.0,
                merchant_category=MerchantCategory.GROCERY,
                time_context=self.time_context,
                location_risk=LocationRisk.NORMAL,
                spending_pattern=SpendingPattern.TYPICAL
            )
    
    def test_enum_values(self):
        """Test enum value assignments"""
        self.assertEqual(MerchantCategory.GROCERY.value, "grocery")
        self.assertEqual(LocationRisk.NORMAL.value, "normal")
        self.assertEqual(SpendingPattern.TYPICAL.value, "typical")


class TestFeatureMappers(unittest.TestCase):
    """Test feature mapping models"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic training data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 8
        self.n_pca_components = 28
        
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.randn(self.n_samples, self.n_pca_components)
        
        self.X_test = np.random.randn(20, self.n_features)
        self.y_test = np.random.randn(20, self.n_pca_components)
        
        self.user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
    
    def test_random_forest_mapper(self):
        """Test Random Forest mapper"""
        mapper = RandomForestMapper(n_estimators=10, random_state=42)
        
        # Test training
        mapper.fit(self.X_train, self.y_train)
        self.assertTrue(mapper.is_trained)
        
        # Test prediction
        predictions = mapper.predict(self.X_test)
        self.assertEqual(predictions.shape, (20, 28))
        
        # Test single prediction
        single_pred = mapper.predict_single(self.user_input)
        self.assertEqual(single_pred.shape, (28,))
        
        # Test feature importance
        importance = mapper.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 8)
    
    def test_xgboost_mapper(self):
        """Test XGBoost mapper"""
        mapper = XGBoostMapper(n_estimators=10, random_state=42)
        
        # Test training
        mapper.fit(self.X_train, self.y_train)
        self.assertTrue(mapper.is_trained)
        
        # Test prediction
        predictions = mapper.predict(self.X_test)
        self.assertEqual(predictions.shape, (20, 28))
        
        # Test single prediction
        single_pred = mapper.predict_single(self.user_input)
        self.assertEqual(single_pred.shape, (28,))
        
        # Test feature importance
        importance = mapper.get_feature_importance()
        self.assertIsInstance(importance, dict)
    
    def test_mapper_validation(self):
        """Test mapper validation methods"""
        mapper = RandomForestMapper(n_estimators=10, random_state=42)
        mapper.fit(self.X_train, self.y_train)
        
        # Test validation
        metrics = mapper.validate_predictions(self.X_test, self.y_test)
        self.assertIn('overall_mse', metrics)
        self.assertIn('overall_mae', metrics)
        self.assertIn('avg_correlation', metrics)
        
        # Test evaluation
        eval_metrics = mapper.evaluate_mapping_quality(self.X_test, self.y_test)
        self.assertIn('mse', eval_metrics)
        self.assertIn('mae', eval_metrics)


class TestFeatureAssembler(unittest.TestCase):
    """Test feature vector assembler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.assembler = FeatureVectorAssembler()
        
        self.user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
        
        self.mapped_pca = np.random.randn(28)
    
    def test_feature_vector_assembly(self):
        """Test feature vector assembly"""
        feature_vector = self.assembler.assemble_feature_vector(
            self.user_input, self.mapped_pca
        )
        
        # Should have 30 features: Time, V1-V28, Amount
        self.assertEqual(feature_vector.shape, (30,))
        
        # Check that Amount is in the right place (last position)
        self.assertEqual(feature_vector[29], 100.0)
    
    def test_feature_bounds_validation(self):
        """Test feature bounds validation"""
        feature_vector = np.random.randn(30)
        
        # Without bounds, should pass
        is_valid, messages = self.assembler.validate_feature_bounds(feature_vector)
        self.assertTrue(is_valid)
        self.assertEqual(len(messages), 0)
    
    def test_statistical_corrections(self):
        """Test statistical corrections"""
        # Create extreme feature vector
        feature_vector = np.ones(30) * 1000  # Very large values
        
        # Without bounds, should return unchanged
        corrected = self.assembler.apply_statistical_corrections(feature_vector)
        np.testing.assert_array_equal(feature_vector, corrected)
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        feature_vector = np.random.randn(30)
        
        quality_metrics = self.assembler.calculate_quality_metrics(
            feature_vector, self.mapped_pca
        )
        
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertGreaterEqual(quality_metrics.confidence_score, 0.0)
        self.assertLessEqual(quality_metrics.confidence_score, 1.0)


class TestInputValidator(unittest.TestCase):
    """Test input validator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = InputValidator()
        
        self.valid_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
    
    def test_amount_validation(self):
        """Test transaction amount validation"""
        # Valid amount
        is_valid, errors, warnings = self.validator.validate_transaction_amount(100.0)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid amount (negative)
        is_valid, errors, warnings = self.validator.validate_transaction_amount(-50.0)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Very high amount
        is_valid, errors, warnings = self.validator.validate_transaction_amount(100000.0)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_time_validation(self):
        """Test time context validation"""
        # Valid time
        time_ctx = TimeContext(14, 2, False)
        is_valid, errors, warnings = self.validator.validate_time_context(time_ctx)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid hour
        time_ctx = TimeContext(25, 2, False)
        is_valid, errors, warnings = self.validator.validate_time_context(time_ctx)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_complete_validation(self):
        """Test complete input validation"""
        result = self.validator.validate_user_input(self.valid_input)
        
        self.assertTrue(result.is_valid)
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.warnings, list)
        self.assertIsInstance(result.suggestions, dict)
    
    def test_merchant_suggestions(self):
        """Test merchant category suggestions"""
        suggestions = self.validator.get_merchant_suggestions(50.0)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # All suggestions should be MerchantCategory enums
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, MerchantCategory)


class TestConfidenceScorer(unittest.TestCase):
    """Test confidence scoring system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scorer = ConfidenceScorer()
        
        self.user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
        
        self.mapped_features = np.random.randn(28)
    
    def test_input_completeness(self):
        """Test input completeness calculation"""
        score = self.scorer.calculate_input_completeness(self.user_input)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_input_consistency(self):
        """Test input consistency calculation"""
        score = self.scorer.calculate_input_consistency(self.user_input)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_overall_confidence(self):
        """Test overall confidence calculation"""
        confidence, factors = self.scorer.calculate_overall_confidence(
            self.user_input, self.mapped_features
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Check that all factors are present
        self.assertGreaterEqual(factors.input_completeness, 0.0)
        self.assertGreaterEqual(factors.input_consistency, 0.0)
        self.assertGreaterEqual(factors.model_uncertainty, 0.0)
    
    def test_confidence_explanation(self):
        """Test confidence explanation generation"""
        confidence, factors = self.scorer.calculate_overall_confidence(
            self.user_input, self.mapped_features
        )
        
        explanation = self.scorer.generate_confidence_explanation(confidence, factors)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('overall_confidence', explanation)
        self.assertIn('contributing_factors', explanation)
        self.assertIn('recommendations', explanation)


class TestMappingQualityMonitor(unittest.TestCase):
    """Test mapping quality monitor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MappingQualityMonitor()
        
        # Create synthetic data
        np.random.seed(42)
        self.original_features = np.random.randn(100, 28)
        self.mapped_features = self.original_features + np.random.randn(100, 28) * 0.1
    
    def test_correlation_preservation(self):
        """Test correlation preservation measurement"""
        score = self.monitor.measure_correlation_preservation(
            self.original_features, self.mapped_features
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_distribution_similarity(self):
        """Test distribution similarity measurement"""
        score = self.monitor.measure_distribution_similarity(
            self.original_features, self.mapped_features
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_mapping_uncertainty(self):
        """Test mapping uncertainty calculation"""
        uncertainty = self.monitor.calculate_mapping_uncertainty(self.mapped_features)
        
        self.assertGreaterEqual(uncertainty, 0.0)
        self.assertLessEqual(uncertainty, 1.0)
    
    def test_quality_assessment(self):
        """Test comprehensive quality assessment"""
        quality_metrics = self.monitor.assess_mapping_quality(
            self.original_features, self.mapped_features
        )
        
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertGreaterEqual(quality_metrics.confidence_score, 0.0)
        self.assertLessEqual(quality_metrics.confidence_score, 1.0)


class TestErrorHandler(unittest.TestCase):
    """Test error handling system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()
        
        self.user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
    
    def test_error_handling(self):
        """Test basic error handling"""
        test_error = ValueError("Test error")
        context = {'user_input': self.user_input}
        
        result = self.error_handler.handle_error(
            test_error, context, ErrorCategory.FEATURE_MAPPING, ErrorSeverity.MEDIUM
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('error_id', result)
    
    def test_fallback_strategies(self):
        """Test fallback strategies"""
        context = {'user_input': self.user_input}
        
        # Test conservative mapping fallback
        conservative_fallback = self.error_handler.fallback_strategies[0]
        result = conservative_fallback.execute(context)
        
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('fraud_probability', result)
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        # Generate some test errors
        for i in range(5):
            test_error = ValueError(f"Test error {i}")
            context = {'user_input': self.user_input}
            self.error_handler.handle_error(test_error, context)
        
        stats = self.error_handler.get_error_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_errors', stats)
        self.assertIn('recovery_rate_percent', stats)
        self.assertEqual(stats['total_errors'], 5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_extreme_transaction_amounts(self):
        """Test handling of extreme transaction amounts"""
        validator = InputValidator()
        
        # Very small amount
        is_valid, errors, warnings = validator.validate_transaction_amount(0.01)
        self.assertTrue(is_valid)
        
        # Zero amount
        is_valid, errors, warnings = validator.validate_transaction_amount(0.0)
        self.assertFalse(is_valid)
        
        # Very large amount
        is_valid, errors, warnings = validator.validate_transaction_amount(1000000.0)
        self.assertFalse(is_valid)
    
    def test_boundary_time_values(self):
        """Test boundary time values"""
        # Midnight
        time_ctx = TimeContext(0, 0, False)
        self.assertEqual(time_ctx.hour_of_day, 0)
        
        # End of day
        time_ctx = TimeContext(23, 6, True)
        self.assertEqual(time_ctx.hour_of_day, 23)
        self.assertTrue(time_ctx.is_weekend)
    
    def test_empty_feature_vectors(self):
        """Test handling of empty or malformed feature vectors"""
        assembler = FeatureVectorAssembler()
        
        # Test with wrong size PCA vector
        user_input = UserTransactionInput(
            transaction_amount=100.0,
            merchant_category=MerchantCategory.GROCERY,
            time_context=TimeContext(14, 2, False),
            location_risk=LocationRisk.NORMAL,
            spending_pattern=SpendingPattern.TYPICAL
        )
        
        # Wrong size should raise error
        with self.assertRaises(ValueError):
            assembler.assemble_feature_vector(user_input, np.array([1, 2, 3]))  # Wrong size
    
    def test_model_with_no_training_data(self):
        """Test model behavior with no training data"""
        mapper = RandomForestMapper(n_estimators=5)
        
        # Should not be trained initially
        self.assertFalse(mapper.is_trained)
        
        # Prediction without training should raise error
        with self.assertRaises(ValueError):
            mapper.predict(np.random.randn(1, 8))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFeatureMappingEntities,
        TestFeatureMappers,
        TestFeatureAssembler,
        TestInputValidator,
        TestConfidenceScorer,
        TestMappingQualityMonitor,
        TestErrorHandler,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")