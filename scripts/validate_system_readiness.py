#!/usr/bin/env python3
"""
System Readiness Validation Script
Quick validation to ensure the intelligent feature mapping system is ready for testing.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import time
import json
from typing import Dict, Any, List
import numpy as np

from fraudguard.entity.feature_mapping_entity import UserTransactionInput
from fraudguard.logger import fraud_logger


class SystemReadinessValidator:
    """Validates system readiness for comprehensive testing"""
    
    def __init__(self):
        self.validation_results = {
            'overall_ready': False,
            'checks': {},
            'missing_components': [],
            'recommendations': []
        }
    
    def check_directory_structure(self) -> bool:
        """Check if required directory structure exists"""
        required_dirs = [
            "src/fraudguard",
            "src/fraudguard/models",
            "src/fraudguard/components",
            "src/fraudguard/pipeline",
            "src/fraudguard/entity",
            "src/fraudguard/mlops",
            "artifacts",
            "tests"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        self.validation_results['checks']['directory_structure'] = {
            'passed': len(missing_dirs) == 0,
            'missing_directories': missing_dirs
        }
        
        if missing_dirs:
            self.validation_results['missing_components'].extend(missing_dirs)
        
        return len(missing_dirs) == 0
    
    def check_core_modules(self) -> bool:
        """Check if core modules can be imported"""
        core_modules = [
            ('fraudguard.entity.feature_mapping_entity', 'UserTransactionInput'),
            ('fraudguard.models.base_feature_mapper', 'BaseFeatureMapper'),
            ('fraudguard.models.random_forest_mapper', 'RandomForestMapper'),
            ('fraudguard.components.feature_assembler', 'FeatureVectorAssembler'),
            ('fraudguard.pipeline.intelligent_prediction_pipeline', 'IntelligentPredictionPipeline'),
            ('fraudguard.logger', 'fraud_logger')
        ]
        
        import_errors = []
        
        for module_name, class_name in core_modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
            except ImportError as e:
                import_errors.append(f"{module_name}.{class_name}: {e}")
            except AttributeError as e:
                import_errors.append(f"{module_name}.{class_name}: {e}")
        
        self.validation_results['checks']['core_modules'] = {
            'passed': len(import_errors) == 0,
            'import_errors': import_errors
        }
        
        if import_errors:
            self.validation_results['missing_components'].extend(import_errors)
        
        return len(import_errors) == 0
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        required_packages = [
            'numpy',
            'pandas',
            'scikit-learn',
            'joblib',
            'flask'
        ]
        
        optional_packages = [
            'xgboost',
            'tensorflow',
            'shap',
            'redis'
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)
        
        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)
        
        self.validation_results['checks']['dependencies'] = {
            'passed': len(missing_required) == 0,
            'missing_required': missing_required,
            'missing_optional': missing_optional
        }
        
        if missing_required:
            self.validation_results['missing_components'].extend(missing_required)
        
        return len(missing_required) == 0
    
    def check_basic_functionality(self) -> bool:
        """Check if basic functionality works"""
        try:
            # Test basic entity creation
            test_input = UserTransactionInput(
                transaction_amount=100.0,
                merchant_category="grocery_store",
                location_risk_score=0.1,
                time_since_last_transaction=2.0,
                spending_pattern_score=0.3
            )
            
            # Test interpretable features conversion
            interpretable_features = test_input.to_interpretable_features()
            
            if len(interpretable_features) != 5:
                raise ValueError(f"Expected 5 interpretable features, got {len(interpretable_features)}")
            
            self.validation_results['checks']['basic_functionality'] = {
                'passed': True,
                'test_input_created': True,
                'interpretable_features_count': len(interpretable_features)
            }
            
            return True
            
        except Exception as e:
            self.validation_results['checks']['basic_functionality'] = {
                'passed': False,
                'error': str(e)
            }
            
            self.validation_results['missing_components'].append(f"Basic functionality: {e}")
            return False
    
    def check_model_artifacts(self) -> bool:
        """Check if model artifacts directory structure exists"""
        artifact_paths = [
            "artifacts/models",
            "artifacts/feature_mapping",
            "artifacts/feature_mapping/mappers"
        ]
        
        missing_artifacts = []
        for path in artifact_paths:
            if not Path(path).exists():
                missing_artifacts.append(path)
        
        # Check for any existing models
        existing_models = []
        models_dir = Path("artifacts/models")
        if models_dir.exists():
            existing_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        mappers_dir = Path("artifacts/feature_mapping/mappers")
        existing_mappers = []
        if mappers_dir.exists():
            existing_mappers = [d.name for d in mappers_dir.iterdir() if d.is_dir()]
        
        self.validation_results['checks']['model_artifacts'] = {
            'passed': len(missing_artifacts) == 0,
            'missing_artifact_dirs': missing_artifacts,
            'existing_models': existing_models,
            'existing_mappers': existing_mappers
        }
        
        # This is not a hard requirement - models can be created during testing
        return True
    
    def check_test_files(self) -> bool:
        """Check if test files exist"""
        test_files = [
            "tests/test_feature_mapping.py",
            "tests/test_integration_performance.py",
            "tests/test_mapping_performance.py"
        ]
        
        missing_tests = []
        for test_file in test_files:
            if not Path(test_file).exists():
                missing_tests.append(test_file)
        
        self.validation_results['checks']['test_files'] = {
            'passed': len(missing_tests) == 0,
            'missing_test_files': missing_tests
        }
        
        if missing_tests:
            self.validation_results['missing_components'].extend(missing_tests)
        
        return len(missing_tests) == 0
    
    def check_dataset_availability(self) -> bool:
        """Check if ULB dataset is available"""
        dataset_path = Path("data/creditcard.csv")
        
        dataset_available = dataset_path.exists()
        dataset_size = 0
        
        if dataset_available:
            try:
                dataset_size = dataset_path.stat().st_size / (1024 * 1024)  # MB
            except Exception:
                dataset_size = 0
        
        self.validation_results['checks']['dataset'] = {
            'passed': True,  # Not required for basic testing
            'dataset_available': dataset_available,
            'dataset_size_mb': dataset_size,
            'dataset_path': str(dataset_path)
        }
        
        return True  # Not a hard requirement
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        print("🔍 Validating System Readiness for Intelligent Feature Mapping...")
        print("=" * 70)
        
        validation_checks = [
            ("Directory Structure", self.check_directory_structure),
            ("Core Modules", self.check_core_modules),
            ("Dependencies", self.check_dependencies),
            ("Basic Functionality", self.check_basic_functionality),
            ("Model Artifacts", self.check_model_artifacts),
            ("Test Files", self.check_test_files),
            ("Dataset Availability", self.check_dataset_availability)
        ]
        
        passed_checks = 0
        total_checks = len(validation_checks)
        
        for check_name, check_function in validation_checks:
            print(f"\\n🔎 Checking {check_name}...")
            
            try:
                result = check_function()
                if result:
                    print(f"  ✅ {check_name}: PASSED")
                    passed_checks += 1
                else:
                    print(f"  ❌ {check_name}: FAILED")
            except Exception as e:
                print(f"  ❌ {check_name}: ERROR - {e}")
                self.validation_results['checks'][check_name.lower().replace(' ', '_')] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Determine overall readiness
        critical_checks = ['directory_structure', 'core_modules', 'dependencies', 'basic_functionality', 'test_files']
        critical_passed = sum(1 for check in critical_checks if self.validation_results['checks'].get(check, {}).get('passed', False))
        
        self.validation_results['overall_ready'] = critical_passed == len(critical_checks)
        self.validation_results['validation_summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'critical_checks_passed': critical_passed,
            'critical_checks_total': len(critical_checks)
        }
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Print summary
        print("\\n" + "=" * 70)
        print("📋 VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Overall Readiness: {'✅ READY' if self.validation_results['overall_ready'] else '❌ NOT READY'}")
        print(f"Checks Passed: {passed_checks}/{total_checks}")
        print(f"Critical Checks: {critical_passed}/{len(critical_checks)}")
        
        if not self.validation_results['overall_ready']:
            print("\\n⚠️  Issues Found:")
            for component in self.validation_results['missing_components'][:5]:  # Show first 5
                print(f"  - {component}")
            if len(self.validation_results['missing_components']) > 5:
                print(f"  ... and {len(self.validation_results['missing_components']) - 5} more")
        
        print("\\n💡 Recommendations:")
        for rec in self.validation_results['recommendations']:
            print(f"  - {rec}")
        
        return self.validation_results
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results"""
        if self.validation_results['overall_ready']:
            self.validation_results['recommendations'].append("System is ready for comprehensive testing")
            self.validation_results['recommendations'].append("Run: python3 scripts/run_comprehensive_tests.py")
        else:
            self.validation_results['recommendations'].append("Fix missing components before running tests")
            
            # Specific recommendations based on failed checks
            if not self.validation_results['checks'].get('dependencies', {}).get('passed', True):
                missing_deps = self.validation_results['checks']['dependencies'].get('missing_required', [])
                if missing_deps:
                    self.validation_results['recommendations'].append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
            
            if not self.validation_results['checks'].get('core_modules', {}).get('passed', True):
                self.validation_results['recommendations'].append("Check Python path and module imports")
            
            if not self.validation_results['checks'].get('directory_structure', {}).get('passed', True):
                self.validation_results['recommendations'].append("Ensure all source code directories are present")
        
        # Dataset recommendations
        if not self.validation_results['checks'].get('dataset', {}).get('dataset_available', False):
            self.validation_results['recommendations'].append("Download ULB dataset for full accuracy testing (optional)")
            self.validation_results['recommendations'].append("Dataset URL: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        
        # Model recommendations
        existing_models = self.validation_results['checks'].get('model_artifacts', {}).get('existing_models', [])
        existing_mappers = self.validation_results['checks'].get('model_artifacts', {}).get('existing_mappers', [])
        
        if not existing_models and not existing_mappers:
            self.validation_results['recommendations'].append("Run model training before testing: python3 scripts/setup_intelligent_system.py")
    
    def save_validation_results(self, output_path: str = "test_results/system_validation.json"):
        """Save validation results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            
            print(f"\\n📁 Validation results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to save validation results: {e}")


def main():
    """Main validation function"""
    try:
        validator = SystemReadinessValidator()
        results = validator.run_comprehensive_validation()
        validator.save_validation_results()
        
        # Return appropriate exit code
        return 0 if results['overall_ready'] else 1
        
    except KeyboardInterrupt:
        print("\\n⚠️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())