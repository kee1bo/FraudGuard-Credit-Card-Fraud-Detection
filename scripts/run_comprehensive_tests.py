#!/usr/bin/env python3
"""
Comprehensive Test Runner for Intelligent Feature Mapping System
Orchestrates all integration, performance, and accuracy tests.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any

# Import test suites
from tests.test_integration_performance import IntegrationTestSuite
from tests.test_mapping_performance import MappingPerformanceBenchmark
from tests.test_feature_mapping import FeatureMappingTestSuite
from fraudguard.logger import fraud_logger


class ComprehensiveTestRunner:
    """Orchestrates all test suites for the intelligent mapping system"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {
            'test_run_info': {
                'start_time': datetime.now().isoformat(),
                'test_runner_version': '1.0.0',
                'system_info': self._get_system_info()
            },
            'test_suites': {},
            'overall_summary': {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for test context"""
        try:
            import platform
            import psutil
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_space_gb': round(psutil.disk_usage('/').total / (1024**3), 2)
            }
        except Exception as e:
            return {'error': f'Failed to get system info: {e}'}
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for feature mapping components"""
        print("🧪 Running Unit Tests...")
        print("-" * 50)
        
        try:
            test_suite = FeatureMappingTestSuite()
            results = test_suite.run_all_tests()
            
            self.test_results['test_suites']['unit_tests'] = results
            
            print(f"✅ Unit tests completed: {results.get('tests_passed', 0)}/{results.get('total_tests', 0)} passed")
            return results
            
        except Exception as e:
            error_result = {'error': str(e), 'status': 'FAILED'}
            self.test_results['test_suites']['unit_tests'] = error_result
            print(f"❌ Unit tests failed: {e}")
            return error_result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for end-to-end pipeline"""
        print("\\n🔗 Running Integration Tests...")
        print("-" * 50)
        
        try:
            test_suite = IntegrationTestSuite()
            results = test_suite.run_comprehensive_test_suite()
            
            self.test_results['test_suites']['integration_tests'] = results
            
            status = results.get('overall_status', 'UNKNOWN')
            print(f"✅ Integration tests completed: {status}")
            
            if 'test_categories' in results:
                for category, info in results['test_categories'].items():
                    print(f"  - {category.title()}: {info.get('status', 'UNKNOWN')}")
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e), 'status': 'FAILED'}
            self.test_results['test_suites']['integration_tests'] = error_result
            print(f"❌ Integration tests failed: {e}")
            return error_result
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("\\n⚡ Running Performance Benchmarks...")
        print("-" * 50)
        
        try:
            benchmark = MappingPerformanceBenchmark()
            results = benchmark.run_comprehensive_benchmark()
            
            self.test_results['test_suites']['performance_benchmarks'] = results
            
            print("✅ Performance benchmarks completed")
            
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                if 'best_single_prediction_time_ms' in metrics:
                    print(f"  - Best single prediction time: {metrics['best_single_prediction_time_ms']:.2f}ms")
                if 'max_throughput_per_second' in metrics:
                    print(f"  - Max throughput: {metrics['max_throughput_per_second']:.1f} pred/s")
                if 'total_memory_usage_mb' in metrics:
                    print(f"  - Memory usage: {metrics['total_memory_usage_mb']:.1f}MB")
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e), 'status': 'FAILED'}
            self.test_results['test_suites']['performance_benchmarks'] = error_result
            print(f"❌ Performance benchmarks failed: {e}")
            return error_result
    
    def run_load_tests(self, concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run load tests with concurrent requests"""
        print(f"\\n🚀 Running Load Tests ({concurrent_requests} concurrent requests)...")
        print("-" * 50)
        
        try:
            test_suite = IntegrationTestSuite()
            test_suite.setup_test_environment()
            results = test_suite.test_concurrent_load(concurrent_requests)
            
            self.test_results['test_suites']['load_tests'] = results
            
            success_rate = results.get('success_rate', 0)
            rps = results.get('requests_per_second', 0)
            
            print(f"✅ Load tests completed: {success_rate:.1f}% success rate, {rps:.2f} RPS")
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e), 'status': 'FAILED'}
            self.test_results['test_suites']['load_tests'] = error_result
            print(f"❌ Load tests failed: {e}")
            return error_result
    
    def run_accuracy_validation(self) -> Dict[str, Any]:
        """Run accuracy validation against reference data"""
        print("\\n🎯 Running Accuracy Validation...")
        print("-" * 50)
        
        try:
            test_suite = IntegrationTestSuite()
            test_suite.setup_test_environment()
            results = test_suite.test_mapping_accuracy()
            
            self.test_results['test_suites']['accuracy_validation'] = results
            
            if 'overall_accuracy_score' in results:
                accuracy = results['overall_accuracy_score']
                tests_performed = results.get('tests_performed', 0)
                print(f"✅ Accuracy validation completed: {accuracy:.3f} correlation score ({tests_performed} tests)")
            else:
                print("⚠️  Accuracy validation completed with limited data")
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e), 'status': 'FAILED'}
            self.test_results['test_suites']['accuracy_validation'] = error_result
            print(f"❌ Accuracy validation failed: {e}")
            return error_result
    
    def generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall test summary and recommendations"""
        summary = {
            'overall_status': 'PASS',
            'test_suite_results': {},
            'performance_summary': {},
            'recommendations': [],
            'critical_issues': [],
            'test_coverage': {}
        }
        
        # Analyze each test suite
        total_suites = 0
        passed_suites = 0
        
        for suite_name, suite_results in self.test_results['test_suites'].items():
            total_suites += 1
            
            if isinstance(suite_results, dict) and 'error' not in suite_results:
                # Determine suite status
                suite_status = 'PASS'
                
                if suite_name == 'integration_tests':
                    suite_status = suite_results.get('overall_status', 'UNKNOWN')
                elif suite_name == 'unit_tests':
                    total_tests = suite_results.get('total_tests', 0)
                    passed_tests = suite_results.get('tests_passed', 0)
                    suite_status = 'PASS' if passed_tests == total_tests else 'FAIL'
                elif suite_name == 'load_tests':
                    success_rate = suite_results.get('success_rate', 0)
                    suite_status = 'PASS' if success_rate >= 95 else 'FAIL'
                elif suite_name == 'accuracy_validation':
                    accuracy = suite_results.get('overall_accuracy_score', 0)
                    suite_status = 'PASS' if accuracy >= 0.7 else 'FAIL'
                
                if suite_status == 'PASS':
                    passed_suites += 1
                else:
                    summary['overall_status'] = 'FAIL'
                
                summary['test_suite_results'][suite_name] = suite_status
            else:
                summary['test_suite_results'][suite_name] = 'FAIL'
                summary['overall_status'] = 'FAIL'
                if isinstance(suite_results, dict) and 'error' in suite_results:
                    summary['critical_issues'].append(f"{suite_name}: {suite_results['error']}")
        
        # Performance summary
        if 'performance_benchmarks' in self.test_results['test_suites']:
            perf_results = self.test_results['test_suites']['performance_benchmarks']
            if 'performance_metrics' in perf_results:
                summary['performance_summary'] = perf_results['performance_metrics']
        
        # Test coverage
        summary['test_coverage'] = {
            'total_test_suites': total_suites,
            'passed_test_suites': passed_suites,
            'coverage_percentage': (passed_suites / total_suites * 100) if total_suites > 0 else 0
        }
        
        # Generate recommendations
        if summary['overall_status'] == 'PASS':
            summary['recommendations'].append('✅ All test suites passed - system ready for production deployment')
            summary['recommendations'].append('🔄 Consider setting up automated testing in CI/CD pipeline')
            summary['recommendations'].append('📊 Monitor performance metrics in production environment')
        else:
            summary['recommendations'].append('❌ Some test suites failed - review issues before deployment')
            summary['recommendations'].append('🔧 Address critical issues identified in test results')
            summary['recommendations'].append('🧪 Re-run tests after fixes are implemented')
        
        # Performance-specific recommendations
        if 'best_single_prediction_time_ms' in summary.get('performance_summary', {}):
            pred_time = summary['performance_summary']['best_single_prediction_time_ms']
            if pred_time > 50:
                summary['recommendations'].append(f'⚠️  Prediction time ({pred_time:.2f}ms) exceeds target (50ms)')
            else:
                summary['recommendations'].append(f'✅ Prediction time ({pred_time:.2f}ms) meets performance requirements')
        
        self.test_results['overall_summary'] = summary
        return summary
    
    def save_test_results(self):
        """Save comprehensive test results to files"""
        try:
            # Save main results
            main_results_file = self.output_dir / "comprehensive_test_results.json"
            with open(main_results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            # Save summary report
            summary_file = self.output_dir / "test_summary_report.json"
            with open(summary_file, 'w') as f:
                json.dump(self.test_results['overall_summary'], f, indent=2, default=str)
            
            # Generate human-readable report
            self._generate_html_report()
            
            print(f"\\n📁 Test results saved to: {self.output_dir}")
            print(f"  - Main results: {main_results_file}")
            print(f"  - Summary: {summary_file}")
            print(f"  - HTML report: {self.output_dir / 'test_report.html'}")
            
        except Exception as e:
            print(f"❌ Failed to save test results: {e}")
    
    def _generate_html_report(self):
        """Generate HTML test report"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Intelligent Feature Mapping - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .recommendation {{ margin: 5px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007cba; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Intelligent Feature Mapping System - Test Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Overall Status:</strong> <span class="{self.test_results['overall_summary']['overall_status'].lower()}">{self.test_results['overall_summary']['overall_status']}</span></p>
    </div>
    
    <div class="section">
        <h2>Test Suite Results</h2>
        {self._generate_test_suite_html()}
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        {self._generate_performance_html()}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {self._generate_recommendations_html()}
    </div>
    
    <div class="section">
        <h2>Test Coverage</h2>
        {self._generate_coverage_html()}
    </div>
</body>
</html>
"""
            
            html_file = self.output_dir / "test_report.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            print(f"Warning: Failed to generate HTML report: {e}")
    
    def _generate_test_suite_html(self) -> str:
        """Generate HTML for test suite results"""
        html = ""
        for suite_name, status in self.test_results['overall_summary']['test_suite_results'].items():
            status_class = status.lower()
            html += f'<div class="metric"><strong>{suite_name.replace("_", " ").title()}:</strong> <span class="{status_class}">{status}</span></div>'
        return html
    
    def _generate_performance_html(self) -> str:
        """Generate HTML for performance metrics"""
        perf_summary = self.test_results['overall_summary'].get('performance_summary', {})
        if not perf_summary:
            return "<p>No performance metrics available</p>"
        
        html = ""
        for metric, value in perf_summary.items():
            if isinstance(value, (int, float)):
                html += f'<div class="metric"><strong>{metric.replace("_", " ").title()}:</strong> {value:.2f}</div>'
            else:
                html += f'<div class="metric"><strong>{metric.replace("_", " ").title()}:</strong> {value}</div>'
        return html
    
    def _generate_recommendations_html(self) -> str:
        """Generate HTML for recommendations"""
        recommendations = self.test_results['overall_summary'].get('recommendations', [])
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        return html
    
    def _generate_coverage_html(self) -> str:
        """Generate HTML for test coverage"""
        coverage = self.test_results['overall_summary'].get('test_coverage', {})
        html = f"""
        <div class="metric"><strong>Total Test Suites:</strong> {coverage.get('total_test_suites', 0)}</div>
        <div class="metric"><strong>Passed Test Suites:</strong> {coverage.get('passed_test_suites', 0)}</div>
        <div class="metric"><strong>Coverage Percentage:</strong> {coverage.get('coverage_percentage', 0):.1f}%</div>
        """
        return html
    
    def run_comprehensive_test_suite(self, 
                                   run_unit: bool = True,
                                   run_integration: bool = True,
                                   run_performance: bool = True,
                                   run_load: bool = True,
                                   run_accuracy: bool = True,
                                   concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run comprehensive test suite with all categories"""
        
        print("🎯 Starting Comprehensive Test Suite for Intelligent Feature Mapping System")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run test categories based on parameters
        if run_unit:
            self.run_unit_tests()
        
        if run_integration:
            self.run_integration_tests()
        
        if run_performance:
            self.run_performance_benchmarks()
        
        if run_load:
            self.run_load_tests(concurrent_requests)
        
        if run_accuracy:
            self.run_accuracy_validation()
        
        # Generate overall summary
        summary = self.generate_overall_summary()
        
        # Calculate total time
        total_time = time.time() - start_time
        self.test_results['test_run_info']['end_time'] = datetime.now().isoformat()
        self.test_results['test_run_info']['total_duration_seconds'] = total_time
        
        # Save results
        self.save_test_results()
        
        # Print final summary
        print("\\n" + "=" * 80)
        print("🏁 COMPREHENSIVE TEST SUITE COMPLETED")
        print("=" * 80)
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📊 Overall status: {summary['overall_status']}")
        print(f"📈 Test coverage: {summary['test_coverage']['coverage_percentage']:.1f}%")
        
        if summary['overall_status'] == 'PASS':
            print("\\n🎉 All tests passed! System is ready for production deployment.")
        else:
            print("\\n⚠️  Some tests failed. Please review the issues before deployment.")
        
        print(f"\\n📁 Detailed results saved to: {self.output_dir}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive tests for Intelligent Feature Mapping System')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for test results')
    parser.add_argument('--skip-unit', action='store_true', help='Skip unit tests')
    parser.add_argument('--skip-integration', action='store_true', help='Skip integration tests')
    parser.add_argument('--skip-performance', action='store_true', help='Skip performance benchmarks')
    parser.add_argument('--skip-load', action='store_true', help='Skip load tests')
    parser.add_argument('--skip-accuracy', action='store_true', help='Skip accuracy validation')
    parser.add_argument('--concurrent-requests', type=int, default=10, help='Number of concurrent requests for load testing')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (reduced load testing)')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick testing
    if args.quick:
        args.concurrent_requests = min(args.concurrent_requests, 5)
    
    try:
        # Initialize test runner
        test_runner = ComprehensiveTestRunner(output_dir=args.output_dir)
        
        # Run comprehensive test suite
        summary = test_runner.run_comprehensive_test_suite(
            run_unit=not args.skip_unit,
            run_integration=not args.skip_integration,
            run_performance=not args.skip_performance,
            run_load=not args.skip_load,
            run_accuracy=not args.skip_accuracy,
            concurrent_requests=args.concurrent_requests
        )
        
        # Exit with appropriate code
        exit_code = 0 if summary['overall_status'] == 'PASS' else 1
        return exit_code
        
    except KeyboardInterrupt:
        print("\\n⚠️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())