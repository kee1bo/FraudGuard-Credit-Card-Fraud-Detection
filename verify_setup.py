#!/usr/bin/env python3
"""
FraudGuard AI - Setup Verification Script
This script checks if the environment is properly configured for the project.
"""

import sys
import os
import importlib
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[1;34m",
        "SUCCESS": "\033[1;32m",
        "ERROR": "\033[1;31m",
        "WARNING": "\033[1;33m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{message}{reset}")

def check_python_version():
    """Check if Python 3.9 is being used"""
    print_status("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor == 9:
        print_status(f"✅ Python {version.major}.{version.minor}.{version.micro}", "SUCCESS")
        return True
    else:
        print_status(f"❌ Python {version.major}.{version.minor}.{version.micro} (Expected 3.9.x)", "ERROR")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_status("Checking dependencies...")
    
    required_packages = [
        "sklearn",
        "pandas", 
        "numpy",
        "flask",
        "matplotlib",
        "joblib",
        "yaml",
        "xgboost",
        "lightgbm",
        "catboost",
        "shap",
        "lime",
        "plotly",
        "imblearn"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_status(f"✅ {package}", "SUCCESS")
        except ImportError:
            print_status(f"❌ {package}", "ERROR")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def check_project_structure():
    """Check if project structure is correct"""
    print_status("Checking project structure...")
    
    required_paths = [
        "src/fraudguard",
        "app",
        "config.yaml",
        "main.py",
        "run_app.py",
        "requirements.txt"
    ]
    
    missing_paths = []
    
    for path in required_paths:
        if Path(path).exists():
            print_status(f"✅ {path}", "SUCCESS")
        else:
            print_status(f"❌ {path}", "ERROR")
            missing_paths.append(path)
    
    return len(missing_paths) == 0

def check_dataset():
    """Check if dataset is available"""
    print_status("Checking dataset...")
    
    dataset_path = Path("data/creditcard.csv")
    
    if dataset_path.exists():
        file_size = dataset_path.stat().st_size / (1024 * 1024)  # MB
        print_status(f"✅ Dataset found ({file_size:.1f} MB)", "SUCCESS")
        return True
    else:
        print_status("❌ Dataset not found at data/creditcard.csv", "WARNING")
        print_status("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "INFO")
        return False

def check_directories():
    """Check if necessary directories exist"""
    print_status("Checking directories...")
    
    required_dirs = [
        "data",
        "logs", 
        "artifacts",
        "artifacts/models",
        "artifacts/preprocessors",
        "artifacts/explainers",
        "artifacts/reports"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_status(f"✅ Created {dir_path}", "SUCCESS")
        else:
            print_status(f"✅ {dir_path}", "SUCCESS")
    
    return True

def test_fraudguard_import():
    """Test importing the fraudguard package"""
    print_status("Testing FraudGuard package import...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path("src")))
        
        # Test imports
        from fraudguard.logger import fraud_logger
        from fraudguard.constants.constants import AVAILABLE_MODELS
        from fraudguard.models.model_factory import ModelFactory
        
        print_status("✅ FraudGuard package imports successful", "SUCCESS")
        print_status(f"Available models: {AVAILABLE_MODELS}", "INFO")
        return True
        
    except ImportError as e:
        print_status(f"❌ FraudGuard package import failed: {e}", "ERROR")
        return False

def main():
    """Main verification function"""
    print_status("🎓 FraudGuard AI - Setup Verification", "INFO")
    print_status("=" * 50, "INFO")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Directories", check_directories),
        ("Dataset", check_dataset),
        ("Package Import", test_fraudguard_import)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print_status(f"\n{check_name}:", "INFO")
        result = check_func()
        results.append((check_name, result))
    
    print_status("\n" + "=" * 50, "INFO")
    print_status("VERIFICATION SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        color = "SUCCESS" if result else "ERROR"
        print_status(f"{check_name:.<30} {status}", color)
        if not result:
            all_passed = False
    
    print_status("\n" + "=" * 50, "INFO")
    
    if all_passed:
        print_status("🎉 ALL CHECKS PASSED!", "SUCCESS")
        print_status("Your environment is ready for FraudGuard AI!", "SUCCESS")
        print_status("\nNext steps:", "INFO")
        print_status("1. Run training: python main.py", "INFO")
        print_status("2. Start web app: python run_app.py", "INFO")
    else:
        print_status("⚠️  SOME CHECKS FAILED", "WARNING")
        print_status("Please address the issues above before proceeding.", "WARNING")
        print_status("See README.md for detailed setup instructions.", "INFO")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())