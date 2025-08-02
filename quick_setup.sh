#!/bin/bash

# FraudGuard AI - Quick Setup Script
# This script automates the Python 3.9 environment setup

set -e  # Exit on any error

echo "🎓 FraudGuard AI - MSc Data Science Project Setup"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check if Python 3.9 is available
print_status "Checking Python 3.9 availability..."
if command -v python3.9 &> /dev/null; then
    PYTHON_VERSION=$(python3.9 --version | cut -d' ' -f2)
    print_success "Found Python $PYTHON_VERSION"
else
    print_error "Python 3.9 not found!"
    echo ""
    echo "Please install Python 3.9 first:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install software-properties-common"
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "  sudo apt update"
    echo "  sudo apt install python3.9 python3.9-venv python3.9-dev"
    echo ""
    echo "macOS (with Homebrew):"
    echo "  brew install python@3.9"
    echo ""
    echo "Windows:"
    echo "  Download from https://www.python.org/downloads/release/python-3918/"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -f "main.py" ]; then
    print_error "Please run this script from the FraudGuard project root directory"
    exit 1
fi

# Create virtual environment
print_status "Creating Python 3.9 virtual environment..."
if [ -d "venv_fraudguard" ]; then
    print_warning "Virtual environment already exists. Removing old one..."
    rm -rf venv_fraudguard
fi

python3.9 -m venv venv_fraudguard
print_success "Virtual environment created: venv_fraudguard"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv_fraudguard/bin/activate

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python --version | cut -d' ' -f2)
if [[ $VENV_PYTHON_VERSION == 3.9* ]]; then
    print_success "Virtual environment using Python $VENV_PYTHON_VERSION"
else
    print_error "Virtual environment not using Python 3.9. Got: $VENV_PYTHON_VERSION"
    exit 1
fi

# Upgrade pip and install build tools
print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install dependencies
print_status "Installing project dependencies..."
pip install -r requirements.txt

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install project in development mode
print_status "Installing FraudGuard package in development mode..."
pip install -e .

# Verify installation
print_status "Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import sklearn
    import pandas
    import numpy
    import flask
    import shap
    import lime
    import matplotlib
    import xgboost
    import lightgbm
    print('✅ All core dependencies imported successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data logs artifacts/{models,preprocessors,explainers,reports}
print_success "Project directories created"

# Check for dataset
print_status "Checking for dataset..."
if [ ! -f "data/creditcard.csv" ]; then
    print_warning "Dataset not found at data/creditcard.csv"
    echo ""
    echo "Please download the ULB Credit Card Fraud dataset:"
    echo "1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo "2. Download creditcard.csv"
    echo "3. Place it in the data/ folder"
    echo ""
    echo "Or use Kaggle API:"
    echo "  pip install kaggle"
    echo "  kaggle datasets download -d mlg-ulb/creditcardfraud"
    echo "  unzip creditcardfraud.zip -d data/"
else
    print_success "Dataset found at data/creditcard.csv"
fi

echo ""
print_success "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv_fraudguard/bin/activate"
echo ""
echo "2. Download the dataset (if not already done):"
echo "   Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
echo "   Place creditcard.csv in the data/ folder"
echo ""
echo "3. Run the training pipeline:"
echo "   python main.py"
echo ""
echo "4. Start the web application:"
echo "   python run_app.py"
echo ""
echo "For more information, see README.md"