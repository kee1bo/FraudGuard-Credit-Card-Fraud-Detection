#!/bin/bash

echo "==============================================="
echo "    FraudGuard AI - macOS Setup Script"
echo "==============================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Use setup_linux.sh for Linux systems."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install Homebrew"
        exit 1
    fi
    print_success "Homebrew installed successfully"
fi

# Check if Python 3.9 is installed
if ! command -v python3.9 &> /dev/null; then
    print_warning "Python 3.9 not found. Installing via Homebrew..."
    brew install python@3.9
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install Python 3.9"
        exit 1
    fi
    print_success "Python 3.9 installed successfully"
fi

# Check Python version
PYTHON_VERSION=$(python3.9 --version 2>&1 | awk '{print $2}')
print_status "Found Python version: $PYTHON_VERSION"

# Verify Python 3.9
if [[ $PYTHON_VERSION != 3.9* ]]; then
    print_warning "Python 3.9.x is recommended for best compatibility"
    print_warning "Current version: $PYTHON_VERSION"
    read -p "Continue anyway? (y/N): " continue_setup
    if [[ $continue_setup != [Yy] ]]; then
        exit 1
    fi
fi

echo
print_status "Step 1: Creating virtual environment..."
python3.9 -m venv venv_fraudguard

if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment"
    exit 1
fi
print_success "Virtual environment created"

print_status "Step 2: Activating virtual environment..."
source venv_fraudguard/bin/activate

if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated"

print_status "Step 3: Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

if [ $? -ne 0 ]; then
    print_error "Failed to upgrade pip and setuptools"
    exit 1
fi
print_success "Pip and setuptools upgraded"

print_status "Step 4: Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    print_error "This might be due to:"
    print_error "- Network connectivity issues"
    print_error "- Missing system dependencies"
    print_error "- Insufficient permissions"
    exit 1
fi
print_success "Dependencies installed"

print_status "Step 5: Installing FraudGuard package..."
pip install -e .

if [ $? -ne 0 ]; then
    print_error "Failed to install FraudGuard package"
    exit 1
fi
print_success "FraudGuard package installed"

print_status "Step 6: Verifying installation..."
python verify_setup.py

if [ $? -ne 0 ]; then
    print_warning "Setup verification failed"
    print_warning "The installation may be incomplete"
else
    print_success "Installation verified successfully"
fi

echo
echo "==============================================="
echo "           Setup Complete!"
echo "==============================================="
echo
echo "To start using FraudGuard:"
echo "1. Activate the environment: source venv_fraudguard/bin/activate"
echo "2. Run the application: python run_app.py"
echo "3. Open your browser to: http://localhost:5000"
echo
echo "For training models: python main.py"
echo "For help: python run_app.py --help"
echo

# Ask if user wants to start the application now
read -p "Start the application now? (y/N): " start_now
if [[ $start_now == [Yy] ]]; then
    print_status "Starting FraudGuard..."
    python run_app.py
fi