@echo off
REM FraudGuard AI - Quick Setup Script for Windows
REM This script automates the Python 3.9 environment setup

echo 🎓 FraudGuard AI - MSc Data Science Project Setup
echo ==================================================

REM Check if Python 3.9 is available
echo [INFO] Checking Python 3.9 availability...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.9 from:
    echo https://www.python.org/downloads/release/python-3918/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [SUCCESS] Found Python %PYTHON_VERSION%

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo [ERROR] Please run this script from the FraudGuard project root directory
    pause
    exit /b 1
)

if not exist "main.py" (
    echo [ERROR] Please run this script from the FraudGuard project root directory
    pause
    exit /b 1
)

REM Create virtual environment
echo [INFO] Creating Python virtual environment...
if exist "venv_fraudguard" (
    echo [WARNING] Virtual environment already exists. Removing old one...
    rmdir /s /q venv_fraudguard
)

python -m venv venv_fraudguard
echo [SUCCESS] Virtual environment created: venv_fraudguard

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv_fraudguard\Scripts\activate.bat

REM Upgrade pip and install build tools
echo [INFO] Upgrading pip and installing build tools...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo [INFO] Installing project dependencies...
pip install -r requirements.txt

REM Install development dependencies
if exist "requirements-dev.txt" (
    echo [INFO] Installing development dependencies...
    pip install -r requirements-dev.txt
)

REM Install project in development mode
echo [INFO] Installing FraudGuard package in development mode...
pip install -e .

REM Verify installation
echo [INFO] Verifying installation...
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import sklearn, pandas, numpy, flask, shap, lime, matplotlib, xgboost, lightgbm; print('✅ All core dependencies imported successfully!')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Some dependencies failed to import
    pause
    exit /b 1
)

REM Create necessary directories
echo [INFO] Creating project directories...
mkdir data 2>nul
mkdir logs 2>nul
mkdir artifacts 2>nul
mkdir artifacts\models 2>nul
mkdir artifacts\preprocessors 2>nul
mkdir artifacts\explainers 2>nul
mkdir artifacts\reports 2>nul
echo [SUCCESS] Project directories created

REM Check for dataset
echo [INFO] Checking for dataset...
if not exist "data\creditcard.csv" (
    echo [WARNING] Dataset not found at data\creditcard.csv
    echo.
    echo Please download the ULB Credit Card Fraud dataset:
    echo 1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    echo 2. Download creditcard.csv
    echo 3. Place it in the data\ folder
    echo.
) else (
    echo [SUCCESS] Dataset found at data\creditcard.csv
)

echo.
echo [SUCCESS] 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    venv_fraudguard\Scripts\activate
echo.
echo 2. Download the dataset (if not already done):
echo    Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
echo    Place creditcard.csv in the data\ folder
echo.
echo 3. Run the training pipeline:
echo    python main.py
echo.
echo 4. Start the web application:
echo    python run_app.py
echo.
echo For more information, see README.md
echo.
pause