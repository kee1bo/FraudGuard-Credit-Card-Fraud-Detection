@echo off
echo ===============================================
echo    FraudGuard AI - Windows Setup Script
echo ===============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

:: Check if version starts with 3.9
echo %PYTHON_VERSION% | findstr /B "3.9" >nul
if %errorlevel% neq 0 (
    echo WARNING: Python 3.9.x is recommended for best compatibility
    echo Current version: %PYTHON_VERSION%
    echo Continue anyway? (y/N)
    set /p continue=
    if /i not "%continue%"=="y" exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
python -m venv venv_fraudguard
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv_fraudguard\Scripts\activate.bat

echo Step 3: Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip and setuptools
    pause
    exit /b 1
)

echo Step 4: Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo This might be due to:
    echo - Network connectivity issues
    echo - Missing Microsoft Visual C++ Build Tools
    echo - Insufficient permissions
    pause
    exit /b 1
)

echo Step 5: Installing FraudGuard package...
pip install -e .
if %errorlevel% neq 0 (
    echo ERROR: Failed to install FraudGuard package
    pause
    exit /b 1
)

echo Step 6: Verifying installation...
python verify_setup.py
if %errorlevel% neq 0 (
    echo WARNING: Setup verification failed
    echo The installation may be incomplete
    pause
)

echo.
echo ===============================================
echo           Setup Complete!
echo ===============================================
echo.
echo To start using FraudGuard:
echo 1. Activate the environment: venv_fraudguard\Scripts\activate
echo 2. Run the application: python run_app.py
echo 3. Open your browser to: http://localhost:5000
echo.
echo For training models: python main.py
echo For help: python run_app.py --help
echo.

:: Ask if user wants to start the application now
echo Start the application now? (y/N)
set /p start_now=
if /i "%start_now%"=="y" (
    echo Starting FraudGuard...
    python run_app.py
)

pause