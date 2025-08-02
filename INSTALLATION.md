# FraudGuard AI - Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.9.x (Required for compatibility)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+

### Prerequisites
- Git (for cloning the repository)
- Internet connection (for downloading dependencies)

## Quick Start (Recommended)

Choose your operating system and run the appropriate setup script:

### Windows
```cmd
# Open Command Prompt as Administrator
git clone https://github.com/yourusername/fraudguard-ai.git
cd fraudguard-ai
setup_windows.bat
```

### macOS
```bash
# Open Terminal
git clone https://github.com/yourusername/fraudguard-ai.git
cd fraudguard-ai
chmod +x setup_macos.sh
./setup_macos.sh
```

### Linux (Ubuntu/Debian)
```bash
# Open Terminal
git clone https://github.com/yourusername/fraudguard-ai.git
cd fraudguard-ai
chmod +x setup_linux.sh
./setup_linux.sh
```

## Manual Installation

If you prefer to install manually or the automated scripts fail, follow these steps:

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/fraudguard-ai.git
cd fraudguard-ai
```

### Step 2: Python Version Check
Verify you have Python 3.9.x installed:
```bash
python --version
# or
python3 --version
```

If you don't have Python 3.9, download it from [python.org](https://www.python.org/downloads/).

### Step 3: Create Virtual Environment

**Windows:**
```cmd
python -m venv venv_fraudguard
venv_fraudguard\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv_fraudguard
source venv_fraudguard/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Step 5: Verify Installation
```bash
python verify_setup.py
```

### Step 6: Run Application
```bash
python run_app.py
```

Open your browser and navigate to: `http://localhost:5000`

## Detailed OS-Specific Instructions

### Windows Detailed Setup

1. **Install Python 3.9**
   - Download from [python.org](https://www.python.org/downloads/windows/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install Git**
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default installation options

3. **Run Setup**
   - Open Command Prompt as Administrator
   - Navigate to desired installation directory
   - Run the setup commands above

### macOS Detailed Setup

1. **Install Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.9**
   ```bash
   brew install python@3.9
   ```

3. **Install Git** (if not already installed)
   ```bash
   brew install git
   ```

4. **Run Setup**
   - Follow the macOS quick start commands above

### Linux (Ubuntu/Debian) Detailed Setup

1. **Update Package List**
   ```bash
   sudo apt update
   ```

2. **Install Python 3.9 and Dependencies**
   ```bash
   sudo apt install python3.9 python3.9-venv python3.9-dev python3-pip git
   ```

3. **Run Setup**
   - Follow the Linux quick start commands above

## Troubleshooting

### Common Issues

**Issue: "Python 3.9 not found"**
- Solution: Ensure Python 3.9.x is installed and accessible via PATH
- On some systems, use `python3.9` instead of `python`

**Issue: "Permission denied" on macOS/Linux**
- Solution: Run setup script with execute permissions: `chmod +x setup_script.sh`

**Issue: "Microsoft Visual C++ 14.0 is required" (Windows)**
- Solution: Install Microsoft C++ Build Tools or Visual Studio Community

**Issue: "Failed to build wheel" errors**
- Solution: Upgrade pip and setuptools: `pip install --upgrade pip setuptools wheel`

**Issue: Port 5000 already in use**
- Solution: Kill existing process or change port in `run_app.py`

### Getting Help

1. **Check the logs**: Look in the `logs/` directory for error messages
2. **Verify environment**: Run `python verify_setup.py` to check installation
3. **Check dependencies**: Ensure all packages in `requirements.txt` are installed
4. **System resources**: Ensure you have enough RAM and storage space

## Development Setup

For development or advanced users:

### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### Run Tests
```bash
pytest tests/
```

### Code Quality Checks
```bash
pylint src/
mypy src/
```

## Performance Optimization

### For Large Datasets
- Increase Python memory limit: `export PYTHONHASHSEED=0`
- Use SSD storage for better I/O performance
- Consider increasing system swap space

### For Production Deployment
- Use a WSGI server like Gunicorn
- Configure reverse proxy (nginx)
- Set up proper logging and monitoring
- Use environment variables for configuration

## Uninstallation

To completely remove FraudGuard:

1. **Deactivate virtual environment** (if active)
   ```bash
   deactivate
   ```

2. **Remove project directory**
   ```bash
   rm -rf fraudguard-ai  # Linux/macOS
   rmdir /s fraudguard-ai  # Windows
   ```

3. **Remove Python packages** (optional)
   - If you want to remove Python 3.9 and pip packages, follow your OS-specific uninstallation procedures

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python run_app.py` | Start the web application |
| `python main.py` | Train models from scratch |
| `python verify_setup.py` | Verify installation |
| `deactivate` | Exit virtual environment |
| `source venv_fraudguard/bin/activate` | Reactivate environment (Linux/macOS) |
| `venv_fraudguard\Scripts\activate` | Reactivate environment (Windows) |

For additional help or issues, please refer to the project documentation or create an issue in the GitHub repository.