# Technology Stack & Build System

## Core Technologies

### Backend Framework
- **Flask 3.0.0**: Web application framework with blueprint-based routing
- **Python 3.9+**: Primary development language

### Machine Learning Stack
- **scikit-learn 1.3.2**: Core ML algorithms and utilities
- **XGBoost 2.0.3**: Gradient boosting framework
- **LightGBM 4.1.0**: Microsoft's gradient boosting framework
- **CatBoost 1.2.2**: Yandex's gradient boosting with categorical features
- **imbalanced-learn 0.11.0**: Techniques for imbalanced datasets

### Explainable AI
- **SHAP 0.44.0**: SHapley Additive exPlanations
- **LIME 0.2.0.1**: Local Interpretable Model-agnostic Explanations

### Data Processing
- **pandas 2.1.4**: Data manipulation and analysis
- **numpy 1.26.2**: Numerical computing
- **joblib 1.3.2**: Model serialization and parallel processing

### Visualization
- **matplotlib 3.8.2**: Static plotting
- **plotly 5.17.0**: Interactive visualizations
- **seaborn 0.13.0**: Statistical data visualization

### Configuration & Environment
- **PyYAML 6.0.1**: YAML configuration parsing
- **python-dotenv 1.0.0**: Environment variable management

## Build System & Package Management

### Installation
```bash
# Create virtual environment (Python 3.9 required)
python3.9 -m venv venv_fraudguard
source venv_fraudguard/bin/activate  # Linux/macOS
# OR: venv_fraudguard\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Setup Scripts
- **Linux/macOS**: `./setup_linux.sh` or `./setup_macos.sh`
- **Windows**: `setup_windows.bat`
- **Quick setup**: `./quick_setup.sh`

## Common Commands

### Training & Model Management
```bash
# Train all models with default configuration
python main.py

# Train specific models
python scripts/train_all_models.py --models xgboost catboost

# Quick training (reduced parameters)
python scripts/train_all_models.py --quick

# Train mapping models for intelligent prediction
python train_mapping_models.py
```

### Web Application
```bash
# Start development server
python run_app.py

# Start with custom configuration
python run_app.py --host 0.0.0.0 --port 8080

# Simple prediction interface
python run_simple.py
```

### Testing & Validation
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Validate system readiness
python scripts/validate_system_readiness.py

# Run comprehensive tests
python scripts/run_comprehensive_tests.py

# Performance testing
python tests/test_integration_performance.py
python tests/test_mapping_performance.py
```

### Development Tools
```bash
# Code quality checks
pylint src/
mypy src/

# Jupyter notebooks for research
jupyter lab notebooks/

# Generate documentation
python scripts/generate_explanations.py
```

### Data Management
```bash
# Create test data
python scripts/create_test_data.py

# Verify setup and data
python verify_setup.py
```

## Configuration Management

### Primary Config File
- **config.yaml**: Main configuration for models, data, and application settings
- Environment variables override config file settings
- Flask configuration in `src/fraudguard/constants/constants.py`

### Key Configuration Sections
- `data`: Dataset paths and preprocessing settings
- `models`: Model parameters and training configuration
- `explainability`: SHAP/LIME settings
- `web_app`: Flask application settings
- `logging`: Log levels and file management

## Development Environment

### Required Python Version
- **Python 3.9+** (tested with 3.9-3.12)
- Virtual environment strongly recommended

### IDE Configuration
- Project uses `src/` layout with package in `src/fraudguard/`
- Python path automatically configured in entry scripts
- Type hints used throughout codebase

### Logging
- Centralized logging via `fraudguard.logger`
- Logs saved to `logs/fraudguard_YYYYMMDD.log`
- Console and file output with configurable levels