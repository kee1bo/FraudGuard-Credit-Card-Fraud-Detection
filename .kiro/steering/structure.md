# Project Structure & Organization

## Overall Architecture

FraudGuard follows a modular, layered architecture with clear separation of concerns:

```
fraudguard-ai/
├── 📱 app/                          # Flask web application layer
├── 🧮 src/fraudguard/              # Core business logic package
├── 📊 artifacts/                    # Generated models and reports
├── 📓 notebooks/                    # Research and experimentation
├── 🧪 tests/                        # Test suites
├── 📖 docs/                         # Documentation
└── 🔧 scripts/                      # Automation and utilities
```

## Core Package Structure (`src/fraudguard/`)

### Components (`src/fraudguard/components/`)
Reusable business logic components following single responsibility principle:
- **Data Processing**: `data_ingestion.py`, `data_transformation.py`, `feature_engineering.py`
- **Model Training**: `model_trainer.py`, `model_evaluation.py`, `imbalance_handler.py`
- **Quality & Monitoring**: `input_validator.py`, `confidence_scorer.py`, `mapping_quality_monitor.py`
- **System Support**: `audit_logger.py`, `error_handler.py`, `performance_optimizer.py`

### Models (`src/fraudguard/models/`)
ML model implementations with consistent interfaces:
- **Base Classes**: `base_model.py`, `base_feature_mapper.py`
- **Traditional Models**: `logistic_regression.py`, `random_forest.py`
- **Advanced Models**: `xgboost_model.py`, `catboost_model.py`, `lightgbm_model.py`
- **Ensemble**: `ensemble_model.py`, `ensemble_mapper.py`
- **Feature Mapping**: `*_mapper.py` files for intelligent feature mapping
- **Factory**: `model_factory.py` for model instantiation

### Pipelines (`src/fraudguard/pipeline/`)
Orchestration layer for complex workflows:
- **Training**: `training_pipeline.py` - End-to-end model training
- **Prediction**: `prediction_pipeline.py` - Real-time inference
- **Intelligent Prediction**: `intelligent_prediction_pipeline.py` - User-friendly predictions
- **Explanation**: `explanation_pipeline.py` - Generate model explanations
- **Mapping Training**: `mapping_training_pipeline.py` - Train feature mappers

### Explainers (`src/fraudguard/explainers/`)
Explainable AI implementations:
- **Base**: `base_explainer.py` - Common explainer interface
- **SHAP**: `shap_explainer.py` - SHAP-based explanations
- **LIME**: `lime_explainer.py` - LIME-based explanations
- **Feature Importance**: `feature_importance.py` - Model-specific importance
- **Visualization**: `explanation_visualizer.py` - Charts and plots

### MLOps (`src/fraudguard/mlops/`)
Production and lifecycle management:
- **Model Registry**: `model_registry.py`, `mapping_model_registry.py`
- **Experiment Tracking**: `experiment_tracker.py`
- **Deployment**: `deployment_manager.py`
- **Monitoring**: `performance_monitor.py`
- **Metadata**: `model_metadata.py`

### Configuration & Utilities
- **Config**: `config/` - Configuration management and model parameters
- **Entity**: `entity/` - Data classes and type definitions
- **Utils**: `utils/` - Common utilities, metrics, and visualization helpers
- **Constants**: `constants/` - Project-wide constants and settings

## Web Application (`app/`)

### Flask Application Structure
- **Main**: `main.py` - Application factory and blueprint registration
- **Routes**: `routes/` - Blueprint-based route organization
  - `main_routes.py` - Home and general pages
  - `prediction_routes.py` - Fraud prediction endpoints
  - `intelligent_prediction_routes.py` - User-friendly prediction interface
  - `dashboard_routes.py` - Analytics and monitoring
  - `api_routes.py` - REST API endpoints
  - `mlops_routes.py` - MLOps and model management
  - `documentation_routes.py` - Documentation pages

### Frontend Assets
- **Templates**: `templates/` - Jinja2 HTML templates with inheritance
- **Static Assets**: `static/` - CSS, JavaScript, and images
  - `css/` - Modular stylesheets (main, dashboard, components, design-system)
  - `js/` - Feature-specific JavaScript modules
  - `images/` - Logos, icons, and graphics

## Artifacts Directory (`artifacts/`)

### Generated Assets Organization
- **Models**: `models/` - Trained model artifacts by algorithm
- **Feature Mapping**: `feature_mapping/` - Feature mapping models and metadata
- **Explainers**: `explainers/` - Saved explainer configurations
- **Reports**: `reports/` - Generated performance reports and dashboards
- **Experiments**: `experiments/` - MLflow experiment tracking data

## Testing Structure (`tests/`)

### Test Organization
- **Unit Tests**: `unit/` - Component-level testing
- **Integration Tests**: `integration/` - End-to-end workflow testing
- **Performance Tests**: Root level - Performance and load testing
- **Specialized Tests**: Feature mapping, integration performance

## Development Scripts (`scripts/`)

### Automation and Utilities
- **Training**: `train_all_models.py` - Comprehensive model training
- **Validation**: `validate_system_readiness.py` - System health checks
- **Testing**: `run_comprehensive_tests.py` - Full test suite execution
- **Data**: `create_test_data.py` - Generate test datasets
- **Deployment**: `deploy.py` - Production deployment automation

## Key Architectural Patterns

### 1. Layered Architecture
- **Presentation Layer**: Flask web application
- **Business Logic Layer**: Core fraudguard package
- **Data Layer**: Artifacts and model storage

### 2. Pipeline Pattern
- Complex workflows broken into discrete, testable steps
- Each pipeline manages its own configuration and error handling
- Consistent interfaces across all pipeline types

### 3. Factory Pattern
- `model_factory.py` for dynamic model instantiation
- Configuration-driven model selection and parameter setting

### 4. Repository Pattern
- Model registry for artifact management
- Consistent storage and retrieval interfaces

### 5. Component-Based Design
- Single responsibility components
- Dependency injection through configuration
- Easy testing and mocking

## File Naming Conventions

### Python Files
- **Snake_case**: All Python files use snake_case naming
- **Descriptive Names**: Files clearly indicate their purpose
- **Suffixes**: `_pipeline.py`, `_model.py`, `_explainer.py` for clarity

### Configuration Files
- **YAML**: Primary configuration in `config.yaml`
- **Environment**: `.env` files for environment-specific settings

### Artifacts
- **Versioned**: Models include version informati