class ModelSelector {
    constructor(selectElement, options = {}) {
        this.selectElement = selectElement;
        this.options = {
            showDescriptions: true,
            showPerformance: true,
            allowMultiple: false,
            ...options
        };
        
        this.models = {};
        this.selectedModels = new Set();
        
        this.init();
    }

    init() {
        this.loadModelData();
        this.enhanceSelector();
        this.setupEventListeners();
    }

    async loadModelData() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (data.models) {
                // Load detailed model information
                for (const model of data.models) {
                    try {
                        const metricsResponse = await fetch(`/api/model_metrics/${model}`);
                        if (metricsResponse.ok) {
                            const metrics = await metricsResponse.json();
                            this.models[model] = {
                                name: model,
                                displayName: this.formatModelName(model),
                                description: this.getModelDescription(model),
                                metrics: metrics,
                                performance: this.calculateOverallPerformance(metrics)
                            };
                        }
                    } catch (error) {
                        console.warn(`Could not load metrics for ${model}:`, error);
                        this.models[model] = {
                            name: model,
                            displayName: this.formatModelName(model),
                            description: this.getModelDescription(model),
                            metrics: null,
                            performance: 0
                        };
                    }
                }
            }
            
            this.updateSelector();
        } catch (error) {
            console.error('Error loading model data:', error);
        }
    }

    formatModelName(modelName) {
        const nameMap = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'catboost': 'CatBoost',
            'ensemble': 'Ensemble Model'
        };
        
        return nameMap[modelName] || modelName.replace('_', ' ').title();
    }

    getModelDescription(modelName) {
        const descriptions = {
            'logistic_regression': 'Linear classifier with regularization - fast and interpretable',
            'random_forest': 'Ensemble of decision trees - robust and handles overfitting well',
            'xgboost': 'Gradient boosting framework - high performance on structured data',
            'lightgbm': 'Gradient boosting with optimized speed and memory usage',
            'catboost': 'Gradient boosting with automatic categorical feature handling',
            'ensemble': 'Combination of multiple models for improved accuracy'
        };
        
        return descriptions[modelName] || 'Advanced machine learning model for fraud detection';
    }

    calculateOverallPerformance(metrics) {
        if (!metrics || !metrics.classification_report) return 0;
        
        const fraudMetrics = metrics.classification_report['1'];
        if (!fraudMetrics) return 0;
        
        // Weighted score: ROC AUC (40%) + F1 Score (30%) + Precision (20%) + Recall (10%)
        const rocAuc = metrics.roc_auc_score || 0;
        const f1Score = fraudMetrics['f1-score'] || 0;
        const precision = fraudMetrics.precision || 0;
        const recall = fraudMetrics.recall || 0;
        
        return (rocAuc * 0.4 + f1Score * 0.3 + precision * 0.2 + recall * 0.1) * 100;
    }

    enhanceSelector() {
        // Create enhanced selector container
        const container = document.createElement('div');
        container.className = 'model-selector-enhanced';
        
        // Insert container after original select
        this.selectElement.parentNode.insertBefore(container, this.selectElement.nextSibling);
        
        // Hide original select
        this.selectElement.style.display = 'none';
        
        // Create enhanced UI
        this.createEnhancedUI(container);
    }

    createEnhancedUI(container) {
        container.innerHTML = `
            <div class="model-selector-dropdown">
                <button class="btn btn-outline-primary dropdown-toggle w-100 text-start" 
                        type="button" data-bs-toggle="dropdown">
                    <span class="selected-text">Choose a model...</span>
                </button>
                <ul class="dropdown-menu w-100" id="modelDropdown">
                    <!-- Will be populated by updateSelector -->
                </ul>
            </div>
            
            ${this.options.showPerformance ? `
                <div class="model-performance-preview mt-2" style="display: none;">
                    <div class="card border-info">
                        <div class="card-body p-2">
                            <h6 class="card-title mb-1">Model Performance</h6>
                            <div class="performance-metrics"></div>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;
        
        this.dropdownMenu = container.querySelector('#modelDropdown');
        this.selectedText = container.querySelector('.selected-text');
        this.performancePreview = container.querySelector('.model-performance-preview');
    }

    updateSelector() {
        if (!this.dropdownMenu) return;
        
        this.dropdownMenu.innerHTML = '';
        
        // Sort models by performance
        const sortedModels = Object.values(this.models)
            .sort((a, b) => b.performance - a.performance);
        
        sortedModels.forEach(model => {
            const li = document.createElement('li');
            li.innerHTML = `
                <a class="dropdown-item model-option" href="#" data-model="${model.name}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="model-name">${model.displayName}</div>
                            ${this.options.showDescriptions ? `
                                <small class="text-muted">${model.description}</small>
                            ` : ''}
                        </div>
                        ${this.options.showPerformance && model.performance > 0 ? `
                            <div class="performance-badge">
                                <span class="badge bg-${this.getPerformanceBadgeColor(model.performance)}">
                                    ${model.performance.toFixed(1)}%
                                </span>
                            </div>
                        ` : ''}
                    </div>
                </a>
            `;
            
            this.dropdownMenu.appendChild(li);
        });
    }

    getPerformanceBadgeColor(performance) {
        if (performance >= 90) return 'success';
        if (performance >= 80) return 'primary';
        if (performance >= 70) return 'warning';
        return 'secondary';
    }

    setupEventListeners() {
        // Model selection
        this.dropdownMenu?.addEventListener('click', (e) => {
            e.preventDefault();
            const modelOption = e.target.closest('.model-option');
            if (modelOption) {
                const modelName = modelOption.dataset.model;
                this.selectModel(modelName);
            }
        });
        
        // Show performance preview on hover
        if (this.options.showPerformance) {
            this.dropdownMenu?.addEventListener('mouseenter', (e) => {
                const modelOption = e.target.closest('.model-option');
                if (modelOption) {
                    const modelName = modelOption.dataset.model;
                    this.showPerformancePreview(modelName);
                }
            }, true);
            
            this.dropdownMenu?.addEventListener('mouseleave', () => {
                this.hidePerformancePreview();
            });
        }
    }

    selectModel(modelName) {
        const model = this.models[modelName];
        if (!model) return;
        
        // Update original select
        this.selectElement.value = modelName;
        
        // Update display
        this.selectedText.textContent = model.displayName;
        
        // Add selected class
        this.dropdownMenu.querySelectorAll('.model-option').forEach(option => {
            option.classList.toggle('active', option.dataset.model === modelName);
        });
        
        // Trigger change event
        this.selectElement.dispatchEvent(new Event('change'));
        
        // Show performance preview for selected model
        if (this.options.showPerformance) {
            this.showPerformancePreview(modelName, true);
        }
    }

    showPerformancePreview(modelName, persist = false) {
        const model = this.models[modelName];
        if (!model || !model.metrics || !this.performancePreview) return;
        
        const metricsContainer = this.performancePreview.querySelector('.performance-metrics');
        const fraudMetrics = model.metrics.classification_report['1'];
        
        metricsContainer.innerHTML = `
            <div class="row g-2 small">
                <div class="col-6">
                    <strong>ROC AUC:</strong> ${model.metrics.roc_auc_score.toFixed(3)}
                </div>
                <div class="col-6">
                    <strong>F1 Score:</strong> ${fraudMetrics['f1-score'].toFixed(3)}
                </div>
                <div class="col-6">
                    <strong>Precision:</strong> ${fraudMetrics.precision.toFixed(3)}
                </div>
                <div class="col-6">
                    <strong>Recall:</strong> ${fraudMetrics.recall.toFixed(3)}
                </div>
            </div>
        `;
        
        this.performancePreview.style.display = 'block';
        
        if (!persist) {
            this.previewTimeout = setTimeout(() => {
                this.hidePerformancePreview();
            }, 3000);
        }
    }

    hidePerformancePreview() {
        if (this.performancePreview) {
            this.performancePreview.style.display = 'none';
        }
        
        if (this.previewTimeout) {
            clearTimeout(this.previewTimeout);
        }
    }

    getSelectedModel() {
        return this.selectElement.value;
    }

    getSelectedModelData() {
        const selectedModel = this.getSelectedModel();
        return selectedModel ? this.models[selectedModel] : null;
    }
}

// Auto-initialize model selectors
document.addEventListener('DOMContentLoaded', function() {
    const modelSelectors = document.querySelectorAll('select[name="model_type"]');
    modelSelectors.forEach(select => {
        if (!select.dataset.enhanced) {
            new ModelSelector(select, {
                showDescriptions: true,
                showPerformance: true
            });
            select.dataset.enhanced = 'true';
        }
    });
});