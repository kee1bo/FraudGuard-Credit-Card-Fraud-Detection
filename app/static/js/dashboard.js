class DashboardManager {
    constructor() {
        this.currentModel = null;
        this.charts = {};
        this.modelMetrics = {};
        
        this.modelSelector = document.getElementById('modelSelector');
        this.comparisonMode = document.getElementById('comparisonMode');
        this.refreshBtn = document.getElementById('refreshDashboard');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
    }

    setupEventListeners() {
        this.modelSelector?.addEventListener('change', (e) => {
            this.selectModel(e.target.value);
        });

        this.comparisonMode?.addEventListener('change', (e) => {
            this.setViewMode(e.target.value);
        });

        this.refreshBtn?.addEventListener('click', () => {
            this.refreshDashboard();
        });

        // Export buttons
        document.getElementById('exportPDF')?.addEventListener('click', () => {
            this.exportToPDF();
        });

        document.getElementById('exportExcel')?.addEventListener('click', () => {
            this.exportToExcel();
        });
    }

    async selectModel(modelType) {
        if (!modelType) return;

        this.showLoading(true);
        this.currentModel = modelType;

        try {
            const response = await fetch(`/api/model_metrics/${modelType}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const metrics = await response.json();
            this.modelMetrics[modelType] = metrics;
            
            this.updateMetricsCards(metrics);
            this.updateConfusionMatrix(metrics);
            this.updateModelComparison();
            this.updatePerformanceTable();
            
        } catch (error) {
            console.error('Error fetching model metrics:', error);
            this.showError('Failed to load model metrics');
        } finally {
            this.showLoading(false);
        }
    }

    updateMetricsCards(metrics) {
        const classificationReport = metrics.classification_report;
        const fraudClass = classificationReport['1']; // Fraud class metrics

        this.updateMetricCard('rocAucValue', metrics.roc_auc_score);
        this.updateMetricCard('precisionValue', fraudClass.precision);
        this.updateMetricCard('recallValue', fraudClass.recall);
        this.updateMetricCard('f1Value', fraudClass['f1-score']);
    }

    updateMetricCard(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
            element.textContent = formattedValue;
            
            // Add animation
            element.style.transform = 'scale(0.8)';
            element.style.transition = 'transform 0.3s ease';
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 100);
        }
    }

    updateConfusionMatrix(metrics) {
        const confusionMatrix = metrics.confusion_matrix;
        const ctx = document.getElementById('confusionMatrixChart');
        
        if (!ctx) return;

        if (this.charts.confusionMatrix) {
            this.charts.confusionMatrix.destroy();
        }

        this.charts.confusionMatrix = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                datasets: [{
                    label: 'Count',
                    data: [
                        confusionMatrix[0][0], // TN
                        confusionMatrix[0][1], // FP
                        confusionMatrix[1][0], // FN
                        confusionMatrix[1][1]  // TP
                    ],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(255, 159, 64, 0.8)',
                        'rgba(54, 162, 235, 0.8)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(54, 162, 235, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Confusion Matrix Breakdown'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                }
            }
        });
    }

    updateModelComparison() {
        const ctx = document.getElementById('modelComparisonChart');
        if (!ctx) return;

        if (this.charts.modelComparison) {
            this.charts.modelComparison.destroy();
        }

        const models = Object.keys(this.modelMetrics);
        if (models.length === 0) return;

        const rocAucScores = models.map(model => this.modelMetrics[model].roc_auc_score);
        const precisionScores = models.map(model => 
            this.modelMetrics[model].classification_report['1'].precision);
        const recallScores = models.map(model => 
            this.modelMetrics[model].classification_report['1'].recall);

        this.charts.modelComparison = new Chart(ctx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: models.map(m => m.replace('_', ' ').toUpperCase()),
                datasets: [
                    {
                        label: 'ROC AUC',
                        data: rocAucScores,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        pointBackgroundColor: 'rgb(255, 99, 132)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgb(255, 99, 132)'
                    },
                    {
                        label: 'Precision',
                        data: precisionScores,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        pointBackgroundColor: 'rgb(54, 162, 235)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgb(54, 162, 235)'
                    },
                    {
                        label: 'Recall',
                        data: recallScores,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        pointBackgroundColor: 'rgb(75, 192, 192)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgb(75, 192, 192)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 1,
                        pointLabels: {
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Model Performance Comparison'
                    }
                }
            }
        });
    }

    updatePerformanceTable() {
        const tableBody = document.querySelector('#performanceTable tbody');
        if (!tableBody) return;

        tableBody.innerHTML = '';

        Object.keys(this.modelMetrics).forEach(modelName => {
            const metrics = this.modelMetrics[modelName];
            const fraudMetrics = metrics.classification_report['1'];
            
            const row = tableBody.insertRow();
            row.innerHTML = `
                <td><strong>${modelName.replace('_', ' ').title()}</strong></td>
                <td>${metrics.roc_auc_score.toFixed(4)}</td>
                <td>${fraudMetrics.precision.toFixed(4)}</td>
                <td>${fraudMetrics.recall.toFixed(4)}</td>
                <td>${fraudMetrics['f1-score'].toFixed(4)}</td>
                <td>${metrics.classification_report.accuracy.toFixed(4)}</td>
                <td><span class="badge bg-success">Active</span></td>
            `;
        });
    }

    setViewMode(mode) {
        if (mode === 'comparison') {
            this.loadAllModelMetrics();
        }
    }

    async loadAllModelMetrics() {
        this.showLoading(true);

        try {
            const modelsResponse = await fetch('/api/models');
            const modelsData = await modelsResponse.json();
            const availableModels = modelsData.models || [];

            for (const model of availableModels) {
                if (!this.modelMetrics[model]) {
                    try {
                        const response = await fetch(`/api/model_metrics/${model}`);
                        if (response.ok) {
                            const metrics = await response.json();
                            this.modelMetrics[model] = metrics;
                        }
                    } catch (error) {
                        console.warn(`Could not load metrics for ${model}:`, error);
                    }
                }
            }
            
            this.updateModelComparison();
            this.updatePerformanceTable();
            
        } catch (error) {
            console.error('Error loading all model metrics:', error);
            this.showError('Failed to load comparison data');
        } finally {
            this.showLoading(false);
        }
    }

    initializeCharts() {
        // Set Chart.js global defaults
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
    }

    showLoading(show) {
        if (this.loadingIndicator) {
            this.loadingIndicator.classList.toggle('d-none', !show);
        }
    }

    showError(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container-fluid');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
        }
    }

    refreshDashboard() {
        if (this.currentModel) {
            this.selectModel(this.currentModel);
        }
        
        // Add refresh animation
        this.refreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Refreshing...';
        setTimeout(() => {
            this.refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
        }, 2000);
    }

    exportToPDF() {
        // Simple implementation - would need PDF library in production
        window.print();
    }

    exportToExcel() {
        // Simple CSV export
        const data = Object.keys(this.modelMetrics).map(model => {
            const metrics = this.modelMetrics[model];
            const fraudMetrics = metrics.classification_report['1'];
            return {
                Model: model,
                'ROC AUC': metrics.roc_auc_score,
                Precision: fraudMetrics.precision,
                Recall: fraudMetrics.recall,
                'F1 Score': fraudMetrics['f1-score'],
                Accuracy: metrics.classification_report.accuracy
            };
        });

        this.downloadCSV(data, 'model_performance.csv');
    }

    downloadCSV(data, filename) {
        const csv = this.arrayToCSV(data);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    arrayToCSV(data) {
        if (!data.length) return '';
        
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => row[header]).join(','))
        ].join('\n');
        
        return csvContent;
    }
}

// Initialize dashboard
function initializeDashboard() {
    window.dashboardManager = new DashboardManager();
    
    // Load first model by default if available
    if (window.availableModels && window.availableModels.length > 0) {
        const firstModel = window.availableModels[0];
        document.getElementById('modelSelector').value = firstModel;
        window.dashboardManager.selectModel(firstModel);
    }
}

// Populate performance table with initial data
function populatePerformanceTable() {
    if (window.initialMetrics) {
        window.dashboardManager.modelMetrics = window.initialMetrics;
        window.dashboardManager.updatePerformanceTable();
    }
}