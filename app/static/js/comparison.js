/**
 * Model Comparison Dashboard JavaScript
 * Handles interactive comparison of fraud detection models
 */

let comparisonChart = null;

function initializeComparison() {
    console.log('Initializing model comparison dashboard...');
    
    // Initialize comparison chart
    initializeComparisonChart();
    
    // Setup event listeners
    setupEventListeners();
    
    // Update rankings
    updateRankings();
}

function initializeComparisonChart() {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx || !window.modelMetrics) {
        console.warn('Chart canvas or model metrics not found');
        return;
    }
    
    // Prepare data for chart
    const models = Object.keys(window.modelMetrics);
    const metrics = ['precision', 'recall', 'f1_score', 'roc_auc_score'];
    const colors = [
        'rgba(54, 162, 235, 0.8)',   // Blue
        'rgba(255, 99, 132, 0.8)',   // Red
        'rgba(75, 192, 192, 0.8)',   // Green
        'rgba(255, 205, 86, 0.8)',   // Yellow
        'rgba(153, 102, 255, 0.8)',  // Purple
        'rgba(255, 159, 64, 0.8)'    // Orange
    ];
    
    const datasets = models.map((model, index) => {
        const data = metrics.map(metric => {
            if (metric === 'precision' || metric === 'recall') {
                return window.modelMetrics[model]?.classification_report?.['1']?.[metric] || 0;
            } else if (metric === 'f1_score') {
                return window.modelMetrics[model]?.classification_report?.['1']?.['f1-score'] || 0;
            } else if (metric === 'roc_auc_score') {
                return window.modelMetrics[model]?.roc_auc_score || 0;
            }
            return 0;
        });
        
        return {
            label: model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            data: data,
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length].replace('0.8', '1'),
            borderWidth: 2
        };
    });
    
    comparisonChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                },
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}

function setupEventListeners() {
    // Model selection checkboxes
    const checkboxes = document.querySelectorAll('.model-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateComparison);
    });
    
    // Metric selection dropdown
    const metricSelect = document.getElementById('comparisonMetric');
    if (metricSelect) {
        metricSelect.addEventListener('change', updateComparison);
    }
}

function updateComparison() {
    // Get selected models
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
        .map(cb => cb.value);
    
    // Get selected metric
    const selectedMetric = document.getElementById('comparisonMetric')?.value || 'roc_auc_score';
    
    // Update chart visibility
    updateChartVisibility(selectedModels);
    
    // Update rankings
    updateRankings(selectedMetric);
    
    // Update comparison cards visibility
    updateCardsVisibility(selectedModels);
}

function updateChartVisibility(selectedModels) {
    if (!comparisonChart) return;
    
    comparisonChart.data.datasets.forEach((dataset, index) => {
        const modelName = Object.keys(window.modelMetrics)[index];
        dataset.hidden = !selectedModels.includes(modelName);
    });
    
    comparisonChart.update();
}

function updateRankings(metric = 'roc_auc_score') {
    const rankingContainer = document.getElementById('rankingList');
    if (!rankingContainer || !window.modelMetrics) return;
    
    // Sort models by the selected metric
    const sortedModels = Object.entries(window.modelMetrics).sort((a, b) => {
        let valueA = 0, valueB = 0;
        
        if (metric === 'precision' || metric === 'recall') {
            valueA = a[1]?.classification_report?.['1']?.[metric] || 0;
            valueB = b[1]?.classification_report?.['1']?.[metric] || 0;
        } else if (metric === 'f1_score') {
            valueA = a[1]?.classification_report?.['1']?.['f1-score'] || 0;
            valueB = b[1]?.classification_report?.['1']?.['f1-score'] || 0;
        } else {
            valueA = a[1]?.[metric] || 0;
            valueB = b[1]?.[metric] || 0;
        }
        
        return valueB - valueA; // Descending order
    });
    
    // Update the ranking display
    rankingContainer.innerHTML = sortedModels.map((entry, index) => {
        const [modelName, modelData] = entry;
        let value = 0;
        
        if (metric === 'precision' || metric === 'recall') {
            value = modelData?.classification_report?.['1']?.[metric] || 0;
        } else if (metric === 'f1_score') {
            value = modelData?.classification_report?.['1']?.['f1-score'] || 0;
        } else {
            value = modelData?.[metric] || 0;
        }
        
        const badgeClass = index === 0 ? 'bg-success' : (index === 1 ? 'bg-warning' : 'bg-primary');
        
        return `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span>${index + 1}. ${modelName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                <span class="badge ${badgeClass}">${value.toFixed(3)}</span>
            </div>
        `;
    }).join('');
}

function updateCardsVisibility(selectedModels) {
    const cards = document.querySelectorAll('#comparisonCards .col-lg-4');
    
    cards.forEach((card, index) => {
        const modelName = Object.keys(window.modelMetrics)[index];
        if (selectedModels.length === 0 || selectedModels.includes(modelName)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

function exportComparison() {
    // Generate a simple report
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
        .map(cb => cb.value);
    
    let report = "Model Comparison Report\n";
    report += "=" * 30 + "\n\n";
    
    selectedModels.forEach(model => {
        const metrics = window.modelMetrics[model];
        if (metrics) {
            report += `${model.replace('_', ' ').toUpperCase()}\n`;
            report += `-${''.repeat(model.length)}\n`;
            report += `ROC AUC: ${(metrics.roc_auc_score || 0).toFixed(4)}\n`;
            report += `Precision: ${(metrics.classification_report?.['1']?.precision || 0).toFixed(4)}\n`;
            report += `Recall: ${(metrics.classification_report?.['1']?.recall || 0).toFixed(4)}\n`;
            report += `F1 Score: ${(metrics.classification_report?.['1']?.['f1-score'] || 0).toFixed(4)}\n\n`;
        }
    });
    
    // Download as text file
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_comparison_report.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Make functions available globally
window.initializeComparison = initializeComparison;
window.exportComparison = exportComparison;