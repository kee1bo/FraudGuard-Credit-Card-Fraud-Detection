class ChartManager {
    constructor() {
        this.defaultColors = {
            primary: '#007bff',
            success: '#28a745',
            danger: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8',
            secondary: '#6c757d'
        };
        
        this.gradients = {};
        this.initializeGradients();
    }

    initializeGradients() {
        // Create gradient patterns for charts
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Primary gradient
        this.gradients.primary = ctx.createLinearGradient(0, 0, 0, 400);
        this.gradients.primary.addColorStop(0, 'rgba(0, 123, 255, 0.8)');
        this.gradients.primary.addColorStop(1, 'rgba(0, 123, 255, 0.1)');
        
        // Success gradient
        this.gradients.success = ctx.createLinearGradient(0, 0, 0, 400);
        this.gradients.success.addColorStop(0, 'rgba(40, 167, 69, 0.8)');
        this.gradients.success.addColorStop(1, 'rgba(40, 167, 69, 0.1)');
        
        // Danger gradient
        this.gradients.danger = ctx.createLinearGradient(0, 0, 0, 400);
        this.gradients.danger.addColorStop(0, 'rgba(220, 53, 69, 0.8)');
        this.gradients.danger.addColorStop(1, 'rgba(220, 53, 69, 0.1)');
    }

    createROCCurve(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        return new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'ROC Curve',
                    data: data.points || this.generateROCData(),
                    borderColor: this.defaultColors.primary,
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Random Classifier',
                    data: [{x: 0, y: 0}, {x: 1, y: 1}],
                    borderColor: this.defaultColors.secondary,
                    borderDash: [5, 5],
                    borderWidth: 2,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: `ROC Curve (AUC = ${data.auc || '0.985'})`
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 6
                    }
                }
            }
        });
    }

    createPRCurve(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        return new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Precision-Recall Curve',
                    data: data.points || this.generatePRData(),
                    borderColor: this.defaultColors.success,
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Recall'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Precision'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: `Precision-Recall Curve (AP = ${data.ap || '0.887'})`
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 6
                    }
                }
            }
        });
    }

    createFeatureImportance(canvasId, features, importance) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // Sort by importance and take top 15
        const sortedData = features.map((name, index) => ({
            name: name,
            importance: importance[index]
        })).sort((a, b) => b.importance - a.importance).slice(0, 15);

        return new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: sortedData.map(d => d.name),
                datasets: [{
                    label: 'Feature Importance',
                    data: sortedData.map(d => d.importance),
                    backgroundColor: sortedData.map((_, index) => 
                        `hsla(${240 + index * 8}, 70%, 60%, 0.8)`),
                    borderColor: sortedData.map((_, index) => 
                        `hsla(${240 + index * 8}, 70%, 50%, 1)`),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Top 15 Feature Importance'
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    }
                }
            }
        });
    }

    createModelComparison(canvasId, models, metrics) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        return new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: models.map(m => m.replace('_', ' ').toUpperCase()),
                datasets: [
                    {
                        label: 'ROC AUC',
                        data: metrics.roc_auc,
                        backgroundColor: 'rgba(0, 123, 255, 0.8)',
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Precision',
                        data: metrics.precision,
                        backgroundColor: 'rgba(40, 167, 69, 0.8)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Recall',
                        data: metrics.recall,
                        backgroundColor: 'rgba(255, 193, 7, 0.8)',
                        borderColor: 'rgba(255, 193, 7, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'F1 Score',
                        data: metrics.f1,
                        backgroundColor: 'rgba(23, 162, 184, 0.8)',
                        borderColor: 'rgba(23, 162, 184, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    title: {
                        display: true,
                        text: 'Model Performance Comparison'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    }
                }
            }
        });
    }

    generateROCData() {
        // Generate sample ROC curve data
        const points = [];
        for (let i = 0; i <= 100; i++) {
            const fpr = i / 100;
            const tpr = Math.min(1, fpr + 0.3 + Math.random() * 0.2);
            points.push({x: fpr, y: tpr});
        }
        return points;
    }

    generatePRData() {
        // Generate sample Precision-Recall curve data
        const points = [];
        for (let i = 0; i <= 100; i++) {
            const recall = i / 100;
            const precision = Math.max(0, 0.9 - recall * 0.3 + Math.random() * 0.1);
            points.push({x: recall, y: precision});
        }
        return points;
    }

    updateChartData(chart, newData) {
        if (!chart) return;
        
        chart.data = newData;
        chart.update('active');
    }

    addChartAnimation(chart) {
        if (!chart) return;
        
        chart.options.animation = {
            duration: 1000,
            easing: 'easeInOutQuart'
        };
        chart.update();
    }

    exportChart(chart, filename = 'chart.png') {
        if (!chart) return;
        
        const url = chart.toBase64Image('image/png', 1);
        const link = document.createElement('a');
        link.download = filename;
        link.href = url;
        link.click();
    }

    resizeChart(chart) {
        if (!chart) return;
        
        chart.resize();
    }
}

// Chart theme configurations
const ChartThemes = {
    light: {
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        gridColor: 'rgba(0, 0, 0, 0.1)',
        textColor: '#333'
    },
    dark: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        gridColor: 'rgba(255, 255, 255, 0.1)',
        textColor: '#fff'
    }
};

// Chart utility functions
const ChartUtils = {
    formatTooltip: (tooltipItem) => {
        const label = tooltipItem.dataset.label || '';
        const value = typeof tooltipItem.parsed.y === 'number' ? 
            tooltipItem.parsed.y.toFixed(4) : tooltipItem.parsed.y;
        return `${label}: ${value}`;
    },

    generateColors: (count) => {
        const colors = [];
        const hueStep = 360 / count;
        
        for (let i = 0; i < count; i++) {
            const hue = i * hueStep;
            colors.push(`hsla(${hue}, 70%, 60%, 0.8)`);
        }
        
        return colors;
    },

    createGradient: (ctx, colorStart, colorEnd, height = 400) => {
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, colorStart);
        gradient.addColorStop(1, colorEnd);
        return gradient;
    }
};

// Initialize chart manager
document.addEventListener('DOMContentLoaded', function() {
    window.chartManager = new ChartManager();
});