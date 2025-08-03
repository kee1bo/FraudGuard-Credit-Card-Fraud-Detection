/**
 * Professional Chart Visualization System
 * Using Chart.js with professional styling and animations
 */

class ProfessionalCharts {
    constructor() {
        this.defaultColors = {
            primary: '#486581',
            success: '#2f855a',
            warning: '#dd6b20',
            danger: '#c53030',
            neutral: '#4a5568',
            background: '#f7fafc'
        };
        
        this.chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            family: 'Inter, sans-serif',
                            size: 12,
                            weight: '500'
                        },
                        color: '#4a5568'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 32, 44, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#486581',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: false,
                    titleFont: {
                        family: 'Inter, sans-serif',
                        size: 13,
                        weight: '600'
                    },
                    bodyFont: {
                        family: 'Inter, sans-serif',
                        size: 12
                    }
                }
            },
            animation: {
                duration: 750,
                easing: 'easeInOutCubic'
            }
        };
    }

    /**
     * Create professional performance comparison chart
     */
    createPerformanceChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !data) return null;

        const models = Object.keys(data);
        const rocAucScores = models.map(model => {
            const metrics = data[model];
            return metrics && metrics.roc_auc_score ? metrics.roc_auc_score : 0;
        });
        const precisionScores = models.map(model => {
            const metrics = data[model];
            return metrics && metrics.classification_report && metrics.classification_report['1'] 
                ? metrics.classification_report['1'].precision || 0 : 0;
        });
        const recallScores = models.map(model => {
            const metrics = data[model];
            return metrics && metrics.classification_report && metrics.classification_report['1'] 
                ? metrics.classification_report['1'].recall || 0 : 0;
        });

        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(m => this.formatModelName(m)),
                datasets: [
                    {
                        label: 'ROC AUC',
                        data: rocAucScores,
                        backgroundColor: this.defaultColors.primary + '80',
                        borderColor: this.defaultColors.primary,
                        borderWidth: 2,
                        borderRadius: 4,
                        borderSkipped: false
                    },
                    {
                        label: 'Precision',
                        data: precisionScores,
                        backgroundColor: this.defaultColors.success + '80',
                        borderColor: this.defaultColors.success,
                        borderWidth: 2,
                        borderRadius: 4,
                        borderSkipped: false
                    },
                    {
                        label: 'Recall',
                        data: recallScores,
                        backgroundColor: this.defaultColors.warning + '80',
                        borderColor: this.defaultColors.warning,
                        borderWidth: 2,
                        borderRadius: 4,
                        borderSkipped: false
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11,
                                weight: '500'
                            },
                            color: '#718096'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: {
                            color: '#e2e8f0',
                            lineWidth: 1
                        },
                        ticks: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            color: '#718096',
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: false
                    }
                }
            }
        });
    }

    /**
     * Create professional model distribution chart
     */
    createDistributionChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !data) return null;

        const models = Object.keys(data);
        const f1Scores = models.map(model => {
            const metrics = data[model];
            return metrics && metrics.classification_report && metrics.classification_report['1'] 
                ? metrics.classification_report['1']['f1-score'] || 0 : 0;
        });

        const colors = [
            this.defaultColors.primary,
            this.defaultColors.success,
            this.defaultColors.warning,
            this.defaultColors.danger,
            this.defaultColors.neutral,
            '#805ad5'
        ];

        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: models.map(m => this.formatModelName(m)),
                datasets: [{
                    data: f1Scores,
                    backgroundColor: colors.map(color => color + '80'),
                    borderColor: colors,
                    borderWidth: 3,
                    hoverBorderWidth: 4
                }]
            },
            options: {
                ...this.chartDefaults,
                cutout: '60%',
                plugins: {
                    ...this.chartDefaults.plugins,
                    legend: {
                        ...this.chartDefaults.plugins.legend,
                        position: 'right'
                    },
                    tooltip: {
                        ...this.chartDefaults.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = (context.parsed * 100).toFixed(1);
                                return `${label}: ${value}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create professional feature importance chart
     */
    createFeatureImportanceChart(canvasId, features = null, importance = null) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        // Use sample data if none provided
        const sampleFeatures = features || ['Amount', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 'V1', 'Time', 'V4'];
        const sampleImportance = importance || [0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03];

        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sampleFeatures,
                datasets: [{
                    label: 'Feature Importance',
                    data: sampleImportance,
                    backgroundColor: this.defaultColors.primary + '80',
                    borderColor: this.defaultColors.primary,
                    borderWidth: 2,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 0.2,
                        grid: {
                            color: '#e2e8f0',
                            lineWidth: 1
                        },
                        ticks: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            color: '#718096',
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11,
                                weight: '500'
                            },
                            color: '#718096'
                        }
                    }
                },
                plugins: {
                    ...this.chartDefaults.plugins,
                    legend: {
                        display: false
                    },
                    title: {
                        display: false
                    }
                }
            }
        });
    }

    /**
     * Create professional time series chart for model performance over time
     */
    createTimeSeriesChart(canvasId, timeData) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !timeData) return null;

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeData.labels,
                datasets: [{
                    label: 'Model Performance',
                    data: timeData.values,
                    borderColor: this.defaultColors.primary,
                    backgroundColor: this.defaultColors.primary + '20',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.defaultColors.primary,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    x: {
                        grid: {
                            color: '#e2e8f0',
                            lineWidth: 1
                        },
                        ticks: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            color: '#718096'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: {
                            color: '#e2e8f0',
                            lineWidth: 1
                        },
                        ticks: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            color: '#718096',
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    ...this.chartDefaults.plugins,
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    /**
     * Format model names for display
     */
    formatModelName(modelName) {
        return modelName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Update chart with new data and smooth animation
     */
    updateChart(chart, newData) {
        if (!chart || !newData) return;

        chart.data = newData;
        chart.update('active');
    }

    /**
     * Destroy chart instance
     */
    destroyChart(chart) {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    }
}

// Global instance
window.ProfessionalCharts = new ProfessionalCharts();