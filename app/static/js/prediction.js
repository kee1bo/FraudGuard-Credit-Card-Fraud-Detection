     this.predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        this.predictBtn.disabled = true;
        
        // Add loading animation to form
        this.form.classList.add('loading');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }
}

// Prediction API client
class PredictionAPI {
    constructor() {
        this.baseURL = '/api';
    }

    async predict(transactionData, modelType, includeExplanation = true) {
        try {
            const response = await fetch(`${this.baseURL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    transaction_data: transactionData,
                    model_type: modelType,
                    include_explanation: includeExplanation
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Prediction API error:', error);
            throw error;
        }
    }

    async getAvailableModels() {
        try {
            const response = await fetch(`${this.baseURL}/models`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Models API error:', error);
            throw error;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('predictionForm')) {
        window.predictionManager = new PredictionManager();
        window.predictionAPI = new PredictionAPI();
    }
});