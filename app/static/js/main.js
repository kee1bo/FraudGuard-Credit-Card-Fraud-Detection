    });
    }

    setupDemoTransaction() {
        const demoTransaction = document.querySelector('.demo-transaction');
        if (demoTransaction) {
            // Simulate real-time updates
            setInterval(() => {
                this.updateDemoTransaction();
            }, 10000); // Update every 10 seconds
        }
    }

    updateDemoTransaction() {
        // Simulate new transaction data
        const amounts = ['$1,247.32', '$3,891.55', '$892.17', '$2,847.50', '$5,234.88'];
        const riskScores = [15, 85, 32, 67, 91];
        const statuses = [
            { text: 'LEGITIMATE', class: 'bg-success' },
            { text: 'FRAUDULENT', class: 'bg-danger' },
            { text: 'SUSPICIOUS', class: 'bg-warning' }
        ];

        const randomAmount = amounts[Math.floor(Math.random() * amounts.length)];
        const randomRisk = riskScores[Math.floor(Math.random() * riskScores.length)];
        const randomStatus = statuses[randomRisk > 50 ? 1 : 0];

        // Update demo transaction display
        const amountEl = document.querySelector('.demo-transaction strong');
        const progressBar = document.querySelector('.demo-transaction .progress-bar');
        const statusBadge = document.querySelector('.demo-transaction .badge');

        if (amountEl) amountEl.textContent = randomAmount;
        if (progressBar) {
            progressBar.style.width = randomRisk + '%';
            progressBar.textContent = randomRisk + '%';
            progressBar.className = `progress-bar ${randomRisk > 50 ? 'bg-danger' : 'bg-success'}`;
        }
        if (statusBadge) {
            statusBadge.textContent = randomStatus.text;
            statusBadge.className = `badge ${randomStatus.class}`;
        }
    }
}

// Utility Functions
class Utils {
    static formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    static formatPercentage(value, decimals = 2) {
        return (value * 100).toFixed(decimals) + '%';
    }

    static formatNumber(value, decimals = 2) {
        return parseFloat(value).toFixed(decimals);
    }

    static showLoading(element) {
        element.innerHTML = `
            <div class="d-flex justify-content-center align-items-center" style="min-height: 100px;">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }

    static showError(element, message) {
        element.innerHTML = `
            <div class="alert alert-danger text-center">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
    }

    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    static async fetchWithRetry(url, options = {}, retries = 3) {
        for (let i = 0; i < retries; i++) {
            try {
                const response = await fetch(url, options);
                if (response.ok) {
                    return response;
                }
                throw new Error(`HTTP ${response.status}`);
            } catch (error) {
                if (i === retries - 1) throw error;
                await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
            }
        }
    }
}

// API Client
class APIClient {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
    }

    async get(endpoint) {
        try {
            const response = await Utils.fetchWithRetry(`${this.baseURL}${endpoint}`);
            return await response.json();
        } catch (error) {
            console.error(`API GET error for ${endpoint}:`, error);
            throw error;
        }
    }

    async post(endpoint, data) {
        try {
            const response = await Utils.fetchWithRetry(`${this.baseURL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.error(`API POST error for ${endpoint}:`, error);
            throw error;
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.fraudGuardApp = new FraudGuardApp();
    window.utils = Utils;
    window.apiClient = new APIClient();
    
    // Add custom CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes countUp {
            from { transform: scale(0.5); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        .loaded .fade-in {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
});

// Global error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    // Could send error to logging service here
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    // Could send error to logging service here
});