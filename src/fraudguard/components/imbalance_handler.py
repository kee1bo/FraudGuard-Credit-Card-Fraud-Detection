import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from fraudguard.logger import fraud_logger

class ImbalanceHandler:
    """Handle class imbalance in the dataset"""
    
    def __init__(self, method='smote', random_state=42):
        self.method = method
        self.random_state = random_state
        self.sampler = None
        
    def get_sampler(self):
        """Get the appropriate sampling method"""
        if self.method == 'smote':
            return SMOTE(random_state=self.random_state)
        elif self.method == 'adasyn':
            return ADASYN(random_state=self.random_state)
        elif self.method == 'borderline_smote':
            return BorderlineSMOTE(random_state=self.random_state)
        elif self.method == 'smote_tomek':
            return SMOTETomek(random_state=self.random_state)
        elif self.method == 'undersampling':
            return RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
    
    def apply_sampling(self, X, y):
        """Apply the selected sampling method"""
        try:
            if self.method == 'class_weight':
                # Return class weights instead of resampling
                classes = np.unique(y)
                class_weights = compute_class_weight(
                    'balanced', classes=classes, y=y
                )
                return X, y, dict(zip(classes, class_weights))
            
            # Apply resampling
            self.sampler = self.get_sampler()
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            fraud_logger.info(f"Applied {self.method} sampling")
            fraud_logger.info(f"Original shape: {X.shape}, New shape: {X_resampled.shape}")
            fraud_logger.info(f"Original fraud ratio: {y.mean():.4f}, New fraud ratio: {y_resampled.mean():.4f}")
            
            return X_resampled, y_resampled, None
            
        except Exception as e:
            fraud_logger.error(f"Error applying {self.method}: {e}")
            return X, y, None