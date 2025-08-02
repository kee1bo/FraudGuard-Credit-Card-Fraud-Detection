# src/fraudguard/utils/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple, List

class ModelMetrics:
    """Comprehensive model evaluation metrics"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all classification metrics"""
        try:
            # Basic classification metrics
            metrics = {
                'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
                'precision': float(precision_score(self.y_true, self.y_pred, zero_division=0)),
                'recall': float(recall_score(self.y_true, self.y_pred, zero_division=0)),
                'f1_score': float(f1_score(self.y_true, self.y_pred, zero_division=0))
            }
            
            # Confusion matrix
            cm = confusion_matrix(self.y_true, self.y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Classification report
            try:
                clf_report = classification_report(self.y_true, self.y_pred, output_dict=True, zero_division=0)
                metrics['classification_report'] = clf_report
            except Exception as e:
                print(f"Warning: Could not generate classification report: {e}")
                # Create a basic classification report structure
                metrics['classification_report'] = {
                    '0': {'precision': metrics['precision'], 'recall': metrics['recall'], 'f1-score': metrics['f1_score']},
                    '1': {'precision': metrics['precision'], 'recall': metrics['recall'], 'f1-score': metrics['f1_score']},
                    'accuracy': metrics['accuracy']
                }
            
            # Probability-based metrics (if available)
            if self.y_pred_proba is not None:
                try:
                    # Check if we have both classes in predictions
                    if len(np.unique(self.y_true)) > 1 and len(np.unique(self.y_pred_proba)) > 1:
                        metrics['roc_auc_score'] = float(roc_auc_score(self.y_true, self.y_pred_proba))
                        metrics['average_precision'] = float(average_precision_score(self.y_true, self.y_pred_proba))
                    else:
                        print("Warning: ROC AUC not defined for single class predictions")
                        metrics['roc_auc_score'] = 0.5  # Random classifier performance
                        metrics['average_precision'] = float(np.mean(self.y_true))  # Baseline precision
                except Exception as e:
                    print(f"Warning: Could not calculate ROC AUC: {e}")
                    metrics['roc_auc_score'] = 0.5
                    metrics['average_precision'] = float(np.mean(self.y_true)) if len(self.y_true) > 0 else 0.0
            else:
                # No probability predictions available
                metrics['roc_auc_score'] = 0.5
                metrics['average_precision'] = float(np.mean(self.y_true)) if len(self.y_true) > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return default metrics structure
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc_score': 0.5,
                'average_precision': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'classification_report': {
                    '0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
                    '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
                    'accuracy': 0.0
                }
            }