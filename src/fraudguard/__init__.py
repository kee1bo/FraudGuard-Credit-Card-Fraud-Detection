"""
FraudGuard: Credit Card Fraud Detection with Explainable AI

A comprehensive machine learning package for detecting credit card fraud
with explainable AI capabilities and interactive web dashboard.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .logger import fraud_logger
from .exception import FraudGuardException

__all__ = [
    'fraud_logger',
    'FraudGuardException',
]
