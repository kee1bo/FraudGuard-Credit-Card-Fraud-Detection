"""
Input Validator and Suggestion Engine
Validates user inputs and provides intelligent suggestions for the feature mapping system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, time
import re

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, ValidationResult, MerchantCategory, 
    LocationRisk, SpendingPattern, TimeContext
)
from fraudguard.logger import fraud_logger


class InputValidator:
    """Validates user inputs and provides intelligent suggestions"""
    
    def __init__(self):
        # Typical amount ranges for different merchant categories
        self.merchant_amount_ranges = {
            MerchantCategory.GROCERY: (5.0, 300.0),
            MerchantCategory.GAS_STATION: (10.0, 150.0),
            MerchantCategory.ONLINE_RETAIL: (15.0, 1000.0),
            MerchantCategory.RESTAURANT: (8.0, 200.0),
            MerchantCategory.ATM_WITHDRAWAL: (20.0, 800.0),
            MerchantCategory.DEPARTMENT_STORE: (20.0, 500.0),
            MerchantCategory.PHARMACY: (5.0, 100.0),
            MerchantCategory.ENTERTAINMENT: (10.0, 300.0),
            MerchantCategory.TRAVEL: (50.0, 5000.0),
            MerchantCategory.OTHER: (1.0, 10000.0)
        }
        
        # Typical operating hours for different merchant categories
        self.merchant_operating_hours = {
            MerchantCategory.GROCERY: (6, 23),
            MerchantCategory.GAS_STATION: (0, 23),  # Usually 24/7
            MerchantCategory.ONLINE_RETAIL: (0, 23),  # 24/7
            MerchantCategory.RESTAURANT: (6, 23),
            MerchantCategory.ATM_WITHDRAWAL: (0, 23),  # 24/7
            MerchantCategory.DEPARTMENT_STORE: (8, 22),
            MerchantCategory.PHARMACY: (7, 22),
            MerchantCategory.ENTERTAINMENT: (10, 23),
            MerchantCategory.TRAVEL: (5, 23),
            MerchantCategory.OTHER: (0, 23)
        }
        
        # Common fraud patterns to check against
        self.fraud_patterns = [
            {
                'name': 'late_night_high_amount',
                'conditions': lambda data: data.time_context.hour_of_day >= 23 or data.time_context.hour_of_day <= 5,
                'amount_threshold': 500.0,
                'warning': 'Late night transactions with high amounts are often flagged as suspicious'
            },
            {
                'name': 'foreign_high_amount',
                'conditions': lambda data: data.location_risk == LocationRisk.FOREIGN_COUNTRY,
                'amount_threshold': 1000.0,
                'warning': 'High-value transactions in foreign countries require extra verification'
            },
            {
                'name': 'atm_unusual_amount',
                'conditions': lambda data: data.merchant_category == MerchantCategory.ATM_WITHDRAWAL,
                'amount_threshold': 1000.0,
                'warning': 'ATM withdrawals over $1000 are uncommon and may be flagged'
            }
        ]
    
    def validate_transaction_amount(self, amount: float, merchant_category: Optional[MerchantCategory] = None) -> Tuple[bool, List[str], List[str]]:
        """
        Validate transaction amount
        
        Args:
            amount: Transaction amount to validate
            merchant_category: Optional merchant category for context
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        is_valid = True
        
        # Basic validation
        if amount <= 0:
            errors.append("Transaction amount must be positive")
            is_valid = False
        elif amount > 50000:
            errors.append("Transaction amount exceeds maximum limit ($50,000)")
            is_valid = False
        
        # Merchant-specific validation
        if merchant_category and merchant_category in self.merchant_amount_ranges:
            min_amount, max_amount = self.merchant_amount_ranges[merchant_category]
            
            if amount < min_amount * 0.1:
                warnings.append(f"Amount is unusually low for {merchant_category.value} transactions")
            elif amount > max_amount * 3:
                warnings.append(f"Amount is unusually high for {merchant_category.value} transactions")
            elif amount > max_amount:
                warnings.append(f"Amount is higher than typical {merchant_category.value} transactions")
        
        # Fraud pattern checks
        if amount > 10000:
            warnings.append("High-value transactions may require additional verification")
        
        return is_valid, errors, warnings
    
    def validate_time_context(self, time_context: TimeContext, merchant_category: Optional[MerchantCategory] = None) -> Tuple[bool, List[str], List[str]]:
        """
        Validate time context
        
        Args:
            time_context: TimeContext object to validate
            merchant_category: Optional merchant category for context
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        is_valid = True
        
        # Basic validation
        if not (0 <= time_context.hour_of_day <= 23):
            errors.append("Hour of day must be between 0 and 23")
            is_valid = False
        
        if not (0 <= time_context.day_of_week <= 6):
            errors.append("Day of week must be between 0 (Monday) and 6 (Sunday)")
            is_valid = False
        
        # Weekend consistency check
        expected_weekend = time_context.day_of_week >= 5
        if time_context.is_weekend != expected_weekend:
            warnings.append("Weekend flag doesn't match day of week")
        
        # Merchant-specific time validation
        if merchant_category and merchant_category in self.merchant_operating_hours:
            open_hour, close_hour = self.merchant_operating_hours[merchant_category]
            
            if not (open_hour <= time_context.hour_of_day <= close_hour):
                if merchant_category not in [MerchantCategory.GAS_STATION, MerchantCategory.ATM_WITHDRAWAL, MerchantCategory.ONLINE_RETAIL]:
                    warnings.append(f"Transaction time is outside typical {merchant_category.value} operating hours")
        
        # General time-based warnings
        if time_context.hour_of_day >= 23 or time_context.hour_of_day <= 5:
            warnings.append("Late night/early morning transactions may be flagged for review")
        
        return is_valid, errors, warnings
    
    def validate_merchant_category(self, merchant_category: MerchantCategory) -> Tuple[bool, List[str], List[str]]:
        """
        Validate merchant category
        
        Args:
            merchant_category: MerchantCategory to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        is_valid = True
        
        # Basic validation (enum ensures valid values)
        if not isinstance(merchant_category, MerchantCategory):
            errors.append("Invalid merchant category")
            is_valid = False
        
        # Category-specific warnings
        if merchant_category == MerchantCategory.OTHER:
            warnings.append("'Other' category may result in less accurate fraud detection")
        elif merchant_category == MerchantCategory.ONLINE_RETAIL:
            warnings.append("Online transactions have higher fraud risk - extra verification recommended")
        
        return is_valid, errors, warnings
    
    def validate_location_risk(self, location_risk: LocationRisk, merchant_category: Optional[MerchantCategory] = None) -> Tuple[bool, List[str], List[str]]:
        """
        Validate location risk
        
        Args:
            location_risk: LocationRisk to validate
            merchant_category: Optional merchant category for context
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        is_valid = True
        
        # Basic validation (enum ensures valid values)
        if not isinstance(location_risk, LocationRisk):
            errors.append("Invalid location risk level")
            is_valid = False
        
        # Risk-specific warnings
        if location_risk == LocationRisk.FOREIGN_COUNTRY:
            warnings.append("Foreign transactions have higher fraud risk")
        elif location_risk == LocationRisk.HIGHLY_UNUSUAL:
            warnings.append("Highly unusual locations may trigger fraud alerts")
        
        # Merchant-location consistency
        if merchant_category == MerchantCategory.ATM_WITHDRAWAL and location_risk == LocationRisk.FOREIGN_COUNTRY:
            warnings.append("Foreign ATM withdrawals are common fraud targets")
        
        return is_valid, errors, warnings
    
    def validate_spending_pattern(self, spending_pattern: SpendingPattern, amount: Optional[float] = None) -> Tuple[bool, List[str], List[str]]:
        """
        Validate spending pattern
        
        Args:
            spending_pattern: SpendingPattern to validate
            amount: Optional transaction amount for consistency check
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        is_valid = True
        
        # Basic validation (enum ensures valid values)
        if not isinstance(spending_pattern, SpendingPattern):
            errors.append("Invalid spending pattern")
            is_valid = False
        
        # Pattern-amount consistency
        if amount is not None:
            if spending_pattern == SpendingPattern.TYPICAL and amount > 2000:
                warnings.append("High amount marked as 'typical' - please verify")
            elif spending_pattern == SpendingPattern.SUSPICIOUS and amount < 50:
                warnings.append("Low amount marked as 'suspicious' - please verify")
            elif spending_pattern == SpendingPattern.MUCH_HIGHER and amount < 100:
                warnings.append("Low amount marked as 'much higher' - please verify")
        
        # Pattern-specific warnings
        if spending_pattern == SpendingPattern.SUSPICIOUS:
            warnings.append("Suspicious spending patterns will increase fraud risk score")
        
        return is_valid, errors, warnings
    
    def validate_user_input(self, user_input: UserTransactionInput) -> ValidationResult:
        """
        Comprehensive validation of user input
        
        Args:
            user_input: UserTransactionInput object to validate
            
        Returns:
            ValidationResult object with validation details
        """
        all_errors = []
        all_warnings = []
        suggestions = {}
        
        # Validate transaction amount
        amount_valid, amount_errors, amount_warnings = self.validate_transaction_amount(
            user_input.transaction_amount, user_input.merchant_category
        )
        all_errors.extend(amount_errors)
        all_warnings.extend(amount_warnings)
        
        # Validate time context
        time_valid, time_errors, time_warnings = self.validate_time_context(
            user_input.time_context, user_input.merchant_category
        )
        all_errors.extend(time_errors)
        all_warnings.extend(time_warnings)
        
        # Validate merchant category
        merchant_valid, merchant_errors, merchant_warnings = self.validate_merchant_category(
            user_input.merchant_category
        )
        all_errors.extend(merchant_errors)
        all_warnings.extend(merchant_warnings)
        
        # Validate location risk
        location_valid, location_errors, location_warnings = self.validate_location_risk(
            user_input.location_risk, user_input.merchant_category
        )
        all_errors.extend(location_errors)
        all_warnings.extend(location_warnings)
        
        # Validate spending pattern
        spending_valid, spending_errors, spending_warnings = self.validate_spending_pattern(
            user_input.spending_pattern, user_input.transaction_amount
        )
        all_errors.extend(spending_errors)
        all_warnings.extend(spending_warnings)
        
        # Cross-field validation
        cross_errors, cross_warnings, cross_suggestions = self._validate_cross_field_consistency(user_input)
        all_errors.extend(cross_errors)
        all_warnings.extend(cross_warnings)
        suggestions.update(cross_suggestions)
        
        # Check fraud patterns
        fraud_warnings = self._check_fraud_patterns(user_input)
        all_warnings.extend(fraud_warnings)
        
        # Generate suggestions
        input_suggestions = self._generate_input_suggestions(user_input)
        suggestions.update(input_suggestions)
        
        is_valid = (amount_valid and time_valid and merchant_valid and 
                   location_valid and spending_valid and len(all_errors) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=suggestions
        )
    
    def _validate_cross_field_consistency(self, user_input: UserTransactionInput) -> Tuple[List[str], List[str], Dict[str, str]]:
        """Validate consistency across multiple fields"""
        errors = []
        warnings = []
        suggestions = {}
        
        # Amount vs spending pattern consistency
        amount = user_input.transaction_amount
        spending = user_input.spending_pattern
        
        if spending == SpendingPattern.TYPICAL and amount > 1000:
            warnings.append("High amount inconsistent with 'typical' spending pattern")
            suggestions['spending_pattern'] = "Consider 'slightly_higher' or 'much_higher' for amounts over $1000"
        
        # Time vs merchant consistency
        hour = user_input.time_context.hour_of_day
        merchant = user_input.merchant_category
        
        if merchant == MerchantCategory.RESTAURANT and (hour < 6 or hour > 23):
            warnings.append("Restaurant transaction at unusual hour")
        elif merchant == MerchantCategory.GROCERY and (hour < 6 or hour > 23):
            warnings.append("Grocery transaction outside typical hours")
        
        # Location vs spending pattern consistency
        location = user_input.location_risk
        if location == LocationRisk.FOREIGN_COUNTRY and spending == SpendingPattern.TYPICAL:
            warnings.append("Foreign transactions are rarely 'typical' - consider higher risk pattern")
            suggestions['spending_pattern'] = "Consider 'slightly_higher' for foreign transactions"
        
        return errors, warnings, suggestions
    
    def _check_fraud_patterns(self, user_input: UserTransactionInput) -> List[str]:
        """Check for known fraud patterns"""
        warnings = []
        
        for pattern in self.fraud_patterns:
            if pattern['conditions'](user_input):
                if user_input.transaction_amount >= pattern['amount_threshold']:
                    warnings.append(pattern['warning'])
        
        return warnings
    
    def _generate_input_suggestions(self, user_input: UserTransactionInput) -> Dict[str, str]:
        """Generate intelligent suggestions for input improvement"""
        suggestions = {}
        
        # Amount suggestions
        amount = user_input.transaction_amount
        merchant = user_input.merchant_category
        
        if merchant in self.merchant_amount_ranges:
            min_amount, max_amount = self.merchant_amount_ranges[merchant]
            if amount > max_amount * 2:
                suggestions['transaction_amount'] = f"Consider if ${amount:.2f} is correct for {merchant.value} - typical range is ${min_amount:.0f}-${max_amount:.0f}"
        
        # Time suggestions
        hour = user_input.time_context.hour_of_day
        if hour >= 23 or hour <= 5:
            suggestions['time_context'] = "Late night transactions may be flagged - verify timing is correct"
        
        # Merchant suggestions based on amount
        if merchant == MerchantCategory.OTHER and amount > 100:
            suggestions['merchant_category'] = "Consider selecting a more specific merchant category for better accuracy"
        
        return suggestions
    
    def get_merchant_suggestions(self, amount: float) -> List[MerchantCategory]:
        """Suggest merchant categories based on transaction amount"""
        suggestions = []
        
        for merchant, (min_amount, max_amount) in self.merchant_amount_ranges.items():
            if min_amount <= amount <= max_amount * 1.5:  # Allow some flexibility
                suggestions.append(merchant)
        
        # Sort by how well the amount fits the typical range
        def fit_score(merchant):
            min_amount, max_amount = self.merchant_amount_ranges[merchant]
            mid_point = (min_amount + max_amount) / 2
            return abs(amount - mid_point) / max_amount
        
        suggestions.sort(key=fit_score)
        return suggestions[:3]  # Return top 3 suggestions
    
    def get_time_suggestions(self, merchant_category: MerchantCategory) -> Dict[str, Any]:
        """Get time-related suggestions for a merchant category"""
        if merchant_category in self.merchant_operating_hours:
            open_hour, close_hour = self.merchant_operating_hours[merchant_category]
            
            # Suggest peak hours (simplified)
            if merchant_category == MerchantCategory.RESTAURANT:
                peak_hours = [12, 13, 18, 19, 20]  # Lunch and dinner
            elif merchant_category == MerchantCategory.GROCERY:
                peak_hours = [10, 11, 17, 18, 19]  # Morning and evening
            else:
                peak_hours = list(range(max(9, open_hour), min(18, close_hour)))
            
            return {
                'operating_hours': (open_hour, close_hour),
                'peak_hours': peak_hours,
                'suggestion': f"Typical {merchant_category.value} hours: {open_hour}:00 - {close_hour}:00"
            }
        
        return {
            'operating_hours': (0, 23),
            'peak_hours': [9, 10, 11, 14, 15, 16, 17, 18],
            'suggestion': "Most transactions occur during business hours (9 AM - 6 PM)"
        }
    
    def get_risk_level_suggestions(self, merchant_category: MerchantCategory, amount: float) -> Dict[str, str]:
        """Get suggestions for risk level assessment"""
        suggestions = {}
        
        # Location risk suggestions
        if amount > 1000:
            suggestions['location_risk'] = "High-value transactions should use 'slightly_unusual' or higher if location is not completely normal"
        
        # Spending pattern suggestions
        if merchant_category in self.merchant_amount_ranges:
            min_amount, max_amount = self.merchant_amount_ranges[merchant_category]
            
            if amount <= max_amount:
                suggestions['spending_pattern'] = "Amount appears typical for this merchant category"
            elif amount <= max_amount * 2:
                suggestions['spending_pattern'] = "Consider 'slightly_higher' for this amount"
            elif amount <= max_amount * 4:
                suggestions['spending_pattern'] = "Consider 'much_higher' for this amount"
            else:
                suggestions['spending_pattern'] = "Consider 'suspicious' for unusually high amounts"
        
        return suggestions
    
    def validate_and_suggest(self, **kwargs) -> Dict[str, Any]:
        """
        Validate partial input and provide suggestions
        
        Args:
            **kwargs: Partial input data
            
        Returns:
            Dictionary with validation results and suggestions
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': {}
        }
        
        # Validate individual fields if provided
        if 'transaction_amount' in kwargs:
            amount = kwargs['transaction_amount']
            merchant = kwargs.get('merchant_category')
            
            if isinstance(merchant, str):
                try:
                    merchant = MerchantCategory(merchant)
                except ValueError:
                    merchant = None
            
            is_valid, errors, warnings = self.validate_transaction_amount(amount, merchant)
            result['is_valid'] &= is_valid
            result['errors'].extend(errors)
            result['warnings'].extend(warnings)
            
            if amount > 0:
                merchant_suggestions = self.get_merchant_suggestions(amount)
                result['suggestions']['merchant_category'] = [m.value for m in merchant_suggestions]
        
        # Add time suggestions if merchant is provided
        if 'merchant_category' in kwargs:
            merchant_str = kwargs['merchant_category']
            try:
                merchant = MerchantCategory(merchant_str)
                time_suggestions = self.get_time_suggestions(merchant)
                result['suggestions']['time_context'] = time_suggestions
            except ValueError:
                pass
        
        # Add risk suggestions if both merchant and amount are provided
        if 'merchant_category' in kwargs and 'transaction_amount' in kwargs:
            try:
                merchant = MerchantCategory(kwargs['merchant_category'])
                amount = kwargs['transaction_amount']
                risk_suggestions = self.get_risk_level_suggestions(merchant, amount)
                result['suggestions'].update(risk_suggestions)
            except (ValueError, TypeError):
                pass
        
        return result