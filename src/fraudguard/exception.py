import sys
import traceback
from fraudguard.logger import fraud_logger

class FraudGuardException(Exception):
    """Custom exception class for FraudGuard application"""
    
    def __init__(self, error_message: str, error_details: sys = None):
        super().__init__(error_message)
        self.error_message = error_message
        
        if error_details:
            _, _, exc_tb = error_details.exc_info()
            if exc_tb:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                self.error_message = f"Error occurred in script: [{file_name}] line number: [{line_number}] error message: [{error_message}]"
        
        # Log the exception
        fraud_logger.error(f"FraudGuardException: {self.error_message}")
        fraud_logger.error(f"Traceback: {traceback.format_exc()}")
    
    def __str__(self):
        return self.error_message
