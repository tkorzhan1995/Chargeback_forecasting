"""
Logging configuration for the Chargeback Management System.
"""
import logging
import logging.config
from config.settings import LOGGING_CONFIG

def setup_logging():
    """
    Initialize logging configuration for the application.
    """
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)

def get_logger(name):
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Name of the logger (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
