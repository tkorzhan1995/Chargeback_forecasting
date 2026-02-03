"""
Configuration settings for the Chargeback Management System.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'sample_data'
OUTPUT_DIR = BASE_DIR / 'output'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Reconciliation settings
RECONCILIATION_CONFIG = {
    'matching_threshold': 0.85,  # Confidence threshold for automatic matching
    'fuzzy_match_threshold': 80,  # Threshold for fuzzy string matching (0-100)
    'time_window_hours': 24,  # Time window for transaction matching
    'amount_tolerance': 0.01,  # Tolerance for amount matching (1%)
}

# Forecasting settings
FORECASTING_CONFIG = {
    'train_test_split': 0.8,
    'cv_folds': 5,
    'forecast_horizon': 30,  # days
    'min_historical_days': 90,
    'seasonal_period': 7,  # weekly seasonality
    'confidence_level': 0.95,
}

# Model settings
MODEL_CONFIG = {
    'random_state': 42,
    'models': ['arima', 'prophet', 'random_forest', 'ensemble'],
    'default_model': 'ensemble',
}

# Power BI export settings
POWERBI_CONFIG = {
    'export_formats': ['csv', 'parquet'],
    'refresh_schedule': 'daily',
    'incremental_refresh': True,
}

# Logging settings
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': str(LOGS_DIR / 'chargeback_system.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file'],
    },
}
