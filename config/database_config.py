"""
Database configuration for the Chargeback Management System.
"""
import os
from urllib.parse import quote_plus

# Database connection settings
# These can be overridden by environment variables for different environments

DATABASE_CONFIG = {
    'type': os.getenv('DB_TYPE', 'sqlite'),  # sqlite, postgresql, mssql
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'chargeback_db'),
    'username': os.getenv('DB_USER', 'admin'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'driver': os.getenv('DB_DRIVER', 'postgresql+psycopg2'),
}

def get_connection_string():
    """
    Generate database connection string based on configuration.
    
    Returns:
        str: SQLAlchemy connection string
    """
    db_type = DATABASE_CONFIG['type']
    
    if db_type == 'sqlite':
        # For SQLite, use a local file
        return 'sqlite:///chargeback_db.sqlite'
    
    elif db_type == 'postgresql':
        username = DATABASE_CONFIG['username']
        password = quote_plus(DATABASE_CONFIG['password'])
        host = DATABASE_CONFIG['host']
        port = DATABASE_CONFIG['port']
        database = DATABASE_CONFIG['database']
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    elif db_type == 'mssql':
        username = DATABASE_CONFIG['username']
        password = quote_plus(DATABASE_CONFIG['password'])
        host = DATABASE_CONFIG['host']
        database = DATABASE_CONFIG['database']
        driver = quote_plus('ODBC Driver 17 for SQL Server')
        return f"mssql+pyodbc://{username}:{password}@{host}/{database}?driver={driver}"
    
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

# Table names
TABLES = {
    'transactions': 'transactions',
    'chargebacks': 'chargebacks',
    'products': 'products',
    'customers': 'customers',
    'channels': 'channels',
    'reconciliation': 'reconciliation_mapping',
}
