"""
Helper utilities for the Chargeback Management System.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib

def calculate_date_range(days_back: int = 90) -> tuple:
    """
    Calculate date range for data queries.
    
    Args:
        days_back (int): Number of days to look back
        
    Returns:
        tuple: (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date

def normalize_amount(amount: float) -> float:
    """
    Normalize amount to two decimal places.
    
    Args:
        amount (float): Amount to normalize
        
    Returns:
        float: Normalized amount
    """
    return round(float(amount), 2)

def generate_hash(data: str) -> str:
    """
    Generate SHA256 hash for data.
    
    Args:
        data (str): Data to hash
        
    Returns:
        str: Hexadecimal hash string
    """
    return hashlib.sha256(data.encode()).hexdigest()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division fails
        
    Returns:
        float: Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        current (float): Current value
        previous (float): Previous value
        
    Returns:
        float: Percentage change
    """
    if previous == 0:
        return 0.0 if current == 0 else 100.0
    return ((current - previous) / previous) * 100

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (basic approach)
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns: fill with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)
    
    return df

def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that dataframe contains all required columns.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if all required columns present
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def parse_date(date_str: Any) -> Optional[datetime]:
    """
    Parse date string into datetime object.
    
    Args:
        date_str: Date string or datetime object
        
    Returns:
        Optional[datetime]: Parsed datetime or None
    """
    if pd.isna(date_str):
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    # Try common date formats
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%d-%m-%Y',
        '%Y/%m/%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue
    
    # Try pandas parser as fallback
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format amount as currency string.
    
    Args:
        amount (float): Amount to format
        currency (str): Currency code
        
    Returns:
        str: Formatted currency string
    """
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, currency + ' ')
    return f"{symbol}{amount:,.2f}"

def aggregate_by_period(df: pd.DataFrame, date_column: str, 
                        value_column: str, period: str = 'D') -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of date column
        value_column (str): Name of value column to aggregate
        period (str): Period for aggregation ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
        pd.DataFrame: Aggregated dataframe
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(date_column)
    
    aggregated = df_sorted.groupby(pd.Grouper(key=date_column, freq=period))[value_column].agg([
        ('count', 'count'),
        ('sum', 'sum'),
        ('mean', 'mean'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return aggregated
