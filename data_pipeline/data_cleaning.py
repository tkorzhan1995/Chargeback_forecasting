"""
Data Cleaning Module

This module handles data cleaning operations for chargeback forecasting.
Includes validation, missing value handling, outlier detection, and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations for chargeback data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataCleaner with configuration.
        
        Args:
            config: Configuration dictionary with cleaning parameters
        """
        self.config = config or {}
        self.missing_threshold = self.config.get('missing_threshold', 0.5)
        self.outlier_std = self.config.get('outlier_std', 3)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning pipeline")
        
        df_clean = df.copy()
        df_clean = self.remove_duplicates(df_clean)
        df_clean = self.handle_missing_values(df_clean)
        df_clean = self.validate_data_types(df_clean)
        df_clean = self.handle_outliers(df_clean)
        
        logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_shape = df.shape[0]
        df_dedup = df.drop_duplicates()
        removed = initial_shape - df_dedup.shape[0]
        
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate rows")
            
        return df_dedup
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on threshold."""
        missing_pct = df.isnull().sum() / len(df)
        
        # Drop columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index
        if len(cols_to_drop) > 0:
            logger.warning(f"Dropping columns with >{self.missing_threshold*100}% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Forward fill remaining missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        # Add data type validation logic here
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers using specified method.
        
        Args:
            df: Input dataframe
            method: 'clip' or 'remove'
            
        Returns:
            Dataframe with outliers handled
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - (self.outlier_std * std)
            upper_bound = mean + (self.outlier_std * std)
            
            if method == 'clip':
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
        return df
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report.
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        return report
