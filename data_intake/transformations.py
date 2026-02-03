"""
Data transformation module for cleaning and preparing data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils import get_logger, clean_dataframe, normalize_amount

logger = get_logger(__name__)


class DataTransformer:
    """
    Transform and clean data for the chargeback system.
    """
    
    def __init__(self):
        """Initialize DataTransformer instance."""
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing duplicates and handling missing values.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data")
        return clean_dataframe(df)
    
    def normalize_amounts(self, df: pd.DataFrame, amount_columns: List[str]) -> pd.DataFrame:
        """
        Normalize amount columns to two decimal places.
        
        Args:
            df: Input dataframe
            amount_columns: List of amount column names
            
        Returns:
            pd.DataFrame: Dataframe with normalized amounts
        """
        logger.info(f"Normalizing amount columns: {amount_columns}")
        df_copy = df.copy()
        
        for col in amount_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(normalize_amount)
        
        return df_copy
    
    def standardize_dates(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Standardize date columns to datetime format.
        
        Args:
            df: Input dataframe
            date_columns: List of date column names
            
        Returns:
            pd.DataFrame: Dataframe with standardized dates
        """
        logger.info(f"Standardizing date columns: {date_columns}")
        df_copy = df.copy()
        
        for col in date_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        return df_copy
    
    def standardize_text(self, df: pd.DataFrame, text_columns: List[str], 
                        lowercase: bool = True, strip: bool = True) -> pd.DataFrame:
        """
        Standardize text columns.
        
        Args:
            df: Input dataframe
            text_columns: List of text column names
            lowercase: Convert to lowercase
            strip: Strip whitespace
            
        Returns:
            pd.DataFrame: Dataframe with standardized text
        """
        logger.info(f"Standardizing text columns: {text_columns}")
        df_copy = df.copy()
        
        for col in text_columns:
            if col in df_copy.columns:
                if strip:
                    df_copy[col] = df_copy[col].str.strip()
                if lowercase:
                    df_copy[col] = df_copy[col].str.lower()
        
        return df_copy
    
    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns for analysis.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with derived columns
        """
        logger.info("Adding derived columns")
        df_copy = df.copy()
        
        # Add date components if date column exists
        date_cols = df_copy.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df_copy[f'{col}_year'] = df_copy[col].dt.year
            df_copy[f'{col}_month'] = df_copy[col].dt.month
            df_copy[f'{col}_day'] = df_copy[col].dt.day
            df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
            df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter
        
        return df_copy
    
    def aggregate_data(self, df: pd.DataFrame, group_by: List[str], 
                      agg_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Aggregate data by specified columns.
        
        Args:
            df: Input dataframe
            group_by: List of columns to group by
            agg_dict: Dictionary of aggregation functions
            
        Returns:
            pd.DataFrame: Aggregated dataframe
        """
        logger.info(f"Aggregating data by: {group_by}")
        return df.groupby(group_by).agg(agg_dict).reset_index()
    
    def pivot_data(self, df: pd.DataFrame, index: str, columns: str, 
                  values: str, aggfunc: str = 'sum') -> pd.DataFrame:
        """
        Pivot data for analysis.
        
        Args:
            df: Input dataframe
            index: Column to use as index
            columns: Column to use as columns
            values: Column to use as values
            aggfunc: Aggregation function
            
        Returns:
            pd.DataFrame: Pivoted dataframe
        """
        logger.info(f"Pivoting data: index={index}, columns={columns}, values={values}")
        return df.pivot_table(index=index, columns=columns, values=values, 
                             aggfunc=aggfunc, fill_value=0)
    
    def remove_outliers(self, df: pd.DataFrame, column: str, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from a numeric column.
        
        Args:
            df: Input dataframe
            column: Column name
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        logger.info(f"Removing outliers from {column} using {method} method")
        df_copy = df.copy()
        
        if method == 'iqr':
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_copy[column].dropna()))
            df_copy = df_copy[(z_scores < threshold)]
        
        logger.info(f"Removed {len(df) - len(df_copy)} outliers")
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], 
                          method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input dataframe
            columns: List of categorical columns
            method: Encoding method ('onehot' or 'label')
            
        Returns:
            pd.DataFrame: Dataframe with encoded columns
        """
        logger.info(f"Encoding categorical columns: {columns} using {method}")
        df_copy = df.copy()
        
        if method == 'onehot':
            df_copy = pd.get_dummies(df_copy, columns=columns, prefix=columns)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                if col in df_copy.columns:
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        
        return df_copy
    
    def transform_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete transformation pipeline for transaction data.
        
        Args:
            df: Raw transaction dataframe
            
        Returns:
            pd.DataFrame: Transformed transaction data
        """
        logger.info("Transforming transaction data")
        
        # Clean data
        df = self.clean_data(df)
        
        # Normalize amounts
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        if amount_cols:
            df = self.normalize_amounts(df, amount_cols)
        
        # Standardize dates
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df = self.standardize_dates(df, date_cols)
        
        # Standardize text
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            df = self.standardize_text(df, text_cols)
        
        # Add derived columns
        df = self.add_derived_columns(df)
        
        logger.info("Transaction data transformation complete")
        return df
    
    def transform_chargebacks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete transformation pipeline for chargeback data.
        
        Args:
            df: Raw chargeback dataframe
            
        Returns:
            pd.DataFrame: Transformed chargeback data
        """
        logger.info("Transforming chargeback data")
        
        # Clean data
        df = self.clean_data(df)
        
        # Normalize amounts
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        if amount_cols:
            df = self.normalize_amounts(df, amount_cols)
        
        # Standardize dates
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df = self.standardize_dates(df, date_cols)
        
        # Standardize text
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            df = self.standardize_text(df, text_cols)
        
        # Add derived columns
        df = self.add_derived_columns(df)
        
        logger.info("Chargeback data transformation complete")
        return df
