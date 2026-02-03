"""
Data validation module for ensuring data quality and integrity.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validate data quality and integrity for chargeback system.
    """
    
    def __init__(self):
        """Initialize DataValidator instance."""
        self.validation_results = {}
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, type]) -> bool:
        """
        Validate dataframe schema against expected schema.
        
        Args:
            df: Dataframe to validate
            expected_schema: Dictionary mapping column names to expected types
            
        Returns:
            bool: True if schema is valid
        """
        logger.info("Validating data schema")
        errors = []
        
        # Check for missing columns
        missing_cols = set(expected_schema.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(expected_schema.keys())
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
        
        # Check column types
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                actual_type = df[col].dtype
                # Flexible type checking
                if expected_type == str and actual_type != 'object':
                    errors.append(f"Column {col}: expected string, got {actual_type}")
                elif expected_type in [int, float] and not np.issubdtype(actual_type, np.number):
                    errors.append(f"Column {col}: expected numeric, got {actual_type}")
        
        if errors:
            logger.error(f"Schema validation failed: {errors}")
            self.validation_results['schema'] = {'valid': False, 'errors': errors}
            return False
        
        logger.info("Schema validation passed")
        self.validation_results['schema'] = {'valid': True, 'errors': []}
        return True
    
    def validate_required_fields(self, df: pd.DataFrame, required_fields: List[str]) -> bool:
        """
        Validate that required fields are present and non-null.
        
        Args:
            df: Dataframe to validate
            required_fields: List of required field names
            
        Returns:
            bool: True if all required fields are valid
        """
        logger.info("Validating required fields")
        errors = []
        
        for field in required_fields:
            if field not in df.columns:
                errors.append(f"Required field missing: {field}")
            elif df[field].isna().any():
                null_count = df[field].isna().sum()
                errors.append(f"Field {field} has {null_count} null values")
        
        if errors:
            logger.error(f"Required fields validation failed: {errors}")
            self.validation_results['required_fields'] = {'valid': False, 'errors': errors}
            return False
        
        logger.info("Required fields validation passed")
        self.validation_results['required_fields'] = {'valid': True, 'errors': []}
        return True
    
    def validate_data_types(self, df: pd.DataFrame, type_constraints: Dict[str, type]) -> pd.DataFrame:
        """
        Validate and convert data types.
        
        Args:
            df: Dataframe to validate
            type_constraints: Dictionary mapping column names to expected types
            
        Returns:
            pd.DataFrame: Dataframe with corrected types
        """
        logger.info("Validating and converting data types")
        df_copy = df.copy()
        
        for col, expected_type in type_constraints.items():
            if col not in df_copy.columns:
                continue
            
            try:
                if expected_type == str:
                    df_copy[col] = df_copy[col].astype(str)
                elif expected_type == int:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Int64')
                elif expected_type == float:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif expected_type == 'datetime':
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                elif expected_type == bool:
                    df_copy[col] = df_copy[col].astype(bool)
            except Exception as e:
                logger.warning(f"Failed to convert column {col} to {expected_type}: {str(e)}")
        
        return df_copy
    
    def validate_ranges(self, df: pd.DataFrame, range_constraints: Dict[str, Dict]) -> bool:
        """
        Validate that numeric values fall within expected ranges.
        
        Args:
            df: Dataframe to validate
            range_constraints: Dictionary mapping column names to range constraints
                              e.g., {'amount': {'min': 0, 'max': 10000}}
            
        Returns:
            bool: True if all ranges are valid
        """
        logger.info("Validating value ranges")
        errors = []
        
        for col, constraints in range_constraints.items():
            if col not in df.columns:
                continue
            
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None:
                violations = df[df[col] < min_val]
                if len(violations) > 0:
                    errors.append(f"Column {col}: {len(violations)} values below minimum {min_val}")
            
            if max_val is not None:
                violations = df[df[col] > max_val]
                if len(violations) > 0:
                    errors.append(f"Column {col}: {len(violations)} values above maximum {max_val}")
        
        if errors:
            logger.warning(f"Range validation issues: {errors}")
            self.validation_results['ranges'] = {'valid': False, 'errors': errors}
            return False
        
        logger.info("Range validation passed")
        self.validation_results['ranges'] = {'valid': True, 'errors': []}
        return True
    
    def validate_uniqueness(self, df: pd.DataFrame, unique_fields: List[str]) -> bool:
        """
        Validate that specified fields contain unique values.
        
        Args:
            df: Dataframe to validate
            unique_fields: List of fields that should be unique
            
        Returns:
            bool: True if uniqueness constraints are satisfied
        """
        logger.info("Validating uniqueness constraints")
        errors = []
        
        for field in unique_fields:
            if field not in df.columns:
                continue
            
            duplicates = df[field].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Field {field} has {duplicates} duplicate values")
        
        if errors:
            logger.warning(f"Uniqueness validation issues: {errors}")
            self.validation_results['uniqueness'] = {'valid': False, 'errors': errors}
            return False
        
        logger.info("Uniqueness validation passed")
        self.validation_results['uniqueness'] = {'valid': True, 'errors': []}
        return True
    
    def validate_referential_integrity(self, df: pd.DataFrame, 
                                      reference_df: pd.DataFrame,
                                      foreign_key: str,
                                      reference_key: str) -> bool:
        """
        Validate referential integrity between dataframes.
        
        Args:
            df: Dataframe with foreign key
            reference_df: Reference dataframe with primary key
            foreign_key: Foreign key column name
            reference_key: Primary key column name in reference dataframe
            
        Returns:
            bool: True if referential integrity is maintained
        """
        logger.info(f"Validating referential integrity: {foreign_key} -> {reference_key}")
        
        orphaned = df[~df[foreign_key].isin(reference_df[reference_key])]
        
        if len(orphaned) > 0:
            logger.warning(f"Found {len(orphaned)} orphaned records")
            self.validation_results['referential_integrity'] = {
                'valid': False,
                'orphaned_count': len(orphaned)
            }
            return False
        
        logger.info("Referential integrity validation passed")
        self.validation_results['referential_integrity'] = {'valid': True, 'orphaned_count': 0}
        return True
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive validation report.
        
        Returns:
            Dict: Validation results
        """
        return self.validation_results
    
    def validate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate transaction data.
        
        Args:
            df: Transaction dataframe
            
        Returns:
            pd.DataFrame: Validated and cleaned transaction data
        """
        logger.info("Validating transaction data")
        
        required_fields = ['transaction_id', 'transaction_date', 'amount', 'customer_id']
        self.validate_required_fields(df, required_fields)
        
        type_constraints = {
            'transaction_id': str,
            'transaction_date': 'datetime',
            'amount': float,
            'customer_id': str,
            'product_id': str,
            'channel_id': str,
        }
        df = self.validate_data_types(df, type_constraints)
        
        range_constraints = {
            'amount': {'min': 0, 'max': 1000000}
        }
        self.validate_ranges(df, range_constraints)
        
        return df
    
    def validate_chargebacks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate chargeback data.
        
        Args:
            df: Chargeback dataframe
            
        Returns:
            pd.DataFrame: Validated and cleaned chargeback data
        """
        logger.info("Validating chargeback data")
        
        required_fields = ['chargeback_id', 'chargeback_date', 'amount', 'reason_code']
        self.validate_required_fields(df, required_fields)
        
        type_constraints = {
            'chargeback_id': str,
            'chargeback_date': 'datetime',
            'amount': float,
            'reason_code': str,
            'status': str,
        }
        df = self.validate_data_types(df, type_constraints)
        
        range_constraints = {
            'amount': {'min': 0, 'max': 1000000}
        }
        self.validate_ranges(df, range_constraints)
        
        return df
