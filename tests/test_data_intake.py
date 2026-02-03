"""
Unit tests for data intake module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_intake import DataIngestion, DataValidator, DataTransformer


class TestDataIngestion:
    """Test cases for DataIngestion class."""
    
    def test_ingest_csv(self, tmp_path):
        """Test CSV ingestion."""
        # Create sample CSV
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        csv_file = tmp_path / "test.csv"
        test_data.to_csv(csv_file, index=False)
        
        # Test ingestion
        ingestion = DataIngestion()
        result = ingestion.ingest_csv(csv_file)
        
        assert len(result) == 3
        assert list(result.columns) == ['id', 'value']
        assert result['value'].sum() == 60
    
    def test_ingest_json(self, tmp_path):
        """Test JSON ingestion."""
        # Create sample JSON
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        json_file = tmp_path / "test.json"
        test_data.to_json(json_file, orient='records')
        
        # Test ingestion
        ingestion = DataIngestion()
        result = ingestion.ingest_json(json_file)
        
        assert len(result) == 3
        assert 'id' in result.columns
        assert 'value' in result.columns


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_validate_required_fields(self):
        """Test required fields validation."""
        validator = DataValidator()
        
        # Valid data
        valid_df = pd.DataFrame({
            'field1': [1, 2, 3],
            'field2': ['a', 'b', 'c']
        })
        assert validator.validate_required_fields(valid_df, ['field1', 'field2']) is True
        
        # Missing field - validator logs error but returns False instead of raising
        invalid_df = pd.DataFrame({
            'field1': [1, 2, 3]
        })
        assert validator.validate_required_fields(invalid_df, ['field1', 'field3']) is False
    
    def test_validate_data_types(self):
        """Test data type validation and conversion."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'int_col': ['1', '2', '3'],
            'float_col': ['1.5', '2.5', '3.5'],
            'str_col': [1, 2, 3]
        })
        
        type_constraints = {
            'int_col': int,
            'float_col': float,
            'str_col': str
        }
        
        result = validator.validate_data_types(df, type_constraints)
        
        assert result['int_col'].dtype == 'Int64'
        assert result['float_col'].dtype == 'float64'
        # str type can have various representations in different pandas versions
        assert 'str' in str(result['str_col'].dtype).lower() or result['str_col'].dtype == 'object'
    
    def test_validate_ranges(self):
        """Test range validation."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'amount': [10, 50, 100]
        })
        
        range_constraints = {
            'amount': {'min': 0, 'max': 1000}
        }
        
        assert validator.validate_ranges(df, range_constraints) is True
        
        # Test with invalid range
        invalid_df = pd.DataFrame({
            'amount': [-10, 50, 100]
        })
        
        assert validator.validate_ranges(invalid_df, range_constraints) is False


class TestDataTransformer:
    """Test cases for DataTransformer class."""
    
    def test_normalize_amounts(self):
        """Test amount normalization."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'amount': [10.123456, 20.987654, 30.555555]
        })
        
        result = transformer.normalize_amounts(df, ['amount'])
        
        assert all(result['amount'] == result['amount'].round(2))
        assert result['amount'][0] == 10.12
    
    def test_standardize_dates(self):
        """Test date standardization."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'date': ['2026-01-01', '2026-01-02', '2026-01-03']
        })
        
        result = transformer.standardize_dates(df, ['date'])
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_clean_data(self):
        """Test data cleaning."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],  # Has duplicate
            'col2': [1, np.nan, 3, 4],  # Has missing
            'col3': ['a', 'b', 'b', 'c']
        })
        
        result = transformer.clean_data(df)
        
        # Should have removed duplicates
        assert len(result) <= len(df)
        # Should have handled missing values
        assert result['col2'].isna().sum() == 0
    
    def test_add_derived_columns(self):
        """Test derived column creation."""
        transformer = DataTransformer()
        
        df = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=3)
        })
        
        result = transformer.add_derived_columns(df)
        
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_day' in result.columns


def test_integration_data_pipeline(tmp_path):
    """Integration test for complete data pipeline."""
    # Create sample data
    transactions = pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
        'transaction_date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'amount': [100.123, 200.456, 300.789],
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'product_id': ['PROD001', 'PROD001', 'PROD002'],
        'channel_id': ['CH001', 'CH002', 'CH001']
    })
    
    # Save to CSV
    csv_file = tmp_path / "transactions.csv"
    transactions.to_csv(csv_file, index=False)
    
    # Complete pipeline
    ingestion = DataIngestion()
    validator = DataValidator()
    transformer = DataTransformer()
    
    # Ingest
    data = ingestion.ingest_csv(csv_file)
    assert len(data) == 3
    
    # Validate
    data = validator.validate_transactions(data)
    assert 'transaction_id' in data.columns
    
    # Transform
    data = transformer.transform_transactions(data)
    assert all(data['amount'] == data['amount'].round(2))
    
    # Check derived columns were added
    assert 'transaction_date_year' in data.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
