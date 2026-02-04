"""
Unit tests for utilities module.
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

from utils import helpers


class TestHelpers:
    """Test cases for helper functions."""
    
    def test_calculate_date_range(self):
        """Test date range calculation."""
        start_date, end_date = helpers.calculate_date_range(days_back=30)
        
        assert isinstance(start_date, datetime)
        assert isinstance(end_date, datetime)
        assert (end_date - start_date).days == 30
    
    def test_normalize_amount(self):
        """Test amount normalization."""
        assert helpers.normalize_amount(10.123456) == 10.12
        assert helpers.normalize_amount(20.999) == 21.00
        assert helpers.normalize_amount(100) == 100.00
    
    def test_generate_hash(self):
        """Test hash generation."""
        hash1 = helpers.generate_hash("test_data")
        hash2 = helpers.generate_hash("test_data")
        hash3 = helpers.generate_hash("different_data")
        
        # Same input should produce same hash
        assert hash1 == hash2
        # Different input should produce different hash
        assert hash1 != hash3
        # Hash should be 64 characters (SHA256)
        assert len(hash1) == 64
    
    def test_safe_divide(self):
        """Test safe division."""
        assert helpers.safe_divide(10, 2) == 5.0
        assert helpers.safe_divide(10, 0) == 0.0  # Default
        assert helpers.safe_divide(10, 0, default=1.0) == 1.0
    
    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        assert helpers.calculate_percentage_change(100, 50) == 100.0
        assert helpers.calculate_percentage_change(50, 100) == -50.0
        assert helpers.calculate_percentage_change(100, 0) == 100.0
        assert helpers.calculate_percentage_change(0, 0) == 0.0
    
    def test_clean_dataframe(self):
        """Test dataframe cleaning."""
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3, np.nan],
            'col2': ['a', 'b', 'b', 'c', np.nan],
            'col3': [1.5, np.nan, 3.5, 4.5, 5.5]
        })
        
        result = helpers.clean_dataframe(df)
        
        # Should remove duplicates
        assert len(result) <= len(df)
        # Should handle missing values
        assert result.isna().sum().sum() == 0
    
    def test_validate_required_columns(self):
        """Test column validation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Valid case
        assert helpers.validate_required_columns(df, ['col1', 'col2']) is True
        
        # Missing column
        with pytest.raises(ValueError):
            helpers.validate_required_columns(df, ['col1', 'col3'])
    
    def test_parse_date(self):
        """Test date parsing."""
        # String formats
        date1 = helpers.parse_date('2026-01-15')
        assert isinstance(date1, datetime)
        
        date2 = helpers.parse_date('01/15/2026')
        assert isinstance(date2, datetime)
        
        # Already datetime
        now = datetime.now()
        date3 = helpers.parse_date(now)
        assert date3 == now
        
        # Invalid
        date4 = helpers.parse_date('invalid_date')
        assert date4 is None
    
    def test_format_currency(self):
        """Test currency formatting."""
        assert helpers.format_currency(1234.56) == '$1,234.56'
        assert helpers.format_currency(1234.56, 'EUR') == '€1,234.56'
        assert helpers.format_currency(0) == '$0.00'
    
    def test_aggregate_by_period(self):
        """Test period aggregation."""
        df = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=30),
            'value': range(1, 31)
        })
        
        # Daily aggregation
        result_daily = helpers.aggregate_by_period(df, 'date', 'value', period='D')
        assert len(result_daily) == 30
        
        # Weekly aggregation
        result_weekly = helpers.aggregate_by_period(df, 'date', 'value', period='W')
        assert len(result_weekly) <= 5  # ~4-5 weeks


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
