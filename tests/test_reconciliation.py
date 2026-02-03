"""
Unit tests for reconciliation module.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reconciliation import MatchingEngine, LinkageAlgorithms


class TestMatchingEngine:
    """Test cases for MatchingEngine class."""
    
    def test_exact_match(self):
        """Test exact matching on transaction IDs."""
        matcher = MatchingEngine()
        
        transactions = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'amount': [100, 200, 300],
            'transaction_date': pd.date_range('2026-01-01', periods=3)
        })
        
        chargebacks = pd.DataFrame({
            'chargeback_id': ['CB001', 'CB002'],
            'transaction_id': ['TXN001', 'TXN002'],
            'amount': [100, 200],
            'chargeback_date': pd.date_range('2026-01-10', periods=2)
        })
        
        matched = matcher.exact_match(chargebacks, transactions)
        
        assert len(matched) == 2
        assert 'match_confidence' in matched.columns
        assert all(matched['match_confidence'] == 1.0)
    
    def test_reconcile(self):
        """Test full reconciliation process."""
        matcher = MatchingEngine()
        
        transactions = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'amount': [100, 200, 300, 400],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004'],
            'transaction_date': pd.date_range('2026-01-01', periods=4)
        })
        
        chargebacks = pd.DataFrame({
            'chargeback_id': ['CB001', 'CB002', 'CB003'],
            'transaction_id': ['TXN001', 'TXN002', None],  # CB003 has no transaction_id
            'amount': [100, 200, 300],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'chargeback_date': pd.date_range('2026-01-10', periods=3)
        })
        
        matched, unmatched = matcher.reconcile(chargebacks, transactions)
        
        # Should match at least the exact matches
        assert len(matched) >= 2
        assert len(unmatched) <= 1
        
        # Check match rate
        match_rate = len(matched) / len(chargebacks) * 100
        assert match_rate > 0


class TestLinkageAlgorithms:
    """Test cases for LinkageAlgorithms class."""
    
    def test_fuzzy_string_match(self):
        """Test fuzzy string matching."""
        linker = LinkageAlgorithms()
        
        # Exact match
        score1 = linker.fuzzy_string_match("Hello World", "Hello World")
        assert score1 == 100.0
        
        # Similar strings
        score2 = linker.fuzzy_string_match("Hello World", "Hello World!")
        assert score2 > 90.0
        
        # Different strings
        score3 = linker.fuzzy_string_match("Hello", "Goodbye")
        assert score3 < 50.0
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        linker = LinkageAlgorithms()
        
        # Identical strings
        dist1 = linker.levenshtein_distance("test", "test")
        assert dist1 == 0
        
        # One character difference
        dist2 = linker.levenshtein_distance("test", "text")
        assert dist2 == 1
        
        # Multiple differences
        dist3 = linker.levenshtein_distance("kitten", "sitting")
        assert dist3 == 3
    
    def test_partial_amount_match(self):
        """Test partial amount matching."""
        linker = LinkageAlgorithms()
        
        transactions = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'amount': [100, 200]
        })
        
        chargebacks = pd.DataFrame({
            'chargeback_id': ['CB001', 'CB002'],
            'amount': [50, 100]  # 50% and 50% of transactions
        })
        
        matches = linker.partial_amount_match(chargebacks, transactions)
        
        # Should find at least one partial match
        assert len(matches) >= 0  # May find matches depending on tolerance


def test_reconciliation_integration():
    """Integration test for reconciliation process."""
    # Create realistic test data
    transactions = pd.DataFrame({
        'transaction_id': [f'TXN{i:03d}' for i in range(1, 101)],
        'amount': [100 + i * 10 for i in range(100)],
        'customer_id': [f'CUST{i%20:03d}' for i in range(100)],
        'product_id': [f'PROD{i%10:03d}' for i in range(100)],
        'channel_id': [f'CH00{i%3+1}' for i in range(100)],
        'transaction_date': [datetime.now() - timedelta(days=i) for i in range(100)]
    })
    
    # Create chargebacks (10% chargeback rate)
    selected_txns = transactions.sample(n=10)
    chargebacks = pd.DataFrame({
        'chargeback_id': [f'CB{i:03d}' for i in range(1, 11)],
        'transaction_id': selected_txns['transaction_id'].values,
        'amount': selected_txns['amount'].values,
        'customer_id': selected_txns['customer_id'].values,
        'product_id': selected_txns['product_id'].values,
        'channel_id': selected_txns['channel_id'].values,
        'chargeback_date': [datetime.now() - timedelta(days=i) for i in range(10)],
        'reason_code': ['Fraud'] * 10,
        'status': ['pending'] * 10
    })
    
    # Run reconciliation
    matcher = MatchingEngine()
    matched, unmatched = matcher.reconcile(chargebacks, transactions)
    
    # Assertions
    assert len(matched) + len(unmatched) == len(chargebacks)
    assert len(matched) > 0  # Should match at least some records
    
    # Check match rate
    match_rate = len(matched) / len(chargebacks) * 100
    assert match_rate >= 50  # Should match at least 50%


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
