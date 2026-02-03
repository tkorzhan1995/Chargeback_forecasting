"""Unit tests for chargeback data models."""
import unittest
from datetime import datetime
from chargeback_forecasting.models.chargeback import (
    Chargeback, HistoricalRate, WinLossRatio, KeyDriver
)


class TestChargeback(unittest.TestCase):
    """Test Chargeback model."""
    
    def test_chargeback_creation(self):
        """Test creating a chargeback instance."""
        cb = Chargeback(
            transaction_id="TXN001",
            amount=100.0,
            date=datetime(2023, 1, 15),
            reason_code="FRAUD",
            merchant_id="MERCH001",
            status="pending"
        )
        self.assertEqual(cb.transaction_id, "TXN001")
        self.assertEqual(cb.amount, 100.0)
        self.assertEqual(cb.status, "pending")
    
    def test_is_resolved(self):
        """Test is_resolved method."""
        cb_won = Chargeback(
            transaction_id="TXN001",
            amount=100.0,
            date=datetime(2023, 1, 15),
            reason_code="FRAUD",
            merchant_id="MERCH001",
            status="won"
        )
        cb_pending = Chargeback(
            transaction_id="TXN002",
            amount=100.0,
            date=datetime(2023, 1, 15),
            reason_code="FRAUD",
            merchant_id="MERCH001",
            status="pending"
        )
        self.assertTrue(cb_won.is_resolved())
        self.assertFalse(cb_pending.is_resolved())
    
    def test_is_won(self):
        """Test is_won method."""
        cb = Chargeback(
            transaction_id="TXN001",
            amount=100.0,
            date=datetime(2023, 1, 15),
            reason_code="FRAUD",
            merchant_id="MERCH001",
            status="won"
        )
        self.assertTrue(cb.is_won())
        self.assertFalse(cb.is_lost())
    
    def test_is_lost(self):
        """Test is_lost method."""
        cb = Chargeback(
            transaction_id="TXN001",
            amount=100.0,
            date=datetime(2023, 1, 15),
            reason_code="FRAUD",
            merchant_id="MERCH001",
            status="lost"
        )
        self.assertTrue(cb.is_lost())
        self.assertFalse(cb.is_won())


class TestHistoricalRate(unittest.TestCase):
    """Test HistoricalRate model."""
    
    def test_historical_rate_creation(self):
        """Test creating a historical rate instance."""
        rate = HistoricalRate(
            period="2023-01",
            total_transactions=1000,
            total_chargebacks=10,
            chargeback_rate=0.01
        )
        self.assertEqual(rate.period, "2023-01")
        self.assertEqual(rate.total_transactions, 1000)
        self.assertEqual(rate.total_chargebacks, 10)
        self.assertEqual(rate.chargeback_rate, 0.01)
    
    def test_rate_calculation(self):
        """Test automatic rate calculation."""
        rate = HistoricalRate(
            period="2023-01",
            total_transactions=100,
            total_chargebacks=5,
            chargeback_rate=0.0
        )
        self.assertEqual(rate.chargeback_rate, 0.05)


class TestWinLossRatio(unittest.TestCase):
    """Test WinLossRatio model."""
    
    def test_win_loss_ratio_creation(self):
        """Test creating a win/loss ratio instance."""
        ratio = WinLossRatio(
            period="2023-01",
            total_disputes=10,
            won_disputes=6,
            lost_disputes=4,
            win_rate=0.6,
            loss_rate=0.4
        )
        self.assertEqual(ratio.period, "2023-01")
        self.assertEqual(ratio.total_disputes, 10)
        self.assertEqual(ratio.win_rate, 0.6)
    
    def test_rate_calculation(self):
        """Test automatic rate calculation."""
        ratio = WinLossRatio(
            period="2023-01",
            total_disputes=10,
            won_disputes=7,
            lost_disputes=3,
            win_rate=0.0,
            loss_rate=0.0
        )
        self.assertEqual(ratio.win_rate, 0.7)
        self.assertEqual(ratio.loss_rate, 0.3)


class TestKeyDriver(unittest.TestCase):
    """Test KeyDriver model."""
    
    def test_key_driver_creation(self):
        """Test creating a key driver instance."""
        driver = KeyDriver(
            driver_name="Transaction Amount",
            driver_type="numerical",
            impact_score=0.5,
            correlation=0.6,
            description="Transaction amount correlation"
        )
        self.assertEqual(driver.driver_name, "Transaction Amount")
        self.assertEqual(driver.driver_type, "numerical")
        self.assertEqual(driver.impact_score, 0.5)


if __name__ == '__main__':
    unittest.main()
