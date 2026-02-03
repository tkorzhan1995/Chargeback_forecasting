"""Unit tests for analysis modules."""
import unittest
from datetime import datetime
from chargeback_forecasting.models.chargeback import Chargeback
from chargeback_forecasting.analysis.historical_rates import HistoricalRateCalculator
from chargeback_forecasting.analysis.win_loss import WinLossAnalyzer
from chargeback_forecasting.analysis.key_drivers import KeyDriverAnalyzer


class TestHistoricalRateCalculator(unittest.TestCase):
    """Test HistoricalRateCalculator."""
    
    def setUp(self):
        """Set up test data."""
        self.chargebacks = [
            Chargeback("TXN001", 100.0, datetime(2023, 1, 15), "FRAUD", "MERCH001", "won"),
            Chargeback("TXN002", 200.0, datetime(2023, 1, 20), "FRAUD", "MERCH001", "lost"),
            Chargeback("TXN003", 150.0, datetime(2023, 2, 10), "DEFECTIVE", "MERCH002", "won"),
        ]
        self.total_transactions = {
            "2023-01": 1000,
            "2023-02": 900
        }
    
    def test_calculate_monthly_rates(self):
        """Test monthly rate calculation."""
        calculator = HistoricalRateCalculator(self.chargebacks, self.total_transactions)
        rates = calculator.calculate_monthly_rates()
        
        self.assertEqual(len(rates), 2)
        self.assertEqual(rates[0].period, "2023-01")
        self.assertEqual(rates[0].total_chargebacks, 2)
        self.assertEqual(rates[0].chargeback_rate, 2/1000)
    
    def test_get_average_rate(self):
        """Test average rate calculation."""
        calculator = HistoricalRateCalculator(self.chargebacks, self.total_transactions)
        avg_rate = calculator.get_average_rate()
        
        expected_rate = 3 / (1000 + 900)
        self.assertAlmostEqual(avg_rate, expected_rate)
    
    def test_calculate_rate_by_merchant(self):
        """Test merchant-specific rate calculation."""
        calculator = HistoricalRateCalculator(self.chargebacks, self.total_transactions)
        rates = calculator.calculate_rate_by_merchant("MERCH001")
        
        self.assertEqual(len(rates), 1)
        self.assertEqual(rates[0].merchant_id, "MERCH001")
        self.assertEqual(rates[0].total_chargebacks, 2)


class TestWinLossAnalyzer(unittest.TestCase):
    """Test WinLossAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        self.chargebacks = [
            Chargeback("TXN001", 100.0, datetime(2023, 1, 15), "FRAUD", "MERCH001", "won"),
            Chargeback("TXN002", 200.0, datetime(2023, 1, 20), "FRAUD", "MERCH001", "lost"),
            Chargeback("TXN003", 150.0, datetime(2023, 2, 10), "DEFECTIVE", "MERCH002", "won"),
            Chargeback("TXN004", 120.0, datetime(2023, 2, 15), "FRAUD", "MERCH001", "won"),
        ]
    
    def test_calculate_monthly_ratios(self):
        """Test monthly win/loss ratio calculation."""
        analyzer = WinLossAnalyzer(self.chargebacks)
        ratios = analyzer.calculate_monthly_ratios()
        
        self.assertEqual(len(ratios), 2)
        self.assertEqual(ratios[0].period, "2023-01")
        self.assertEqual(ratios[0].total_disputes, 2)
        self.assertEqual(ratios[0].won_disputes, 1)
        self.assertEqual(ratios[0].lost_disputes, 1)
    
    def test_get_overall_win_rate(self):
        """Test overall win rate calculation."""
        analyzer = WinLossAnalyzer(self.chargebacks)
        win_rate = analyzer.get_overall_win_rate()
        
        self.assertEqual(win_rate, 3/4)
    
    def test_get_overall_loss_rate(self):
        """Test overall loss rate calculation."""
        analyzer = WinLossAnalyzer(self.chargebacks)
        loss_rate = analyzer.get_overall_loss_rate()
        
        self.assertEqual(loss_rate, 1/4)


class TestKeyDriverAnalyzer(unittest.TestCase):
    """Test KeyDriverAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        self.chargebacks = [
            Chargeback("TXN001", 100.0, datetime(2023, 1, 15), "FRAUD", "MERCH001", "won"),
            Chargeback("TXN002", 200.0, datetime(2023, 1, 20), "FRAUD", "MERCH001", "lost"),
            Chargeback("TXN003", 150.0, datetime(2023, 2, 10), "DEFECTIVE", "MERCH002", "won"),
        ]
        self.transactions = [
            {"transaction_id": "TXN001", "amount": 100.0, "merchant_id": "MERCH001"},
            {"transaction_id": "TXN002", "amount": 200.0, "merchant_id": "MERCH001"},
            {"transaction_id": "TXN003", "amount": 150.0, "merchant_id": "MERCH002"},
            {"transaction_id": "TXN004", "amount": 80.0, "merchant_id": "MERCH001"},
        ]
    
    def test_analyze_reason_codes(self):
        """Test reason code analysis."""
        analyzer = KeyDriverAnalyzer(self.chargebacks, self.transactions)
        drivers = analyzer.analyze_reason_codes()
        
        self.assertGreater(len(drivers), 0)
        self.assertTrue(any(d.driver_name.startswith("Reason:") for d in drivers))
    
    def test_analyze_merchant_patterns(self):
        """Test merchant pattern analysis."""
        analyzer = KeyDriverAnalyzer(self.chargebacks, self.transactions)
        drivers = analyzer.analyze_merchant_patterns()
        
        self.assertGreater(len(drivers), 0)
        self.assertTrue(any(d.driver_name.startswith("Merchant:") for d in drivers))
    
    def test_analyze_amount_patterns(self):
        """Test amount pattern analysis."""
        analyzer = KeyDriverAnalyzer(self.chargebacks, self.transactions)
        driver = analyzer.analyze_amount_patterns()
        
        self.assertEqual(driver.driver_name, "Transaction Amount")
        self.assertEqual(driver.driver_type, "numerical")
    
    def test_get_top_drivers(self):
        """Test getting top drivers."""
        analyzer = KeyDriverAnalyzer(self.chargebacks, self.transactions)
        top_drivers = analyzer.get_top_drivers(n=5)
        
        self.assertLessEqual(len(top_drivers), 5)


if __name__ == '__main__':
    unittest.main()
