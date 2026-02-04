"""Unit tests for forecasting module."""
import unittest
from datetime import datetime
from chargeback_forecasting.models.chargeback import Chargeback
from chargeback_forecasting.forecasting.forecaster import ChargebackForecaster


class TestChargebackForecaster(unittest.TestCase):
    """Test ChargebackForecaster."""
    
    def setUp(self):
        """Set up test data."""
        self.chargebacks = [
            Chargeback("TXN001", 100.0, datetime(2023, 1, 15), "FRAUD", "MERCH001", "won"),
            Chargeback("TXN002", 200.0, datetime(2023, 1, 20), "FRAUD", "MERCH001", "lost"),
            Chargeback("TXN003", 150.0, datetime(2023, 2, 10), "DEFECTIVE", "MERCH002", "won"),
            Chargeback("TXN004", 120.0, datetime(2023, 2, 15), "FRAUD", "MERCH001", "won"),
            Chargeback("TXN005", 180.0, datetime(2023, 3, 8), "NOT_RECEIVED", "MERCH002", "lost"),
            Chargeback("TXN006", 250.0, datetime(2023, 3, 22), "DEFECTIVE", "MERCH001", "won"),
        ]
        self.transactions = [
            {"transaction_id": f"TXN{i:04d}", "amount": 100.0 + i, "merchant_id": "MERCH001"}
            for i in range(100)
        ]
        self.total_transactions = {
            "2023-01": 1000,
            "2023-02": 900,
            "2023-03": 950
        }
    
    def test_forecast_next_period_simple_average(self):
        """Test forecasting with simple average method."""
        forecaster = ChargebackForecaster(
            self.chargebacks,
            self.transactions,
            self.total_transactions
        )
        
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-04",
            expected_transactions=1000,
            method='simple_average'
        )
        
        self.assertEqual(forecast.forecast_period, "2023-04")
        self.assertGreater(forecast.expected_chargebacks, 0)
        self.assertGreater(forecast.expected_chargeback_rate, 0)
        self.assertGreater(forecast.confidence_interval_high, forecast.confidence_interval_low)
    
    def test_forecast_next_period_weighted_average(self):
        """Test forecasting with weighted average method."""
        forecaster = ChargebackForecaster(
            self.chargebacks,
            self.transactions,
            self.total_transactions
        )
        
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-04",
            expected_transactions=1000,
            method='weighted_average'
        )
        
        self.assertEqual(forecast.forecast_period, "2023-04")
        self.assertGreater(forecast.expected_chargebacks, 0)
        self.assertIsNotNone(forecast.expected_win_rate)
        self.assertIsNotNone(forecast.expected_loss_rate)
    
    def test_forecast_next_period_trend(self):
        """Test forecasting with trend method."""
        forecaster = ChargebackForecaster(
            self.chargebacks,
            self.transactions,
            self.total_transactions
        )
        
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-04",
            expected_transactions=1000,
            method='trend'
        )
        
        self.assertEqual(forecast.forecast_period, "2023-04")
        self.assertGreaterEqual(forecast.expected_chargeback_rate, 0)
    
    def test_forecast_multiple_periods(self):
        """Test forecasting multiple periods."""
        forecaster = ChargebackForecaster(
            self.chargebacks,
            self.transactions,
            self.total_transactions
        )
        
        forecasts = forecaster.forecast_multiple_periods(
            num_periods=3,
            expected_transactions_per_period=1000,
            start_period="2023-04"
        )
        
        self.assertEqual(len(forecasts), 3)
        self.assertEqual(forecasts[0].forecast_period, "2023-04")
        self.assertEqual(forecasts[1].forecast_period, "2023-05")
        self.assertEqual(forecasts[2].forecast_period, "2023-06")
    
    def test_forecast_with_empty_data(self):
        """Test forecasting with no historical data."""
        forecaster = ChargebackForecaster([], [], {})
        
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-04",
            expected_transactions=1000
        )
        
        self.assertEqual(forecast.expected_chargebacks, 0.0)
        self.assertEqual(forecast.expected_chargeback_rate, 0.0)
        self.assertIn("No historical data available", forecast.assumptions)
    
    def test_forecast_includes_key_drivers(self):
        """Test that forecast includes key drivers."""
        forecaster = ChargebackForecaster(
            self.chargebacks,
            self.transactions,
            self.total_transactions
        )
        
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-04",
            expected_transactions=1000
        )
        
        self.assertIsInstance(forecast.key_drivers, list)
    
    def test_forecast_includes_assumptions(self):
        """Test that forecast includes assumptions."""
        forecaster = ChargebackForecaster(
            self.chargebacks,
            self.transactions,
            self.total_transactions
        )
        
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-04",
            expected_transactions=1000
        )
        
        self.assertIsInstance(forecast.assumptions, list)
        self.assertGreater(len(forecast.assumptions), 0)


if __name__ == '__main__':
    unittest.main()
