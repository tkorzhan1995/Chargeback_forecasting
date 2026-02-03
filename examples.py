#!/usr/bin/env python3
"""
Example usage of the Chargeback Forecasting System.

This script demonstrates various ways to use the forecasting system
with custom data and different configurations.
"""
from datetime import datetime
from chargeback_forecasting.models.chargeback import Chargeback
from chargeback_forecasting.forecasting.forecaster import ChargebackForecaster
from chargeback_forecasting.analysis.historical_rates import HistoricalRateCalculator
from chargeback_forecasting.analysis.win_loss import WinLossAnalyzer
from chargeback_forecasting.analysis.key_drivers import KeyDriverAnalyzer


def example_basic_forecast():
    """Example 1: Basic forecasting with minimal data."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Forecasting")
    print("=" * 70)
    
    # Create sample chargeback data
    chargebacks = [
        Chargeback("TXN001", 150.0, datetime(2023, 1, 15), "FRAUD", "MERCH_A", "won"),
        Chargeback("TXN002", 200.0, datetime(2023, 1, 20), "NOT_RECEIVED", "MERCH_B", "lost"),
        Chargeback("TXN003", 100.0, datetime(2023, 2, 10), "DEFECTIVE", "MERCH_A", "won"),
        Chargeback("TXN004", 300.0, datetime(2023, 2, 25), "FRAUD", "MERCH_C", "won"),
    ]
    
    # Create sample transaction data
    transactions = [
        {"transaction_id": f"TXN{i:04d}", "amount": 100 + i, "merchant_id": "MERCH_A"}
        for i in range(50)
    ]
    
    # Total transactions per period
    total_transactions = {
        "2023-01": 500,
        "2023-02": 550
    }
    
    # Create forecaster
    forecaster = ChargebackForecaster(chargebacks, transactions, total_transactions)
    
    # Generate forecast for next month
    forecast = forecaster.forecast_next_period(
        forecast_period="2023-03",
        expected_transactions=600,
        method='weighted_average'
    )
    
    print(f"\nForecast for {forecast.forecast_period}:")
    print(f"  Expected Chargebacks: {forecast.expected_chargebacks:.2f}")
    print(f"  Expected Rate: {forecast.expected_chargeback_rate:.4%}")
    print(f"  Win Rate: {forecast.expected_win_rate:.2%}")
    print(f"  Loss Rate: {forecast.expected_loss_rate:.2%}")
    print()


def example_historical_analysis():
    """Example 2: Detailed historical analysis."""
    print("=" * 70)
    print("EXAMPLE 2: Historical Analysis")
    print("=" * 70)
    
    # Create sample data with more history
    chargebacks = []
    for month in range(1, 7):
        for i in range(3):
            cb = Chargeback(
                transaction_id=f"TXN{month:02d}{i:02d}",
                amount=100.0 + month * 10,
                date=datetime(2023, month, 5 + i * 5),
                reason_code=["FRAUD", "NOT_RECEIVED", "DEFECTIVE"][i % 3],
                merchant_id=f"MERCH_{chr(65 + i % 3)}",
                status=["won", "lost"][i % 2]
            )
            chargebacks.append(cb)
    
    transactions = [{"transaction_id": f"TXN{i:06d}", "amount": 100 + i} for i in range(1000)]
    total_transactions = {f"2023-{m:02d}": 200 for m in range(1, 7)}
    
    # Analyze historical rates
    rate_calc = HistoricalRateCalculator(chargebacks, total_transactions)
    monthly_rates = rate_calc.calculate_monthly_rates()
    
    print("\nHistorical Monthly Rates:")
    for rate in monthly_rates:
        print(f"  {rate.period}: {rate.chargeback_rate:.2%} ({rate.total_chargebacks} chargebacks)")
    
    # Analyze win/loss
    wl_analyzer = WinLossAnalyzer(chargebacks)
    print(f"\nOverall Win Rate: {wl_analyzer.get_overall_win_rate():.2%}")
    print(f"Overall Loss Rate: {wl_analyzer.get_overall_loss_rate():.2%}")
    
    # Analyze key drivers
    driver_analyzer = KeyDriverAnalyzer(chargebacks, transactions)
    top_drivers = driver_analyzer.get_top_drivers(n=3)
    
    print("\nTop 3 Key Drivers:")
    for i, driver in enumerate(top_drivers, 1):
        print(f"  {i}. {driver.driver_name} (Impact: {driver.impact_score:.4f})")
    print()


def example_multi_period_forecast():
    """Example 3: Multi-period forecasting with different methods."""
    print("=" * 70)
    print("EXAMPLE 3: Multi-Period Forecasting Comparison")
    print("=" * 70)
    
    # Create trending data (increasing chargebacks)
    chargebacks = []
    for month in range(1, 7):
        num_chargebacks = month + 2  # Increasing trend
        for i in range(num_chargebacks):
            cb = Chargeback(
                transaction_id=f"TXN{month:02d}{i:03d}",
                amount=100.0,
                date=datetime(2023, month, 10),
                reason_code="FRAUD",
                merchant_id="MERCH_A",
                status="won"
            )
            chargebacks.append(cb)
    
    transactions = []
    total_transactions = {f"2023-{m:02d}": 1000 for m in range(1, 7)}
    
    # Compare different forecasting methods
    forecaster = ChargebackForecaster(chargebacks, transactions, total_transactions)
    
    methods = ['simple_average', 'weighted_average', 'trend']
    print("\nForecasting for 2023-07 with different methods:")
    print(f"{'Method':<20} {'Expected CBs':<15} {'Expected Rate':<15}")
    print("-" * 50)
    
    for method in methods:
        forecast = forecaster.forecast_next_period(
            forecast_period="2023-07",
            expected_transactions=1000,
            method=method
        )
        print(f"{method:<20} {forecast.expected_chargebacks:<15.2f} {forecast.expected_chargeback_rate:<15.4%}")
    print()


def example_merchant_specific_analysis():
    """Example 4: Merchant-specific analysis."""
    print("=" * 70)
    print("EXAMPLE 4: Merchant-Specific Analysis")
    print("=" * 70)
    
    # Create data for multiple merchants
    merchants = ["MERCH_A", "MERCH_B", "MERCH_C"]
    chargebacks = []
    
    # MERCH_A has high chargeback rate
    for i in range(10):
        chargebacks.append(Chargeback(
            f"TXN_A{i:03d}", 100.0, datetime(2023, 1, i + 1),
            "FRAUD", "MERCH_A", "lost"
        ))
    
    # MERCH_B has medium chargeback rate
    for i in range(5):
        chargebacks.append(Chargeback(
            f"TXN_B{i:03d}", 100.0, datetime(2023, 1, i + 1),
            "NOT_RECEIVED", "MERCH_B", "won"
        ))
    
    # MERCH_C has low chargeback rate
    for i in range(2):
        chargebacks.append(Chargeback(
            f"TXN_C{i:03d}", 100.0, datetime(2023, 1, i + 1),
            "DEFECTIVE", "MERCH_C", "won"
        ))
    
    total_transactions = {"2023-01": 1000}
    transactions = []
    
    # Analyze per merchant
    rate_calc = HistoricalRateCalculator(chargebacks, total_transactions)
    wl_analyzer = WinLossAnalyzer(chargebacks)
    
    print("\nMerchant Analysis:")
    print(f"{'Merchant':<15} {'Chargebacks':<15} {'Win Rate':<15}")
    print("-" * 45)
    
    for merchant in merchants:
        merchant_rates = rate_calc.calculate_rate_by_merchant(merchant)
        merchant_wl = wl_analyzer.calculate_ratio_by_merchant(merchant)
        
        cb_count = sum(r.total_chargebacks for r in merchant_rates)
        win_rate = merchant_wl[0].win_rate if merchant_wl else 0.0
        
        print(f"{merchant:<15} {cb_count:<15} {win_rate:<15.2%}")
    print()


if __name__ == '__main__':
    example_basic_forecast()
    example_historical_analysis()
    example_multi_period_forecast()
    example_merchant_specific_analysis()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
