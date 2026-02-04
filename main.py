#!/usr/bin/env python3
"""
Main application for Chargeback Forecasting.

This script demonstrates how to use the chargeback forecasting system
to analyze historical data and generate forecasts.
"""
import argparse
import sys
from pathlib import Path

from chargeback_forecasting.utils.data_loader import (
    load_chargebacks_from_json,
    load_transactions_from_json,
    calculate_transaction_counts,
    export_forecast_to_csv,
    export_forecast_to_json
)
from chargeback_forecasting.forecasting.forecaster import ChargebackForecaster
from chargeback_forecasting.analysis.historical_rates import HistoricalRateCalculator
from chargeback_forecasting.analysis.win_loss import WinLossAnalyzer
from chargeback_forecasting.analysis.key_drivers import KeyDriverAnalyzer


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("CHARGEBACK FORECASTING SYSTEM")
    print("=" * 70)
    print()


def print_historical_analysis(chargebacks, transactions, total_transactions):
    """Print historical analysis results."""
    print("\n" + "=" * 70)
    print("HISTORICAL ANALYSIS")
    print("=" * 70)
    
    # Historical rates
    rate_calc = HistoricalRateCalculator(chargebacks, total_transactions)
    monthly_rates = rate_calc.calculate_monthly_rates()
    
    print("\nMonthly Chargeback Rates:")
    print(f"{'Period':<15} {'Transactions':<15} {'Chargebacks':<15} {'Rate':<10}")
    print("-" * 70)
    for rate in monthly_rates[-6:]:  # Show last 6 months
        print(f"{rate.period:<15} {rate.total_transactions:<15,} {rate.total_chargebacks:<15} {rate.chargeback_rate:<10.2%}")
    
    avg_rate = rate_calc.get_average_rate()
    print(f"\nAverage Chargeback Rate: {avg_rate:.4%}")
    
    # Win/Loss analysis
    wl_analyzer = WinLossAnalyzer(chargebacks)
    monthly_ratios = wl_analyzer.calculate_monthly_ratios()
    
    print("\n\nWin/Loss Ratios:")
    print(f"{'Period':<15} {'Total':<10} {'Won':<10} {'Lost':<10} {'Win Rate':<10}")
    print("-" * 70)
    for ratio in monthly_ratios[-6:]:  # Show last 6 months
        print(f"{ratio.period:<15} {ratio.total_disputes:<10} {ratio.won_disputes:<10} {ratio.lost_disputes:<10} {ratio.win_rate:<10.2%}")
    
    overall_win = wl_analyzer.get_overall_win_rate()
    overall_loss = wl_analyzer.get_overall_loss_rate()
    print(f"\nOverall Win Rate: {overall_win:.2%}")
    print(f"Overall Loss Rate: {overall_loss:.2%}")
    
    # Key drivers
    driver_analyzer = KeyDriverAnalyzer(chargebacks, transactions)
    top_drivers = driver_analyzer.get_top_drivers(n=5)
    
    print("\n\nTop 5 Key Drivers:")
    print(f"{'Driver':<30} {'Type':<15} {'Impact':<15} {'Correlation':<15}")
    print("-" * 70)
    for driver in top_drivers:
        print(f"{driver.driver_name:<30} {driver.driver_type:<15} {driver.impact_score:<15.4f} {driver.correlation:<15.4f}")


def print_forecast(forecast_result):
    """Print a single forecast result."""
    print(f"\n{'Period:':<30} {forecast_result.forecast_period}")
    print(f"{'Expected Chargebacks:':<30} {forecast_result.expected_chargebacks:.2f}")
    print(f"{'Expected Rate:':<30} {forecast_result.expected_chargeback_rate:.4%}")
    print(f"{'95% Confidence Interval:':<30} {forecast_result.confidence_interval_low:.2f} - {forecast_result.confidence_interval_high:.2f}")
    print(f"{'Expected Win Rate:':<30} {forecast_result.expected_win_rate:.2%}")
    print(f"{'Expected Loss Rate:':<30} {forecast_result.expected_loss_rate:.2%}")
    
    if forecast_result.key_drivers:
        print("\nTop Key Drivers:")
        for i, driver in enumerate(forecast_result.key_drivers[:3], 1):
            print(f"  {i}. {driver.driver_name} (Impact: {driver.impact_score:.4f})")
    
    print("\nAssumptions:")
    for assumption in forecast_result.assumptions:
        print(f"  - {assumption}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Chargeback Forecasting System'
    )
    parser.add_argument(
        '--chargebacks',
        default='data/sample/chargebacks.json',
        help='Path to chargebacks data file (JSON)'
    )
    parser.add_argument(
        '--transactions',
        default='data/sample/transactions.json',
        help='Path to transactions data file (JSON)'
    )
    parser.add_argument(
        '--forecast-periods',
        type=int,
        default=3,
        help='Number of periods to forecast (default: 3)'
    )
    parser.add_argument(
        '--expected-transactions',
        type=int,
        default=600,
        help='Expected transactions per period (default: 600)'
    )
    parser.add_argument(
        '--method',
        choices=['simple_average', 'weighted_average', 'trend'],
        default='weighted_average',
        help='Forecasting method (default: weighted_average)'
    )
    parser.add_argument(
        '--output-csv',
        help='Export forecast to CSV file'
    )
    parser.add_argument(
        '--output-json',
        help='Export forecast to JSON file'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Load data
    print("Loading data...")
    try:
        chargebacks = load_chargebacks_from_json(args.chargebacks)
        transactions = load_transactions_from_json(args.transactions)
        total_transactions = calculate_transaction_counts(transactions)
        print(f"✓ Loaded {len(chargebacks)} chargebacks")
        print(f"✓ Loaded {len(transactions)} transactions")
    except FileNotFoundError as e:
        print(f"Error: Could not find data file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Print historical analysis
    print_historical_analysis(chargebacks, transactions, total_transactions)
    
    # Create forecaster
    forecaster = ChargebackForecaster(chargebacks, transactions, total_transactions)
    
    # Generate forecasts
    print("\n" + "=" * 70)
    print("FORECAST RESULTS")
    print("=" * 70)
    
    forecasts = forecaster.forecast_multiple_periods(
        num_periods=args.forecast_periods,
        expected_transactions_per_period=args.expected_transactions
    )
    
    for i, forecast in enumerate(forecasts, 1):
        print(f"\n--- Forecast #{i} ---")
        print_forecast(forecast)
    
    # Export if requested
    if args.output_csv:
        export_forecast_to_csv(forecasts, args.output_csv)
        print(f"\n✓ Forecast exported to {args.output_csv}")
    
    if args.output_json:
        export_forecast_to_json(forecasts, args.output_json)
        print(f"✓ Forecast exported to {args.output_json}")
    
    print("\n" + "=" * 70)
    print("FORECAST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
