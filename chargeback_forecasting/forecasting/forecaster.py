"""Chargeback forecasting engine."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from chargeback_forecasting.models.chargeback import Chargeback, HistoricalRate, WinLossRatio, KeyDriver
from chargeback_forecasting.analysis.historical_rates import HistoricalRateCalculator
from chargeback_forecasting.analysis.win_loss import WinLossAnalyzer
from chargeback_forecasting.analysis.key_drivers import KeyDriverAnalyzer


@dataclass
class ForecastResult:
    """Result of a chargeback forecast."""
    
    forecast_period: str
    expected_chargebacks: float
    expected_chargeback_rate: float
    confidence_interval_low: float
    confidence_interval_high: float
    expected_win_rate: float
    expected_loss_rate: float
    key_drivers: List[KeyDriver]
    assumptions: List[str]


class ChargebackForecaster:
    """Main forecasting engine for chargebacks."""
    
    def __init__(
        self,
        chargebacks: List[Chargeback],
        transactions: List[Dict[str, Any]],
        total_transactions_by_period: Dict[str, int]
    ):
        """
        Initialize the forecaster.
        
        Args:
            chargebacks: Historical chargeback data
            transactions: All transaction data with attributes
            total_transactions_by_period: Total transactions per period
        """
        self.chargebacks = chargebacks
        self.transactions = transactions
        self.total_transactions_by_period = total_transactions_by_period
        
        # Initialize analyzers
        self.rate_calculator = HistoricalRateCalculator(chargebacks, total_transactions_by_period)
        self.win_loss_analyzer = WinLossAnalyzer(chargebacks)
        self.driver_analyzer = KeyDriverAnalyzer(chargebacks, transactions)
    
    def forecast_next_period(
        self,
        forecast_period: str,
        expected_transactions: int,
        method: str = 'weighted_average'
    ) -> ForecastResult:
        """
        Forecast chargebacks for the next period.
        
        Args:
            forecast_period: Period to forecast (e.g., '2024-01')
            expected_transactions: Expected number of transactions
            method: Forecasting method ('simple_average', 'weighted_average', 'trend')
        
        Returns:
            ForecastResult with forecast details
        """
        # Get historical rates
        historical_rates = self.rate_calculator.calculate_monthly_rates()
        
        if not historical_rates:
            return self._create_default_forecast(forecast_period, expected_transactions)
        
        # Calculate expected chargeback rate based on method
        if method == 'simple_average':
            expected_rate = sum(r.chargeback_rate for r in historical_rates) / len(historical_rates)
        elif method == 'weighted_average':
            # Give more weight to recent periods
            weights = [i + 1 for i in range(len(historical_rates))]
            total_weight = sum(weights)
            expected_rate = sum(
                r.chargeback_rate * w for r, w in zip(historical_rates, weights)
            ) / total_weight
        elif method == 'trend':
            # Simple linear trend
            expected_rate = self._calculate_trend_rate(historical_rates)
        else:
            expected_rate = self.rate_calculator.get_average_rate()
        
        # Calculate confidence intervals (using simple standard deviation approach)
        rates = [r.chargeback_rate for r in historical_rates]
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        std_dev = variance ** 0.5
        
        # 95% confidence interval (approximately 2 standard deviations)
        ci_low = max(0, expected_rate - 2 * std_dev)
        ci_high = expected_rate + 2 * std_dev
        
        # Expected chargebacks
        expected_chargebacks = expected_rate * expected_transactions
        
        # Get win/loss rates
        win_rate = self.win_loss_analyzer.get_overall_win_rate()
        loss_rate = self.win_loss_analyzer.get_overall_loss_rate()
        
        # Get key drivers
        key_drivers = self.driver_analyzer.get_top_drivers(n=5)
        
        # Create assumptions list
        assumptions = [
            f"Based on {len(historical_rates)} historical periods",
            f"Using {method} forecasting method",
            f"Expected transaction volume: {expected_transactions:,}",
            f"Historical average rate: {mean_rate:.4%}",
            f"Win rate based on {len([cb for cb in self.chargebacks if cb.is_resolved()])} resolved disputes"
        ]
        
        return ForecastResult(
            forecast_period=forecast_period,
            expected_chargebacks=expected_chargebacks,
            expected_chargeback_rate=expected_rate,
            confidence_interval_low=ci_low * expected_transactions,
            confidence_interval_high=ci_high * expected_transactions,
            expected_win_rate=win_rate,
            expected_loss_rate=loss_rate,
            key_drivers=key_drivers,
            assumptions=assumptions
        )
    
    def forecast_multiple_periods(
        self,
        num_periods: int,
        expected_transactions_per_period: int,
        start_period: Optional[str] = None
    ) -> List[ForecastResult]:
        """
        Forecast chargebacks for multiple future periods.
        
        Args:
            num_periods: Number of periods to forecast
            expected_transactions_per_period: Expected transactions per period
            start_period: Starting period (defaults to next month)
        
        Returns:
            List of ForecastResult for each period
        """
        if start_period is None:
            # Default to next month
            next_month = datetime.now() + timedelta(days=32)
            start_period = next_month.strftime('%Y-%m')
        
        forecasts = []
        for i in range(num_periods):
            # Parse start period and add i months
            year, month = map(int, start_period.split('-'))
            month += i
            while month > 12:
                month -= 12
                year += 1
            
            period = f"{year:04d}-{month:02d}"
            forecast = self.forecast_next_period(period, expected_transactions_per_period)
            forecasts.append(forecast)
        
        return forecasts
    
    def _calculate_trend_rate(self, historical_rates: List[HistoricalRate]) -> float:
        """Calculate chargeback rate based on linear trend."""
        if len(historical_rates) < 2:
            return historical_rates[0].chargeback_rate if historical_rates else 0.0
        
        # Simple linear regression
        n = len(historical_rates)
        x_values = list(range(n))
        y_values = [r.chargeback_rate for r in historical_rates]
        
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict next period
        next_x = n
        predicted_rate = slope * next_x + intercept
        
        return max(0, predicted_rate)  # Rate can't be negative
    
    def _create_default_forecast(self, forecast_period: str, expected_transactions: int) -> ForecastResult:
        """Create a default forecast when no historical data is available."""
        return ForecastResult(
            forecast_period=forecast_period,
            expected_chargebacks=0.0,
            expected_chargeback_rate=0.0,
            confidence_interval_low=0.0,
            confidence_interval_high=0.0,
            expected_win_rate=0.0,
            expected_loss_rate=0.0,
            key_drivers=[],
            assumptions=["No historical data available"]
        )
