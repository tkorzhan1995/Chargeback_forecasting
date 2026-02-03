"""Historical rate analysis module."""
from typing import List, Dict
from datetime import datetime
from collections import defaultdict
from chargeback_forecasting.models.chargeback import Chargeback, HistoricalRate


class HistoricalRateCalculator:
    """Calculate historical chargeback rates from transaction data."""
    
    def __init__(self, chargebacks: List[Chargeback], total_transactions: Dict[str, int]):
        """
        Initialize the calculator.
        
        Args:
            chargebacks: List of chargeback records
            total_transactions: Dictionary mapping period to total transaction count
        """
        self.chargebacks = chargebacks
        self.total_transactions = total_transactions
    
    def calculate_monthly_rates(self) -> List[HistoricalRate]:
        """Calculate monthly chargeback rates."""
        monthly_chargebacks = defaultdict(int)
        
        for cb in self.chargebacks:
            period = cb.date.strftime('%Y-%m')
            monthly_chargebacks[period] += 1
        
        rates = []
        for period, cb_count in monthly_chargebacks.items():
            total = self.total_transactions.get(period, 0)
            if total > 0:
                rate = cb_count / total
                rates.append(HistoricalRate(
                    period=period,
                    total_transactions=total,
                    total_chargebacks=cb_count,
                    chargeback_rate=rate
                ))
        
        return sorted(rates, key=lambda x: x.period)
    
    def calculate_quarterly_rates(self) -> List[HistoricalRate]:
        """Calculate quarterly chargeback rates."""
        quarterly_chargebacks = defaultdict(int)
        quarterly_transactions = defaultdict(int)
        
        for cb in self.chargebacks:
            quarter = f"Q{(cb.date.month - 1) // 3 + 1}-{cb.date.year}"
            quarterly_chargebacks[quarter] += 1
        
        # Aggregate monthly transactions to quarterly
        for period, count in self.total_transactions.items():
            if '-' in period:  # Monthly format YYYY-MM
                year, month = period.split('-')
                quarter = f"Q{(int(month) - 1) // 3 + 1}-{year}"
                quarterly_transactions[quarter] += count
        
        rates = []
        for period, cb_count in quarterly_chargebacks.items():
            total = quarterly_transactions.get(period, 0)
            if total > 0:
                rate = cb_count / total
                rates.append(HistoricalRate(
                    period=period,
                    total_transactions=total,
                    total_chargebacks=cb_count,
                    chargeback_rate=rate
                ))
        
        return sorted(rates, key=lambda x: x.period)
    
    def calculate_rate_by_merchant(self, merchant_id: str) -> List[HistoricalRate]:
        """Calculate historical rates for a specific merchant."""
        merchant_chargebacks = [cb for cb in self.chargebacks if cb.merchant_id == merchant_id]
        monthly_cb = defaultdict(int)
        
        for cb in merchant_chargebacks:
            period = cb.date.strftime('%Y-%m')
            monthly_cb[period] += 1
        
        rates = []
        for period, cb_count in monthly_cb.items():
            total = self.total_transactions.get(period, 0)
            if total > 0:
                rate = cb_count / total
                rates.append(HistoricalRate(
                    period=period,
                    total_transactions=total,
                    total_chargebacks=cb_count,
                    chargeback_rate=rate,
                    merchant_id=merchant_id
                ))
        
        return sorted(rates, key=lambda x: x.period)
    
    def get_average_rate(self) -> float:
        """Calculate overall average chargeback rate."""
        total_chargebacks = len(self.chargebacks)
        total_transactions = sum(self.total_transactions.values())
        
        if total_transactions == 0:
            return 0.0
        
        return total_chargebacks / total_transactions
