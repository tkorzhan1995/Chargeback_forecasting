"""Key drivers analysis module."""
from typing import List, Dict, Any
import statistics
from chargeback_forecasting.models.chargeback import Chargeback, KeyDriver


class KeyDriverAnalyzer:
    """Analyze key drivers that affect chargeback rates."""
    
    def __init__(self, chargebacks: List[Chargeback], transactions: List[Dict[str, Any]]):
        """
        Initialize the analyzer.
        
        Args:
            chargebacks: List of chargeback records
            transactions: List of all transactions with additional attributes
        """
        self.chargebacks = chargebacks
        self.transactions = transactions
    
    def analyze_reason_codes(self) -> List[KeyDriver]:
        """Analyze chargeback reason codes as key drivers."""
        reason_counts = {}
        for cb in self.chargebacks:
            reason_counts[cb.reason_code] = reason_counts.get(cb.reason_code, 0) + 1
        
        total_chargebacks = len(self.chargebacks)
        drivers = []
        
        for reason, count in reason_counts.items():
            impact = count / total_chargebacks if total_chargebacks > 0 else 0
            drivers.append(KeyDriver(
                driver_name=f"Reason: {reason}",
                driver_type='categorical',
                impact_score=impact,
                correlation=impact,
                description=f"Chargeback reason code {reason}"
            ))
        
        return sorted(drivers, key=lambda x: abs(x.impact_score), reverse=True)
    
    def analyze_merchant_patterns(self) -> List[KeyDriver]:
        """Analyze merchant-specific patterns as drivers."""
        merchant_chargebacks = {}
        merchant_total = {}
        
        for cb in self.chargebacks:
            merchant_chargebacks[cb.merchant_id] = merchant_chargebacks.get(cb.merchant_id, 0) + 1
        
        for txn in self.transactions:
            merchant_id = txn.get('merchant_id', 'unknown')
            merchant_total[merchant_id] = merchant_total.get(merchant_id, 0) + 1
        
        drivers = []
        for merchant_id in merchant_chargebacks:
            cb_count = merchant_chargebacks.get(merchant_id, 0)
            total = merchant_total.get(merchant_id, 0)
            
            if total > 0:
                rate = cb_count / total
                # Higher rate means more impact
                impact = rate * 2 - 1  # Scale to -1 to 1
                drivers.append(KeyDriver(
                    driver_name=f"Merchant: {merchant_id}",
                    driver_type='categorical',
                    impact_score=min(max(impact, -1), 1),
                    correlation=rate,
                    description=f"Merchant {merchant_id} chargeback rate"
                ))
        
        return sorted(drivers, key=lambda x: abs(x.correlation), reverse=True)
    
    def analyze_amount_patterns(self) -> KeyDriver:
        """Analyze transaction amount as a driver."""
        chargeback_amounts = [cb.amount for cb in self.chargebacks]
        all_amounts = [txn.get('amount', 0) for txn in self.transactions]
        
        if not chargeback_amounts or not all_amounts:
            return KeyDriver(
                driver_name="Transaction Amount",
                driver_type='numerical',
                impact_score=0.0,
                correlation=0.0,
                description="Transaction amount correlation with chargebacks"
            )
        
        avg_cb_amount = statistics.mean(chargeback_amounts)
        avg_all_amount = statistics.mean(all_amounts)
        
        # Simple correlation measure: higher amounts tend to have chargebacks
        if avg_all_amount > 0:
            correlation = (avg_cb_amount - avg_all_amount) / avg_all_amount
        else:
            correlation = 0.0
        
        return KeyDriver(
            driver_name="Transaction Amount",
            driver_type='numerical',
            impact_score=min(max(correlation, -1), 1),
            correlation=correlation,
            description="Transaction amount correlation with chargebacks"
        )
    
    def analyze_temporal_patterns(self) -> List[KeyDriver]:
        """Analyze temporal patterns (seasonality, day of week, etc.)."""
        day_counts = {i: 0 for i in range(7)}  # 0=Monday, 6=Sunday
        month_counts = {i: 0 for i in range(1, 13)}
        
        for cb in self.chargebacks:
            day_counts[cb.date.weekday()] += 1
            month_counts[cb.date.month] += 1
        
        total = len(self.chargebacks)
        drivers = []
        
        # Day of week drivers
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day, count in day_counts.items():
            if total > 0:
                rate = count / total
                impact = (rate - 1/7) * 7  # Deviation from expected uniform distribution
                drivers.append(KeyDriver(
                    driver_name=f"Day: {day_names[day]}",
                    driver_type='categorical',
                    impact_score=min(max(impact, -1), 1),
                    correlation=rate,
                    description=f"Chargebacks on {day_names[day]}"
                ))
        
        # Month drivers
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in month_counts.items():
            if total > 0:
                rate = count / total
                impact = (rate - 1/12) * 12  # Deviation from expected uniform distribution
                drivers.append(KeyDriver(
                    driver_name=f"Month: {month_names[month-1]}",
                    driver_type='categorical',
                    impact_score=min(max(impact, -1), 1),
                    correlation=rate,
                    description=f"Chargebacks in {month_names[month-1]}"
                ))
        
        return sorted(drivers, key=lambda x: abs(x.impact_score), reverse=True)[:5]
    
    def get_top_drivers(self, n: int = 10) -> List[KeyDriver]:
        """Get top N key drivers across all analyses."""
        all_drivers = []
        all_drivers.extend(self.analyze_reason_codes())
        all_drivers.extend(self.analyze_merchant_patterns())
        all_drivers.append(self.analyze_amount_patterns())
        all_drivers.extend(self.analyze_temporal_patterns())
        
        return sorted(all_drivers, key=lambda x: abs(x.impact_score), reverse=True)[:n]
