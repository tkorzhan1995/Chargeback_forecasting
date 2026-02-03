"""Win/Loss ratio analysis module."""
from typing import List, Dict
from collections import defaultdict
from chargeback_forecasting.models.chargeback import Chargeback, WinLossRatio


class WinLossAnalyzer:
    """Analyze win/loss ratios from chargeback dispute data."""
    
    def __init__(self, chargebacks: List[Chargeback]):
        """
        Initialize the analyzer.
        
        Args:
            chargebacks: List of chargeback records
        """
        self.chargebacks = chargebacks
    
    def calculate_monthly_ratios(self) -> List[WinLossRatio]:
        """Calculate monthly win/loss ratios."""
        monthly_stats = defaultdict(lambda: {'won': 0, 'lost': 0, 'total': 0})
        
        for cb in self.chargebacks:
            if cb.is_resolved():
                period = cb.date.strftime('%Y-%m')
                monthly_stats[period]['total'] += 1
                if cb.is_won():
                    monthly_stats[period]['won'] += 1
                elif cb.is_lost():
                    monthly_stats[period]['lost'] += 1
        
        ratios = []
        for period, stats in monthly_stats.items():
            if stats['total'] > 0:
                ratios.append(WinLossRatio(
                    period=period,
                    total_disputes=stats['total'],
                    won_disputes=stats['won'],
                    lost_disputes=stats['lost'],
                    win_rate=stats['won'] / stats['total'],
                    loss_rate=stats['lost'] / stats['total']
                ))
        
        return sorted(ratios, key=lambda x: x.period)
    
    def calculate_quarterly_ratios(self) -> List[WinLossRatio]:
        """Calculate quarterly win/loss ratios."""
        quarterly_stats = defaultdict(lambda: {'won': 0, 'lost': 0, 'total': 0})
        
        for cb in self.chargebacks:
            if cb.is_resolved():
                quarter = f"Q{(cb.date.month - 1) // 3 + 1}-{cb.date.year}"
                quarterly_stats[quarter]['total'] += 1
                if cb.is_won():
                    quarterly_stats[quarter]['won'] += 1
                elif cb.is_lost():
                    quarterly_stats[quarter]['lost'] += 1
        
        ratios = []
        for period, stats in quarterly_stats.items():
            if stats['total'] > 0:
                ratios.append(WinLossRatio(
                    period=period,
                    total_disputes=stats['total'],
                    won_disputes=stats['won'],
                    lost_disputes=stats['lost'],
                    win_rate=stats['won'] / stats['total'],
                    loss_rate=stats['lost'] / stats['total']
                ))
        
        return sorted(ratios, key=lambda x: x.period)
    
    def calculate_ratio_by_merchant(self, merchant_id: str) -> List[WinLossRatio]:
        """Calculate win/loss ratios for a specific merchant."""
        merchant_chargebacks = [cb for cb in self.chargebacks if cb.merchant_id == merchant_id]
        monthly_stats = defaultdict(lambda: {'won': 0, 'lost': 0, 'total': 0})
        
        for cb in merchant_chargebacks:
            if cb.is_resolved():
                period = cb.date.strftime('%Y-%m')
                monthly_stats[period]['total'] += 1
                if cb.is_won():
                    monthly_stats[period]['won'] += 1
                elif cb.is_lost():
                    monthly_stats[period]['lost'] += 1
        
        ratios = []
        for period, stats in monthly_stats.items():
            if stats['total'] > 0:
                ratios.append(WinLossRatio(
                    period=period,
                    total_disputes=stats['total'],
                    won_disputes=stats['won'],
                    lost_disputes=stats['lost'],
                    win_rate=stats['won'] / stats['total'],
                    loss_rate=stats['lost'] / stats['total'],
                    merchant_id=merchant_id
                ))
        
        return sorted(ratios, key=lambda x: x.period)
    
    def get_overall_win_rate(self) -> float:
        """Calculate overall win rate across all disputes."""
        resolved = [cb for cb in self.chargebacks if cb.is_resolved()]
        if not resolved:
            return 0.0
        
        won = sum(1 for cb in resolved if cb.is_won())
        return won / len(resolved)
    
    def get_overall_loss_rate(self) -> float:
        """Calculate overall loss rate across all disputes."""
        resolved = [cb for cb in self.chargebacks if cb.is_resolved()]
        if not resolved:
            return 0.0
        
        lost = sum(1 for cb in resolved if cb.is_lost())
        return lost / len(resolved)
