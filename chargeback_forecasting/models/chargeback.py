"""Chargeback data models."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Chargeback:
    """Represents a chargeback transaction."""
    
    transaction_id: str
    amount: float
    date: datetime
    reason_code: str
    merchant_id: str
    status: str  # 'pending', 'won', 'lost'
    resolved_date: Optional[datetime] = None
    resolution_amount: Optional[float] = None
    
    def is_resolved(self) -> bool:
        """Check if chargeback is resolved."""
        return self.status in ['won', 'lost']
    
    def is_won(self) -> bool:
        """Check if chargeback was won."""
        return self.status == 'won'
    
    def is_lost(self) -> bool:
        """Check if chargeback was lost."""
        return self.status == 'lost'


@dataclass
class HistoricalRate:
    """Historical chargeback rate for a time period."""
    
    period: str  # e.g., '2023-01', 'Q1-2023'
    total_transactions: int
    total_chargebacks: int
    chargeback_rate: float
    merchant_id: Optional[str] = None
    
    def __post_init__(self):
        """Calculate chargeback rate if not provided."""
        if self.chargeback_rate == 0 and self.total_transactions > 0:
            self.chargeback_rate = self.total_chargebacks / self.total_transactions


@dataclass
class WinLossRatio:
    """Win/Loss ratio statistics for a time period."""
    
    period: str
    total_disputes: int
    won_disputes: int
    lost_disputes: int
    win_rate: float
    loss_rate: float
    merchant_id: Optional[str] = None
    
    def __post_init__(self):
        """Calculate win/loss rates if not provided."""
        if self.total_disputes > 0 and (self.win_rate == 0.0 and self.loss_rate == 0.0):
            self.win_rate = self.won_disputes / self.total_disputes
            self.loss_rate = self.lost_disputes / self.total_disputes


@dataclass
class KeyDriver:
    """Key driver affecting chargeback rates."""
    
    driver_name: str
    driver_type: str  # 'categorical' or 'numerical'
    impact_score: float  # -1 to 1, negative means reduces chargebacks
    correlation: float  # correlation coefficient
    description: Optional[str] = None
