"""
Aggregations module for preparing Power BI data.
"""
import pandas as pd
from typing import Dict, List
from utils import get_logger, aggregate_by_period

logger = get_logger(__name__)


class DataAggregator:
    """
    Aggregate data for Power BI visualizations.
    """
    
    def __init__(self):
        """Initialize DataAggregator."""
        pass
    
    def aggregate_chargebacks_by_period(self, df: pd.DataFrame,
                                       date_col: str = 'chargeback_date',
                                       period: str = 'M') -> pd.DataFrame:
        """
        Aggregate chargebacks by time period.
        
        Args:
            df: Chargeback dataframe
            date_col: Date column name
            period: Period ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            pd.DataFrame: Aggregated data
        """
        logger.info(f"Aggregating chargebacks by {period}")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        agg_df = df.groupby(pd.Grouper(key=date_col, freq=period)).agg({
            'chargeback_id': 'count',
            'amount': ['sum', 'mean', 'median'],
        }).reset_index()
        
        agg_df.columns = [date_col, 'chargeback_count', 'total_amount', 
                         'avg_amount', 'median_amount']
        
        return agg_df
    
    def aggregate_by_dimension(self, df: pd.DataFrame,
                              dimension: str,
                              metrics: List[str] = None) -> pd.DataFrame:
        """
        Aggregate by a specific dimension.
        
        Args:
            df: Input dataframe
            dimension: Dimension column name
            metrics: List of metric columns to aggregate
            
        Returns:
            pd.DataFrame: Aggregated data
        """
        logger.info(f"Aggregating by {dimension}")
        
        if metrics is None:
            metrics = ['amount']
        
        agg_dict = {}
        for metric in metrics:
            if metric in df.columns:
                agg_dict[metric] = ['count', 'sum', 'mean']
        
        if 'chargeback_id' in df.columns:
            agg_dict['chargeback_id'] = 'count'
        elif 'transaction_id' in df.columns:
            agg_dict['transaction_id'] = 'count'
        
        agg_df = df.groupby(dimension).agg(agg_dict).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in agg_df.columns.values]
        
        return agg_df
    
    def calculate_chargeback_rates(self, transactions: pd.DataFrame,
                                  chargebacks: pd.DataFrame,
                                  group_by: List[str]) -> pd.DataFrame:
        """
        Calculate chargeback rates by dimensions.
        
        Args:
            transactions: Transactions dataframe
            chargebacks: Chargebacks dataframe
            group_by: List of columns to group by
            
        Returns:
            pd.DataFrame: Chargeback rates
        """
        logger.info(f"Calculating chargeback rates by {group_by}")
        
        txn_counts = transactions.groupby(group_by).agg({
            'transaction_id': 'count',
            'amount': 'sum'
        }).reset_index()
        txn_counts.columns = group_by + ['txn_count', 'txn_amount']
        
        cb_counts = chargebacks.groupby(group_by).agg({
            'chargeback_id': 'count',
            'amount': 'sum'
        }).reset_index()
        cb_counts.columns = group_by + ['cb_count', 'cb_amount']
        
        rates = txn_counts.merge(cb_counts, on=group_by, how='left')
        rates['cb_count'].fillna(0, inplace=True)
        rates['cb_amount'].fillna(0, inplace=True)
        
        rates['chargeback_rate_count'] = rates['cb_count'] / rates['txn_count'] * 100
        rates['chargeback_rate_amount'] = rates['cb_amount'] / rates['txn_amount'] * 100
        
        return rates
    
    def calculate_win_loss_ratios(self, chargebacks: pd.DataFrame,
                                  group_by: List[str] = None) -> pd.DataFrame:
        """
        Calculate win/loss ratios.
        
        Args:
            chargebacks: Chargebacks dataframe with status
            group_by: List of columns to group by (optional)
            
        Returns:
            pd.DataFrame: Win/loss ratios
        """
        logger.info("Calculating win/loss ratios")
        
        if 'status' not in chargebacks.columns:
            logger.warning("Status column not found")
            return pd.DataFrame()
        
        # Categorize outcomes
        win_statuses = ['won', 'accepted', 'representment_won']
        loss_statuses = ['lost', 'rejected', 'expired', 'representment_lost']
        
        chargebacks['outcome'] = 'pending'
        chargebacks.loc[chargebacks['status'].str.lower().isin(win_statuses), 'outcome'] = 'won'
        chargebacks.loc[chargebacks['status'].str.lower().isin(loss_statuses), 'outcome'] = 'lost'
        
        if group_by:
            wl_ratios = chargebacks.groupby(group_by + ['outcome']).size().unstack(fill_value=0).reset_index()
        else:
            wl_ratios = chargebacks.groupby('outcome').size().to_frame('count').reset_index()
        
        # Calculate ratio
        if 'won' in wl_ratios.columns and 'lost' in wl_ratios.columns:
            wl_ratios['win_loss_ratio'] = wl_ratios['won'] / (wl_ratios['lost'] + 1)
            wl_ratios['win_rate'] = wl_ratios['won'] / (wl_ratios['won'] + wl_ratios['lost']) * 100
        
        return wl_ratios
    
    def create_trend_data(self, df: pd.DataFrame,
                         date_col: str,
                         value_col: str,
                         period: str = 'M') -> pd.DataFrame:
        """
        Create trend data for time series visualizations.
        
        Args:
            df: Input dataframe
            date_col: Date column name
            value_col: Value column name
            period: Period for aggregation
            
        Returns:
            pd.DataFrame: Trend data
        """
        logger.info("Creating trend data")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        trend = df.groupby(pd.Grouper(key=date_col, freq=period))[value_col].agg([
            'count', 'sum', 'mean'
        ]).reset_index()
        
        # Calculate period-over-period change
        trend['pct_change'] = trend['sum'].pct_change() * 100
        
        return trend
    
    def create_kpi_summary(self, chargebacks: pd.DataFrame,
                          transactions: pd.DataFrame = None,
                          matched: pd.DataFrame = None) -> Dict:
        """
        Create KPI summary for dashboard overview.
        
        Args:
            chargebacks: Chargebacks dataframe
            transactions: Transactions dataframe (optional)
            matched: Matched records (optional)
            
        Returns:
            Dict: KPI metrics
        """
        logger.info("Creating KPI summary")
        
        kpis = {
            'total_chargebacks': len(chargebacks),
            'total_chargeback_amount': chargebacks['amount'].sum() if 'amount' in chargebacks.columns else 0,
            'avg_chargeback_amount': chargebacks['amount'].mean() if 'amount' in chargebacks.columns else 0,
        }
        
        if transactions is not None:
            kpis['total_transactions'] = len(transactions)
            kpis['total_transaction_amount'] = transactions['amount'].sum() if 'amount' in transactions.columns else 0
            kpis['chargeback_rate'] = (len(chargebacks) / len(transactions) * 100) if len(transactions) > 0 else 0
        
        if matched is not None:
            kpis['matched_count'] = len(matched)
            kpis['match_rate'] = (len(matched) / len(chargebacks) * 100) if len(chargebacks) > 0 else 0
        
        return kpis
    
    def prepare_drill_through_data(self, df: pd.DataFrame,
                                   hierarchy: List[str]) -> pd.DataFrame:
        """
        Prepare data for drill-through functionality.
        
        Args:
            df: Input dataframe
            hierarchy: List of columns in hierarchy order
            
        Returns:
            pd.DataFrame: Data with hierarchy levels
        """
        logger.info("Preparing drill-through data")
        
        drill_data = df.copy()
        
        # Add hierarchy level indicators
        for i, level in enumerate(hierarchy):
            drill_data[f'level_{i}'] = drill_data[level]
        
        return drill_data
