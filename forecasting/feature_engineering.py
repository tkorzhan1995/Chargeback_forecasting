"""
Feature engineering module for creating predictive features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from utils import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Create features for chargeback prediction models.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        pass
    
    def create_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create temporal features from date column.
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            
        Returns:
            pd.DataFrame: Dataframe with temporal features
        """
        logger.info("Creating temporal features")
        df_copy = df.copy()
        
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        # Date components
        df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
        df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
        df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
        df_copy[f'{date_column}_dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy[f'{date_column}_dayofyear'] = df_copy[date_column].dt.dayofyear
        df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
        df_copy[f'{date_column}_week'] = df_copy[date_column].dt.isocalendar().week
        
        # Weekend indicator
        df_copy[f'{date_column}_is_weekend'] = df_copy[f'{date_column}_dayofweek'].isin([5, 6]).astype(int)
        
        # Month start/end indicators
        df_copy[f'{date_column}_is_month_start'] = df_copy[date_column].dt.is_month_start.astype(int)
        df_copy[f'{date_column}_is_month_end'] = df_copy[date_column].dt.is_month_end.astype(int)
        
        return df_copy
    
    def create_lag_features(self, df: pd.DataFrame, value_column: str,
                           lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Create lagged features for time series.
        
        Args:
            df: Input dataframe (must be sorted by date)
            value_column: Column to create lags for
            lags: List of lag periods
            
        Returns:
            pd.DataFrame: Dataframe with lag features
        """
        logger.info(f"Creating lag features for {value_column}")
        df_copy = df.copy()
        
        for lag in lags:
            df_copy[f'{value_column}_lag_{lag}'] = df_copy[value_column].shift(lag)
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, value_column: str,
                               windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input dataframe (must be sorted by date)
            value_column: Column to calculate rolling statistics
            windows: List of window sizes
            
        Returns:
            pd.DataFrame: Dataframe with rolling features
        """
        logger.info(f"Creating rolling features for {value_column}")
        df_copy = df.copy()
        
        for window in windows:
            df_copy[f'{value_column}_rolling_mean_{window}'] = df_copy[value_column].rolling(window=window).mean()
            df_copy[f'{value_column}_rolling_std_{window}'] = df_copy[value_column].rolling(window=window).std()
            df_copy[f'{value_column}_rolling_min_{window}'] = df_copy[value_column].rolling(window=window).min()
            df_copy[f'{value_column}_rolling_max_{window}'] = df_copy[value_column].rolling(window=window).max()
        
        return df_copy
    
    def create_chargeback_rate_features(self, transactions: pd.DataFrame,
                                       chargebacks: pd.DataFrame,
                                       group_by_cols: List[str]) -> pd.DataFrame:
        """
        Calculate historical chargeback rates by different dimensions.
        
        Args:
            transactions: Transactions dataframe
            chargebacks: Chargebacks dataframe
            group_by_cols: Columns to group by (e.g., ['product_id'], ['channel_id'])
            
        Returns:
            pd.DataFrame: Chargeback rate features
        """
        logger.info(f"Calculating chargeback rates by {group_by_cols}")
        
        # Count transactions by group
        txn_counts = transactions.groupby(group_by_cols).size().reset_index(name='txn_count')
        
        # Count chargebacks by group
        cb_counts = chargebacks.groupby(group_by_cols).size().reset_index(name='cb_count')
        
        # Merge and calculate rate
        rates = txn_counts.merge(cb_counts, on=group_by_cols, how='left')
        rates['cb_count'].fillna(0, inplace=True)
        rates['chargeback_rate'] = rates['cb_count'] / rates['txn_count']
        
        return rates
    
    def create_customer_features(self, df: pd.DataFrame, customer_id_col: str = 'customer_id') -> pd.DataFrame:
        """
        Create customer-level features.
        
        Args:
            df: Input dataframe with customer transactions
            customer_id_col: Customer identifier column
            
        Returns:
            pd.DataFrame: Customer features
        """
        logger.info("Creating customer features")
        
        customer_features = df.groupby(customer_id_col).agg({
            'transaction_id': 'count',  # Transaction count
            'amount': ['sum', 'mean', 'std', 'min', 'max'],  # Amount statistics
            'transaction_date': ['min', 'max'],  # First and last transaction dates
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            customer_id_col, 'customer_txn_count', 'customer_total_spent',
            'customer_avg_amount', 'customer_std_amount', 'customer_min_amount',
            'customer_max_amount', 'customer_first_txn_date', 'customer_last_txn_date'
        ]
        
        # Calculate customer tenure
        customer_features['customer_tenure_days'] = (
            pd.to_datetime(customer_features['customer_last_txn_date']) -
            pd.to_datetime(customer_features['customer_first_txn_date'])
        ).dt.days
        
        return customer_features
    
    def create_product_features(self, df: pd.DataFrame, product_id_col: str = 'product_id') -> pd.DataFrame:
        """
        Create product-level features.
        
        Args:
            df: Input dataframe with product transactions
            product_id_col: Product identifier column
            
        Returns:
            pd.DataFrame: Product features
        """
        logger.info("Creating product features")
        
        product_features = df.groupby(product_id_col).agg({
            'transaction_id': 'count',  # Sales count
            'amount': ['sum', 'mean', 'std'],  # Amount statistics
        }).reset_index()
        
        product_features.columns = [
            product_id_col, 'product_sales_count', 'product_total_revenue',
            'product_avg_price', 'product_std_price'
        ]
        
        return product_features
    
    def create_channel_features(self, df: pd.DataFrame, channel_id_col: str = 'channel_id') -> pd.DataFrame:
        """
        Create channel-level features.
        
        Args:
            df: Input dataframe with channel transactions
            channel_id_col: Channel identifier column
            
        Returns:
            pd.DataFrame: Channel features
        """
        logger.info("Creating channel features")
        
        channel_features = df.groupby(channel_id_col).agg({
            'transaction_id': 'count',  # Transaction count by channel
            'amount': ['sum', 'mean'],  # Amount statistics
        }).reset_index()
        
        channel_features.columns = [
            channel_id_col, 'channel_txn_count', 'channel_total_revenue',
            'channel_avg_amount'
        ]
        
        return channel_features
    
    def create_win_loss_features(self, chargebacks: pd.DataFrame) -> pd.DataFrame:
        """
        Create win/loss ratio features from historical disputes.
        
        Args:
            chargebacks: Chargebacks dataframe with status/outcome
            
        Returns:
            pd.DataFrame: Win/loss ratio features
        """
        logger.info("Creating win/loss ratio features")
        
        if 'status' not in chargebacks.columns:
            logger.warning("Status column not found in chargebacks data")
            return pd.DataFrame()
        
        # Categorize outcomes
        win_statuses = ['won', 'accepted', 'representment_won']
        loss_statuses = ['lost', 'rejected', 'expired', 'representment_lost']
        
        chargebacks['outcome'] = 'pending'
        chargebacks.loc[chargebacks['status'].str.lower().isin(win_statuses), 'outcome'] = 'won'
        chargebacks.loc[chargebacks['status'].str.lower().isin(loss_statuses), 'outcome'] = 'lost'
        
        # Calculate win/loss ratio by different dimensions
        features_list = []
        
        # Overall win/loss
        overall = chargebacks['outcome'].value_counts()
        overall_ratio = overall.get('won', 0) / (overall.get('lost', 0) + overall.get('won', 0) + 1e-6)
        features_list.append({'dimension': 'overall', 'win_loss_ratio': overall_ratio})
        
        # By reason code if available
        if 'reason_code' in chargebacks.columns:
            by_reason = chargebacks.groupby('reason_code')['outcome'].apply(
                lambda x: x[x == 'won'].count() / (x[x == 'lost'].count() + x[x == 'won'].count() + 1e-6)
            ).reset_index()
            by_reason.columns = ['reason_code', 'win_loss_ratio']
            features_list.append(by_reason)
        
        return chargebacks
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        logger.info("Creating interaction features")
        df_copy = df.copy()
        
        # Amount-based interactions
        if 'amount' in df_copy.columns and 'customer_avg_amount' in df_copy.columns:
            df_copy['amount_vs_customer_avg'] = df_copy['amount'] / (df_copy['customer_avg_amount'] + 1e-6)
        
        if 'amount' in df_copy.columns and 'product_avg_price' in df_copy.columns:
            df_copy['amount_vs_product_avg'] = df_copy['amount'] / (df_copy['product_avg_price'] + 1e-6)
        
        return df_copy
    
    def build_feature_set(self, transactions: pd.DataFrame,
                         chargebacks: pd.DataFrame = None,
                         include_lag: bool = True,
                         include_rolling: bool = True) -> pd.DataFrame:
        """
        Build comprehensive feature set for modeling.
        
        Args:
            transactions: Transactions dataframe
            chargebacks: Chargebacks dataframe (optional)
            include_lag: Whether to include lag features
            include_rolling: Whether to include rolling features
            
        Returns:
            pd.DataFrame: Complete feature set
        """
        logger.info("Building comprehensive feature set")
        
        df = transactions.copy()
        
        # Temporal features
        if 'transaction_date' in df.columns:
            df = self.create_temporal_features(df, 'transaction_date')
            df = df.sort_values('transaction_date')
        
        # Customer features
        if 'customer_id' in df.columns:
            customer_feats = self.create_customer_features(df)
            df = df.merge(customer_feats, on='customer_id', how='left')
        
        # Product features
        if 'product_id' in df.columns:
            product_feats = self.create_product_features(df)
            df = df.merge(product_feats, on='product_id', how='left')
        
        # Channel features
        if 'channel_id' in df.columns:
            channel_feats = self.create_channel_features(df)
            df = df.merge(channel_feats, on='channel_id', how='left')
        
        # Chargeback rate features
        if chargebacks is not None:
            for group_col in ['product_id', 'channel_id', 'customer_id']:
                if group_col in df.columns and group_col in chargebacks.columns:
                    cb_rates = self.create_chargeback_rate_features(df, chargebacks, [group_col])
                    df = df.merge(cb_rates[[group_col, 'chargeback_rate']], 
                                on=group_col, how='left', 
                                suffixes=('', f'_{group_col}'))
        
        # Lag and rolling features
        if include_lag and 'amount' in df.columns:
            df = self.create_lag_features(df, 'amount')
        
        if include_rolling and 'amount' in df.columns:
            df = self.create_rolling_features(df, 'amount')
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature set created with {len(df.columns)} features")
        return df
