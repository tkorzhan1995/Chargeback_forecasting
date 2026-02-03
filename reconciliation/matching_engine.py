"""
Matching engine for reconciling chargebacks with transactions.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, Dict, List
from utils import get_logger
from config.settings import RECONCILIATION_CONFIG

logger = get_logger(__name__)


class MatchingEngine:
    """
    Engine for matching chargebacks to original transactions.
    """
    
    def __init__(self, matching_threshold: float = None):
        """
        Initialize MatchingEngine.
        
        Args:
            matching_threshold: Confidence threshold for automatic matching (0-1)
        """
        self.matching_threshold = matching_threshold or RECONCILIATION_CONFIG['matching_threshold']
        self.time_window_hours = RECONCILIATION_CONFIG['time_window_hours']
        self.amount_tolerance = RECONCILIATION_CONFIG['amount_tolerance']
    
    def exact_match(self, chargebacks: pd.DataFrame, 
                   transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Perform exact matching based on transaction ID.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            
        Returns:
            pd.DataFrame: Matched records with confidence score = 1.0
        """
        logger.info("Performing exact match on transaction IDs")
        
        # Match on transaction_id if available
        if 'transaction_id' in chargebacks.columns and 'transaction_id' in transactions.columns:
            matched = chargebacks.merge(
                transactions,
                on='transaction_id',
                how='inner',
                suffixes=('_chargeback', '_transaction')
            )
            matched['match_confidence'] = 1.0
            matched['match_method'] = 'exact'
            
            logger.info(f"Exact match found {len(matched)} records")
            return matched
        
        return pd.DataFrame()
    
    def amount_and_time_match(self, chargebacks: pd.DataFrame,
                             transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Match based on amount and time proximity.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            
        Returns:
            pd.DataFrame: Matched records with confidence scores
        """
        logger.info("Performing amount and time-based matching")
        matches = []
        
        for idx, cb in chargebacks.iterrows():
            # Filter transactions by amount (with tolerance)
            amount_min = cb['amount'] * (1 - self.amount_tolerance)
            amount_max = cb['amount'] * (1 + self.amount_tolerance)
            
            amount_matches = transactions[
                (transactions['amount'] >= amount_min) &
                (transactions['amount'] <= amount_max)
            ]
            
            if len(amount_matches) == 0:
                continue
            
            # Filter by time window if date columns exist
            if 'chargeback_date' in cb and 'transaction_date' in amount_matches.columns:
                cb_date = pd.to_datetime(cb['chargeback_date'])
                time_window_start = cb_date - timedelta(hours=self.time_window_hours)
                time_window_end = cb_date
                
                time_matches = amount_matches[
                    (pd.to_datetime(amount_matches['transaction_date']) >= time_window_start) &
                    (pd.to_datetime(amount_matches['transaction_date']) <= time_window_end)
                ]
            else:
                time_matches = amount_matches
            
            if len(time_matches) > 0:
                # Calculate confidence score based on amount proximity
                for _, txn in time_matches.iterrows():
                    amount_diff = abs(cb['amount'] - txn['amount'])
                    amount_confidence = 1 - (amount_diff / cb['amount'])
                    
                    # Calculate time proximity confidence if dates available
                    if 'chargeback_date' in cb and 'transaction_date' in txn:
                        time_diff = abs((pd.to_datetime(cb['chargeback_date']) - 
                                       pd.to_datetime(txn['transaction_date'])).total_seconds())
                        max_time_diff = self.time_window_hours * 3600
                        time_confidence = 1 - (time_diff / max_time_diff)
                    else:
                        time_confidence = 0.5
                    
                    # Combined confidence
                    confidence = (amount_confidence * 0.6 + time_confidence * 0.4)
                    
                    if confidence >= self.matching_threshold:
                        match_record = {
                            'chargeback_id': cb.get('chargeback_id'),
                            'transaction_id': txn.get('transaction_id'),
                            'match_confidence': confidence,
                            'match_method': 'amount_time',
                            'amount_chargeback': cb['amount'],
                            'amount_transaction': txn['amount'],
                        }
                        matches.append(match_record)
        
        if matches:
            matched_df = pd.DataFrame(matches)
            logger.info(f"Amount/time match found {len(matched_df)} records")
            return matched_df
        
        return pd.DataFrame()
    
    def customer_based_match(self, chargebacks: pd.DataFrame,
                            transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Match based on customer ID and amount.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            
        Returns:
            pd.DataFrame: Matched records with confidence scores
        """
        logger.info("Performing customer-based matching")
        
        if 'customer_id' not in chargebacks.columns or 'customer_id' not in transactions.columns:
            logger.warning("Customer ID not available for matching")
            return pd.DataFrame()
        
        matches = []
        
        for idx, cb in chargebacks.iterrows():
            # Match on customer_id and amount
            customer_matches = transactions[
                (transactions['customer_id'] == cb['customer_id'])
            ]
            
            if len(customer_matches) == 0:
                continue
            
            # Filter by amount
            amount_min = cb['amount'] * (1 - self.amount_tolerance)
            amount_max = cb['amount'] * (1 + self.amount_tolerance)
            
            amount_matches = customer_matches[
                (customer_matches['amount'] >= amount_min) &
                (customer_matches['amount'] <= amount_max)
            ]
            
            for _, txn in amount_matches.iterrows():
                amount_diff = abs(cb['amount'] - txn['amount'])
                confidence = 0.8 - (amount_diff / cb['amount']) * 0.2  # Base 0.8 for customer match
                
                if confidence >= self.matching_threshold:
                    match_record = {
                        'chargeback_id': cb.get('chargeback_id'),
                        'transaction_id': txn.get('transaction_id'),
                        'match_confidence': confidence,
                        'match_method': 'customer_amount',
                        'customer_id': cb['customer_id'],
                    }
                    matches.append(match_record)
        
        if matches:
            matched_df = pd.DataFrame(matches)
            logger.info(f"Customer-based match found {len(matched_df)} records")
            return matched_df
        
        return pd.DataFrame()
    
    def reconcile(self, chargebacks: pd.DataFrame, 
                 transactions: pd.DataFrame,
                 products: pd.DataFrame = None,
                 customers: pd.DataFrame = None,
                 channels: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main reconciliation function using multiple matching strategies.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            products: Products dataframe (optional)
            customers: Customers dataframe (optional)
            channels: Channels dataframe (optional)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (matched_records, unmatched_chargebacks)
        """
        logger.info("Starting reconciliation process")
        
        all_matches = []
        matched_cb_ids = set()
        
        # Strategy 1: Exact match
        exact_matches = self.exact_match(chargebacks, transactions)
        if len(exact_matches) > 0:
            all_matches.append(exact_matches)
            matched_cb_ids.update(exact_matches['chargeback_id'].unique())
        
        # Get unmatched chargebacks for next strategy
        unmatched_cb = chargebacks[~chargebacks['chargeback_id'].isin(matched_cb_ids)]
        
        # Strategy 2: Amount and time match
        if len(unmatched_cb) > 0:
            amount_time_matches = self.amount_and_time_match(unmatched_cb, transactions)
            if len(amount_time_matches) > 0:
                # Merge back with full chargeback data
                amount_time_matches = amount_time_matches.merge(
                    chargebacks, on='chargeback_id', how='left', suffixes=('', '_cb')
                ).merge(
                    transactions, on='transaction_id', how='left', suffixes=('', '_txn')
                )
                all_matches.append(amount_time_matches)
                matched_cb_ids.update(amount_time_matches['chargeback_id'].unique())
        
        # Get remaining unmatched for next strategy
        unmatched_cb = chargebacks[~chargebacks['chargeback_id'].isin(matched_cb_ids)]
        
        # Strategy 3: Customer-based match
        if len(unmatched_cb) > 0:
            customer_matches = self.customer_based_match(unmatched_cb, transactions)
            if len(customer_matches) > 0:
                customer_matches = customer_matches.merge(
                    chargebacks, on='chargeback_id', how='left', suffixes=('', '_cb')
                ).merge(
                    transactions, on='transaction_id', how='left', suffixes=('', '_txn')
                )
                all_matches.append(customer_matches)
                matched_cb_ids.update(customer_matches['chargeback_id'].unique())
        
        # Combine all matches
        if all_matches:
            matched = pd.concat(all_matches, ignore_index=True)
            
            # Deduplicate - keep highest confidence match for each chargeback
            matched = matched.sort_values('match_confidence', ascending=False)
            matched = matched.drop_duplicates(subset=['chargeback_id'], keep='first')
            
            # Enrich with product, customer, channel data
            if products is not None and 'product_id' in matched.columns:
                matched = matched.merge(products, on='product_id', how='left', suffixes=('', '_product'))
            
            if customers is not None and 'customer_id' in matched.columns:
                matched = matched.merge(customers, on='customer_id', how='left', suffixes=('', '_customer'))
            
            if channels is not None and 'channel_id' in matched.columns:
                matched = matched.merge(channels, on='channel_id', how='left', suffixes=('', '_channel'))
        else:
            matched = pd.DataFrame()
        
        # Get final unmatched chargebacks
        unmatched = chargebacks[~chargebacks['chargeback_id'].isin(matched_cb_ids)]
        
        match_rate = len(matched) / len(chargebacks) * 100 if len(chargebacks) > 0 else 0
        logger.info(f"Reconciliation complete. Match rate: {match_rate:.2f}% ({len(matched)}/{len(chargebacks)})")
        
        return matched, unmatched
