"""
Linkage algorithms for advanced matching including fuzzy matching.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from difflib import SequenceMatcher
from utils import get_logger
from config.settings import RECONCILIATION_CONFIG

logger = get_logger(__name__)


class LinkageAlgorithms:
    """
    Advanced linkage algorithms for matching records.
    """
    
    def __init__(self, fuzzy_threshold: int = None):
        """
        Initialize LinkageAlgorithms.
        
        Args:
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold or RECONCILIATION_CONFIG['fuzzy_match_threshold']
    
    def fuzzy_string_match(self, str1: str, str2: str) -> float:
        """
        Calculate fuzzy string similarity using SequenceMatcher.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            float: Similarity score (0-100)
        """
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
        
        similarity = SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()
        return similarity * 100
    
    def levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            int: Edit distance
        """
        if pd.isna(str1) or pd.isna(str2):
            return 999
        
        str1, str2 = str(str1), str(str2)
        
        if len(str1) < len(str2):
            return self.levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = range(len(str2) + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def fuzzy_match_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame,
                              key_column: str, match_columns: List[str]) -> pd.DataFrame:
        """
        Perform fuzzy matching between two dataframes.
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            key_column: Column to use as identifier
            match_columns: Columns to use for matching
            
        Returns:
            pd.DataFrame: Matched records with similarity scores
        """
        logger.info(f"Performing fuzzy match on columns: {match_columns}")
        matches = []
        
        for idx1, row1 in df1.iterrows():
            best_match = None
            best_score = 0
            
            for idx2, row2 in df2.iterrows():
                # Calculate similarity for each match column
                scores = []
                for col in match_columns:
                    if col in row1 and col in row2:
                        score = self.fuzzy_string_match(row1[col], row2[col])
                        scores.append(score)
                
                # Average similarity
                avg_score = np.mean(scores) if scores else 0
                
                if avg_score > best_score and avg_score >= self.fuzzy_threshold:
                    best_score = avg_score
                    best_match = row2
            
            if best_match is not None:
                match_record = {
                    f'{key_column}_1': row1.get(key_column),
                    f'{key_column}_2': best_match.get(key_column),
                    'fuzzy_match_score': best_score,
                    'match_method': 'fuzzy'
                }
                matches.append(match_record)
        
        if matches:
            matched_df = pd.DataFrame(matches)
            logger.info(f"Fuzzy matching found {len(matched_df)} records")
            return matched_df
        
        return pd.DataFrame()
    
    def probabilistic_linkage(self, df1: pd.DataFrame, df2: pd.DataFrame,
                             blocking_keys: List[str],
                             comparison_fields: List[str]) -> pd.DataFrame:
        """
        Perform probabilistic record linkage.
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            blocking_keys: Keys to use for blocking (reduce comparisons)
            comparison_fields: Fields to compare
            
        Returns:
            pd.DataFrame: Linked records with probability scores
        """
        logger.info("Performing probabilistic record linkage")
        matches = []
        
        # Blocking: only compare records with same blocking key values
        for blocking_key in blocking_keys:
            if blocking_key not in df1.columns or blocking_key not in df2.columns:
                continue
            
            for key_value in df1[blocking_key].unique():
                block1 = df1[df1[blocking_key] == key_value]
                block2 = df2[df2[blocking_key] == key_value]
                
                for idx1, row1 in block1.iterrows():
                    for idx2, row2 in block2.iterrows():
                        # Calculate agreement probability for each field
                        agreement_weights = []
                        
                        for field in comparison_fields:
                            if field in row1 and field in row2:
                                if pd.notna(row1[field]) and pd.notna(row2[field]):
                                    if row1[field] == row2[field]:
                                        weight = 1.0
                                    else:
                                        # Partial agreement for strings
                                        if isinstance(row1[field], str):
                                            similarity = self.fuzzy_string_match(row1[field], row2[field])
                                            weight = similarity / 100
                                        else:
                                            weight = 0.0
                                    agreement_weights.append(weight)
                        
                        if agreement_weights:
                            # Probability score (simple average)
                            prob_score = np.mean(agreement_weights)
                            
                            if prob_score >= (self.fuzzy_threshold / 100):
                                match_record = {
                                    'id_1': row1.name,
                                    'id_2': row2.name,
                                    'linkage_probability': prob_score,
                                    'match_method': 'probabilistic',
                                    'blocking_key': blocking_key,
                                }
                                matches.append(match_record)
        
        if matches:
            matched_df = pd.DataFrame(matches)
            # Deduplicate - keep highest probability
            matched_df = matched_df.sort_values('linkage_probability', ascending=False)
            matched_df = matched_df.drop_duplicates(subset=['id_1'], keep='first')
            logger.info(f"Probabilistic linkage found {len(matched_df)} records")
            return matched_df
        
        return pd.DataFrame()
    
    def partial_amount_match(self, chargebacks: pd.DataFrame,
                           transactions: pd.DataFrame,
                           tolerance: float = 0.1) -> pd.DataFrame:
        """
        Match chargebacks that are partial amounts of transactions.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            tolerance: Tolerance for partial matching
            
        Returns:
            pd.DataFrame: Matched partial chargebacks
        """
        logger.info("Performing partial amount matching")
        matches = []
        
        for idx_cb, cb in chargebacks.iterrows():
            cb_amount = cb['amount']
            
            # Look for transactions where chargeback is a fraction
            for idx_txn, txn in transactions.iterrows():
                txn_amount = txn['amount']
                
                if txn_amount == 0:
                    continue
                
                ratio = cb_amount / txn_amount
                
                # Check if it's a common fraction (0.5, 0.25, 0.33, etc.)
                common_fractions = [0.25, 0.33, 0.5, 0.66, 0.75]
                
                for frac in common_fractions:
                    if abs(ratio - frac) <= tolerance:
                        match_record = {
                            'chargeback_id': cb.get('chargeback_id'),
                            'transaction_id': txn.get('transaction_id'),
                            'partial_ratio': ratio,
                            'match_confidence': 0.7,  # Lower confidence for partial matches
                            'match_method': 'partial_amount',
                        }
                        matches.append(match_record)
                        break
        
        if matches:
            matched_df = pd.DataFrame(matches)
            logger.info(f"Partial amount matching found {len(matched_df)} records")
            return matched_df
        
        return pd.DataFrame()
    
    def multiple_disputes_match(self, chargebacks: pd.DataFrame,
                               transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Handle cases where multiple disputes relate to the same transaction.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            
        Returns:
            pd.DataFrame: Grouped disputes by transaction
        """
        logger.info("Identifying multiple disputes per transaction")
        
        if 'transaction_id' not in chargebacks.columns:
            return pd.DataFrame()
        
        # Group chargebacks by transaction_id
        grouped = chargebacks.groupby('transaction_id').agg({
            'chargeback_id': lambda x: list(x),
            'amount': 'sum',
            'chargeback_date': 'min',
        }).reset_index()
        
        grouped['dispute_count'] = grouped['chargeback_id'].apply(len)
        
        # Filter for multiple disputes
        multiple_disputes = grouped[grouped['dispute_count'] > 1]
        
        if len(multiple_disputes) > 0:
            logger.info(f"Found {len(multiple_disputes)} transactions with multiple disputes")
            return multiple_disputes
        
        return pd.DataFrame()
    
    def smart_match(self, chargebacks: pd.DataFrame, transactions: pd.DataFrame,
                   use_fuzzy: bool = True, use_partial: bool = True) -> pd.DataFrame:
        """
        Intelligent matching using multiple algorithms.
        
        Args:
            chargebacks: Chargeback dataframe
            transactions: Transaction dataframe
            use_fuzzy: Whether to use fuzzy matching
            use_partial: Whether to check for partial amounts
            
        Returns:
            pd.DataFrame: Matched records using smart algorithms
        """
        logger.info("Performing smart matching with multiple algorithms")
        all_matches = []
        
        # Fuzzy matching on merchant name or description if available
        if use_fuzzy:
            fuzzy_cols = []
            for col in ['merchant_name', 'description', 'merchant']:
                if col in chargebacks.columns and col in transactions.columns:
                    fuzzy_cols.append(col)
            
            if fuzzy_cols:
                fuzzy_matches = self.fuzzy_match_dataframes(
                    chargebacks, transactions,
                    'chargeback_id', fuzzy_cols
                )
                if len(fuzzy_matches) > 0:
                    all_matches.append(fuzzy_matches)
        
        # Partial amount matching
        if use_partial:
            partial_matches = self.partial_amount_match(chargebacks, transactions)
            if len(partial_matches) > 0:
                all_matches.append(partial_matches)
        
        if all_matches:
            combined = pd.concat(all_matches, ignore_index=True)
            logger.info(f"Smart matching found {len(combined)} records")
            return combined
        
        return pd.DataFrame()
