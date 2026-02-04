"""
Reconciliation reports generation module.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from utils import get_logger, format_currency, calculate_percentage_change
from config.settings import OUTPUT_DIR

logger = get_logger(__name__)


class ReconciliationReports:
    """
    Generate reconciliation reports and analytics.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize ReconciliationReports.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_summary_report(self, matched: pd.DataFrame, 
                               unmatched: pd.DataFrame,
                               total_chargebacks: int) -> Dict[str, Any]:
        """
        Generate summary reconciliation report.
        
        Args:
            matched: Matched records dataframe
            unmatched: Unmatched records dataframe
            total_chargebacks: Total number of chargebacks processed
            
        Returns:
            Dict: Summary statistics
        """
        logger.info("Generating summary reconciliation report")
        
        matched_count = len(matched)
        unmatched_count = len(unmatched)
        match_rate = (matched_count / total_chargebacks * 100) if total_chargebacks > 0 else 0
        
        # Calculate confidence distribution
        if len(matched) > 0 and 'match_confidence' in matched.columns:
            high_confidence = len(matched[matched['match_confidence'] >= 0.95])
            medium_confidence = len(matched[(matched['match_confidence'] >= 0.85) & 
                                           (matched['match_confidence'] < 0.95)])
            low_confidence = len(matched[matched['match_confidence'] < 0.85])
        else:
            high_confidence = medium_confidence = low_confidence = 0
        
        # Calculate amounts
        if len(matched) > 0:
            if 'amount_chargeback' in matched.columns:
                matched_amount = matched['amount_chargeback'].sum()
            elif 'amount' in matched.columns:
                matched_amount = matched['amount'].sum()
            else:
                matched_amount = 0
        else:
            matched_amount = 0
        
        if len(unmatched) > 0 and 'amount' in unmatched.columns:
            unmatched_amount = unmatched['amount'].sum()
        else:
            unmatched_amount = 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_chargebacks': total_chargebacks,
            'matched_count': matched_count,
            'unmatched_count': unmatched_count,
            'match_rate_percent': round(match_rate, 2),
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence,
            },
            'amounts': {
                'matched_total': round(matched_amount, 2),
                'unmatched_total': round(unmatched_amount, 2),
                'total': round(matched_amount + unmatched_amount, 2),
            }
        }
        
        logger.info(f"Summary: {match_rate:.2f}% match rate, "
                   f"{matched_count} matched, {unmatched_count} unmatched")
        
        return summary
    
    def generate_match_method_report(self, matched: pd.DataFrame) -> pd.DataFrame:
        """
        Generate report showing breakdown by matching method.
        
        Args:
            matched: Matched records dataframe
            
        Returns:
            pd.DataFrame: Match method breakdown
        """
        logger.info("Generating match method report")
        
        if len(matched) == 0 or 'match_method' not in matched.columns:
            return pd.DataFrame()
        
        method_stats = matched.groupby('match_method').agg({
            'chargeback_id': 'count',
            'match_confidence': ['mean', 'min', 'max'],
        }).reset_index()
        
        method_stats.columns = ['match_method', 'count', 'avg_confidence', 
                               'min_confidence', 'max_confidence']
        
        method_stats['percentage'] = (method_stats['count'] / len(matched) * 100).round(2)
        
        return method_stats
    
    def generate_unmatched_analysis(self, unmatched: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze unmatched records to identify patterns.
        
        Args:
            unmatched: Unmatched chargebacks dataframe
            
        Returns:
            Dict: Analysis of unmatched records
        """
        logger.info("Analyzing unmatched records")
        
        if len(unmatched) == 0:
            return {'count': 0, 'patterns': []}
        
        analysis = {
            'count': len(unmatched),
            'total_amount': round(unmatched['amount'].sum(), 2) if 'amount' in unmatched.columns else 0,
        }
        
        # Analyze by reason code if available
        if 'reason_code' in unmatched.columns:
            reason_breakdown = unmatched['reason_code'].value_counts().to_dict()
            analysis['by_reason_code'] = reason_breakdown
        
        # Analyze by channel if available
        if 'channel_id' in unmatched.columns:
            channel_breakdown = unmatched['channel_id'].value_counts().to_dict()
            analysis['by_channel'] = channel_breakdown
        
        # Analyze by date range if available
        if 'chargeback_date' in unmatched.columns:
            unmatched['chargeback_date'] = pd.to_datetime(unmatched['chargeback_date'])
            date_range = {
                'earliest': unmatched['chargeback_date'].min().isoformat() if pd.notna(unmatched['chargeback_date'].min()) else None,
                'latest': unmatched['chargeback_date'].max().isoformat() if pd.notna(unmatched['chargeback_date'].max()) else None,
            }
            analysis['date_range'] = date_range
        
        return analysis
    
    def generate_data_quality_report(self, chargebacks: pd.DataFrame,
                                    transactions: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data quality metrics report.
        
        Args:
            chargebacks: Chargebacks dataframe
            transactions: Transactions dataframe
            
        Returns:
            Dict: Data quality metrics
        """
        logger.info("Generating data quality report")
        
        def calculate_completeness(df: pd.DataFrame) -> Dict[str, float]:
            """Calculate completeness for each column."""
            completeness = {}
            for col in df.columns:
                non_null = df[col].notna().sum()
                completeness[col] = round(non_null / len(df) * 100, 2)
            return completeness
        
        report = {
            'chargebacks': {
                'total_records': len(chargebacks),
                'duplicates': chargebacks.duplicated().sum(),
                'completeness': calculate_completeness(chargebacks),
            },
            'transactions': {
                'total_records': len(transactions),
                'duplicates': transactions.duplicated().sum(),
                'completeness': calculate_completeness(transactions),
            }
        }
        
        return report
    
    def generate_reconciliation_detail_report(self, matched: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed reconciliation report with all matched records.
        
        Args:
            matched: Matched records dataframe
            
        Returns:
            pd.DataFrame: Detailed report
        """
        logger.info("Generating detailed reconciliation report")
        
        if len(matched) == 0:
            return pd.DataFrame()
        
        # Select key columns for the report
        report_columns = [
            'chargeback_id', 'transaction_id', 'match_confidence', 'match_method',
        ]
        
        # Add additional columns if they exist
        optional_columns = [
            'customer_id', 'product_id', 'channel_id',
            'amount_chargeback', 'amount_transaction',
            'chargeback_date', 'transaction_date',
            'reason_code', 'status'
        ]
        
        for col in optional_columns:
            if col in matched.columns:
                report_columns.append(col)
        
        detail_report = matched[report_columns].copy()
        
        return detail_report
    
    def save_reports(self, matched: pd.DataFrame, unmatched: pd.DataFrame,
                    total_chargebacks: int, chargebacks: pd.DataFrame = None,
                    transactions: pd.DataFrame = None):
        """
        Save all reports to files.
        
        Args:
            matched: Matched records dataframe
            unmatched: Unmatched records dataframe
            total_chargebacks: Total number of chargebacks
            chargebacks: Original chargebacks dataframe (optional)
            transactions: Original transactions dataframe (optional)
        """
        logger.info("Saving reconciliation reports")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Summary report
        summary = self.generate_summary_report(matched, unmatched, total_chargebacks)
        summary_path = self.output_dir / f'reconciliation_summary_{timestamp}.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary report to {summary_path}")
        
        # Matched records
        if len(matched) > 0:
            matched_path = self.output_dir / f'matched_records_{timestamp}.csv'
            matched.to_csv(matched_path, index=False)
            logger.info(f"Saved matched records to {matched_path}")
            
            # Match method report
            method_report = self.generate_match_method_report(matched)
            if len(method_report) > 0:
                method_path = self.output_dir / f'match_methods_{timestamp}.csv'
                method_report.to_csv(method_path, index=False)
                logger.info(f"Saved match method report to {method_path}")
        
        # Unmatched records
        if len(unmatched) > 0:
            unmatched_path = self.output_dir / f'unmatched_records_{timestamp}.csv'
            unmatched.to_csv(unmatched_path, index=False)
            logger.info(f"Saved unmatched records to {unmatched_path}")
            
            # Unmatched analysis
            unmatched_analysis = self.generate_unmatched_analysis(unmatched)
            analysis_path = self.output_dir / f'unmatched_analysis_{timestamp}.json'
            with open(analysis_path, 'w') as f:
                json.dump(unmatched_analysis, f, indent=2)
            logger.info(f"Saved unmatched analysis to {analysis_path}")
        
        # Data quality report
        if chargebacks is not None and transactions is not None:
            quality_report = self.generate_data_quality_report(chargebacks, transactions)
            quality_path = self.output_dir / f'data_quality_{timestamp}.json'
            with open(quality_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            logger.info(f"Saved data quality report to {quality_path}")
        
        logger.info("All reports saved successfully")
