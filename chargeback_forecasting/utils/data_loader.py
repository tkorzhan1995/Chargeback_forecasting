"""Data loading and processing utilities."""
import csv
import json
from typing import List, Dict, Any
from datetime import datetime
from chargeback_forecasting.models.chargeback import Chargeback


def load_chargebacks_from_csv(filepath: str) -> List[Chargeback]:
    """
    Load chargeback data from CSV file.
    
    Expected columns: transaction_id, amount, date, reason_code, merchant_id, status, resolved_date, resolution_amount
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of Chargeback objects
    """
    chargebacks = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cb = Chargeback(
                transaction_id=row['transaction_id'],
                amount=float(row['amount']),
                date=datetime.strptime(row['date'], '%Y-%m-%d'),
                reason_code=row['reason_code'],
                merchant_id=row['merchant_id'],
                status=row['status'],
                resolved_date=datetime.strptime(row['resolved_date'], '%Y-%m-%d') if row.get('resolved_date') else None,
                resolution_amount=float(row['resolution_amount']) if row.get('resolution_amount') else None
            )
            chargebacks.append(cb)
    
    return chargebacks


def load_chargebacks_from_json(filepath: str) -> List[Chargeback]:
    """
    Load chargeback data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of Chargeback objects
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    chargebacks = []
    for item in data:
        cb = Chargeback(
            transaction_id=item['transaction_id'],
            amount=float(item['amount']),
            date=datetime.strptime(item['date'], '%Y-%m-%d'),
            reason_code=item['reason_code'],
            merchant_id=item['merchant_id'],
            status=item['status'],
            resolved_date=datetime.strptime(item['resolved_date'], '%Y-%m-%d') if item.get('resolved_date') else None,
            resolution_amount=float(item['resolution_amount']) if item.get('resolution_amount') else None
        )
        chargebacks.append(cb)
    
    return chargebacks


def load_transactions_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load transaction data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            txn = dict(row)
            # Convert numeric fields
            if 'amount' in txn:
                txn['amount'] = float(txn['amount'])
            if 'date' in txn:
                txn['date'] = datetime.strptime(txn['date'], '%Y-%m-%d')
            transactions.append(txn)
    
    return transactions


def load_transactions_from_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load transaction data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of transaction dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for txn in data:
        if 'amount' in txn:
            txn['amount'] = float(txn['amount'])
        if 'date' in txn:
            txn['date'] = datetime.strptime(txn['date'], '%Y-%m-%d')
    
    return data


def calculate_transaction_counts(transactions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate transaction counts by period.
    
    Args:
        transactions: List of transactions
        
    Returns:
        Dictionary mapping period to count
    """
    counts = {}
    
    for txn in transactions:
        if 'date' in txn:
            period = txn['date'].strftime('%Y-%m')
            counts[period] = counts.get(period, 0) + 1
    
    return counts


def export_forecast_to_csv(forecast_results: List[Any], filepath: str):
    """
    Export forecast results to CSV file.
    
    Args:
        forecast_results: List of ForecastResult objects
        filepath: Output CSV file path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Period',
            'Expected Chargebacks',
            'Expected Rate',
            'CI Low',
            'CI High',
            'Win Rate',
            'Loss Rate'
        ])
        
        for result in forecast_results:
            writer.writerow([
                result.forecast_period,
                f"{result.expected_chargebacks:.2f}",
                f"{result.expected_chargeback_rate:.4%}",
                f"{result.confidence_interval_low:.2f}",
                f"{result.confidence_interval_high:.2f}",
                f"{result.expected_win_rate:.2%}",
                f"{result.expected_loss_rate:.2%}"
            ])


def export_forecast_to_json(forecast_results: List[Any], filepath: str):
    """
    Export forecast results to JSON file.
    
    Args:
        forecast_results: List of ForecastResult objects
        filepath: Output JSON file path
    """
    data = []
    for result in forecast_results:
        data.append({
            'forecast_period': result.forecast_period,
            'expected_chargebacks': result.expected_chargebacks,
            'expected_chargeback_rate': result.expected_chargeback_rate,
            'confidence_interval_low': result.confidence_interval_low,
            'confidence_interval_high': result.confidence_interval_high,
            'expected_win_rate': result.expected_win_rate,
            'expected_loss_rate': result.expected_loss_rate,
            'key_drivers': [
                {
                    'name': driver.driver_name,
                    'type': driver.driver_type,
                    'impact': driver.impact_score,
                    'correlation': driver.correlation
                }
                for driver in result.key_drivers
            ],
            'assumptions': result.assumptions
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
