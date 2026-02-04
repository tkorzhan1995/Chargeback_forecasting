"""
Predictions module for generating and managing forecasts.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
from utils import get_logger
from config.settings import OUTPUT_DIR

logger = get_logger(__name__)


class PredictionEngine:
    """
    Generate and manage chargeback predictions.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize PredictionEngine.
        
        Args:
            output_dir: Directory to save predictions
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_volume_predictions(self, forecast_results: Dict[str, Any],
                                   start_date: datetime = None) -> pd.DataFrame:
        """
        Generate volume prediction dataframe from forecast results.
        
        Args:
            forecast_results: Forecast results from ChargebackForecaster
            start_date: Start date for predictions (default: tomorrow)
            
        Returns:
            pd.DataFrame: Prediction dataframe
        """
        logger.info("Generating volume predictions")
        
        if start_date is None:
            start_date = datetime.now() + timedelta(days=1)
        
        predictions = []
        
        for forecast in forecast_results.get('forecasts', []):
            if 'error' in forecast:
                continue
            
            model_name = forecast['model_name']
            forecast_values = forecast['forecast']
            lower_bound = forecast.get('lower_bound', forecast_values)
            upper_bound = forecast.get('upper_bound', forecast_values)
            
            for i, (pred, lower, upper) in enumerate(zip(forecast_values, lower_bound, upper_bound)):
                pred_date = start_date + timedelta(days=i)
                predictions.append({
                    'date': pred_date,
                    'model': model_name,
                    'predicted_volume': max(0, pred),  # Ensure non-negative
                    'lower_bound': max(0, lower),
                    'upper_bound': max(0, upper),
                })
        
        predictions_df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(predictions_df)} predictions")
        return predictions_df
    
    def generate_transaction_risk_scores(self, transactions: pd.DataFrame,
                                        classifier_model,
                                        threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate risk scores for individual transactions.
        
        Args:
            transactions: Transactions dataframe with features
            classifier_model: Trained classification model
            threshold: Classification threshold
            
        Returns:
            pd.DataFrame: Transactions with risk scores
        """
        logger.info("Generating transaction risk scores")
        
        # Get predictions
        predictions, probabilities = classifier_model.predict(transactions)
        
        # Add to dataframe
        result = transactions.copy()
        result['chargeback_risk_score'] = probabilities
        result['high_risk_flag'] = (probabilities >= threshold).astype(int)
        
        # Categorize risk levels
        result['risk_category'] = pd.cut(
            result['chargeback_risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        high_risk_count = result['high_risk_flag'].sum()
        logger.info(f"Identified {high_risk_count} high-risk transactions")
        
        return result
    
    def calculate_expected_chargeback_amount(self, transactions: pd.DataFrame,
                                            amount_col: str = 'amount') -> pd.DataFrame:
        """
        Calculate expected chargeback amounts based on risk scores.
        
        Args:
            transactions: Transactions with risk scores
            amount_col: Amount column name
            
        Returns:
            pd.DataFrame: Transactions with expected chargeback amounts
        """
        logger.info("Calculating expected chargeback amounts")
        
        result = transactions.copy()
        
        if 'chargeback_risk_score' in result.columns and amount_col in result.columns:
            result['expected_chargeback_amount'] = (
                result['chargeback_risk_score'] * result[amount_col]
            )
        else:
            logger.warning("Required columns not found for expected amount calculation")
            result['expected_chargeback_amount'] = 0
        
        total_expected = result['expected_chargeback_amount'].sum()
        logger.info(f"Total expected chargeback amount: ${total_expected:,.2f}")
        
        return result
    
    def aggregate_predictions_by_dimension(self, predictions: pd.DataFrame,
                                          dimension: str) -> pd.DataFrame:
        """
        Aggregate predictions by a specific dimension.
        
        Args:
            predictions: Predictions dataframe
            dimension: Dimension to aggregate by (e.g., 'product_id', 'channel_id')
            
        Returns:
            pd.DataFrame: Aggregated predictions
        """
        logger.info(f"Aggregating predictions by {dimension}")
        
        if dimension not in predictions.columns:
            logger.warning(f"Dimension {dimension} not found in predictions")
            return pd.DataFrame()
        
        aggregated = predictions.groupby(dimension).agg({
            'chargeback_risk_score': 'mean',
            'expected_chargeback_amount': 'sum',
            'high_risk_flag': 'sum',
        }).reset_index()
        
        aggregated.columns = [dimension, 'avg_risk_score', 'total_expected_amount', 'high_risk_count']
        aggregated = aggregated.sort_values('total_expected_amount', ascending=False)
        
        return aggregated
    
    def generate_scenario_analysis(self, base_predictions: pd.DataFrame,
                                   scenarios: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        """
        Generate what-if scenario analysis.
        
        Args:
            base_predictions: Base prediction dataframe
            scenarios: Dictionary of scenarios with multipliers
                      e.g., {'best_case': 0.8, 'worst_case': 1.2}
            
        Returns:
            Dict: Scenario predictions
        """
        logger.info("Generating scenario analysis")
        
        scenario_results = {'base': base_predictions}
        
        for scenario_name, multiplier in scenarios.items():
            scenario_df = base_predictions.copy()
            
            # Apply multiplier to predictions
            for col in ['predicted_volume', 'expected_chargeback_amount']:
                if col in scenario_df.columns:
                    scenario_df[col] = scenario_df[col] * multiplier
            
            for col in ['lower_bound', 'upper_bound']:
                if col in scenario_df.columns:
                    scenario_df[col] = scenario_df[col] * multiplier
            
            scenario_results[scenario_name] = scenario_df
            logger.info(f"Generated {scenario_name} scenario")
        
        return scenario_results
    
    def create_prediction_summary(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for predictions.
        
        Args:
            predictions: Predictions dataframe
            
        Returns:
            Dict: Summary statistics
        """
        logger.info("Creating prediction summary")
        
        summary = {
            'total_predictions': len(predictions),
            'date_range': {
                'start': predictions['date'].min().isoformat() if 'date' in predictions.columns else None,
                'end': predictions['date'].max().isoformat() if 'date' in predictions.columns else None,
            }
        }
        
        if 'predicted_volume' in predictions.columns:
            summary['volume_stats'] = {
                'total': predictions['predicted_volume'].sum(),
                'daily_avg': predictions['predicted_volume'].mean(),
                'daily_min': predictions['predicted_volume'].min(),
                'daily_max': predictions['predicted_volume'].max(),
            }
        
        if 'expected_chargeback_amount' in predictions.columns:
            summary['amount_stats'] = {
                'total': predictions['expected_chargeback_amount'].sum(),
                'avg': predictions['expected_chargeback_amount'].mean(),
                'min': predictions['expected_chargeback_amount'].min(),
                'max': predictions['expected_chargeback_amount'].max(),
            }
        
        if 'high_risk_flag' in predictions.columns:
            summary['high_risk_count'] = predictions['high_risk_flag'].sum()
            summary['high_risk_percentage'] = (summary['high_risk_count'] / len(predictions) * 100)
        
        return summary
    
    def save_predictions(self, predictions: pd.DataFrame, 
                        filename: str = None,
                        include_summary: bool = True):
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions dataframe
            filename: Output filename (optional)
            include_summary: Whether to save summary as well
        """
        logger.info("Saving predictions")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if filename is None:
            filename = f'predictions_{timestamp}.csv'
        
        output_path = self.output_dir / filename
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        if include_summary:
            summary = self.create_prediction_summary(predictions)
            summary_path = self.output_dir / f'prediction_summary_{timestamp}.json'
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Summary saved to {summary_path}")
    
    def compare_predictions_to_actual(self, predictions: pd.DataFrame,
                                     actuals: pd.DataFrame,
                                     date_col: str = 'date',
                                     pred_col: str = 'predicted_volume',
                                     actual_col: str = 'actual_volume') -> Dict[str, Any]:
        """
        Compare predictions to actual values.
        
        Args:
            predictions: Predictions dataframe
            actuals: Actual values dataframe
            date_col: Date column name
            pred_col: Prediction column name
            actual_col: Actual value column name
            
        Returns:
            Dict: Comparison metrics
        """
        logger.info("Comparing predictions to actual values")
        
        # Merge on date
        comparison = predictions.merge(actuals, on=date_col, how='inner')
        
        if len(comparison) == 0:
            logger.warning("No matching dates found for comparison")
            return {'error': 'No matching dates'}
        
        # Calculate accuracy metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(comparison[actual_col], comparison[pred_col])
        rmse = np.sqrt(mean_squared_error(comparison[actual_col], comparison[pred_col]))
        
        # MAPE
        mask = comparison[actual_col] != 0
        mape = np.mean(np.abs((comparison[actual_col][mask] - comparison[pred_col][mask]) / 
                              comparison[actual_col][mask])) * 100 if mask.any() else 0
        
        metrics = {
            'n_comparisons': len(comparison),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'comparison_data': comparison[[date_col, pred_col, actual_col]].to_dict('records')
        }
        
        logger.info(f"Comparison complete - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return metrics
