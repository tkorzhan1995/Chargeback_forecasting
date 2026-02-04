"""
Model evaluation and performance monitoring module.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from utils import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate and monitor model performance.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        pass
    
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict: Regression metrics
        """
        logger.info("Calculating regression metrics")
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
        }
        
        logger.info(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dict: Classification metrics
        """
        logger.info("Calculating classification metrics")
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, roc_auc_score, confusion_matrix)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"Metrics - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def time_series_cross_validation(self, model, X: np.ndarray, y: np.ndarray,
                                    n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature array
            y: Target array
            n_splits: Number of splits
            
        Returns:
            Dict: Cross-validation results
        """
        logger.info(f"Performing time series cross-validation with {n_splits} splits")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'mse': [],
            'mae': [],
            'r2': [],
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            scores['mse'].append(mean_squared_error(y_test, y_pred))
            scores['mae'].append(mean_absolute_error(y_test, y_pred))
            scores['r2'].append(r2_score(y_test, y_pred))
        
        results = {
            'mean_mse': np.mean(scores['mse']),
            'mean_mae': np.mean(scores['mae']),
            'mean_r2': np.mean(scores['r2']),
            'std_mse': np.std(scores['mse']),
            'std_mae': np.std(scores['mae']),
            'std_r2': np.std(scores['r2']),
        }
        
        logger.info(f"CV Results - Mean MAE: {results['mean_mae']:.4f}, Mean R2: {results['mean_r2']:.4f}")
        return results
    
    def backtest_forecast(self, df: pd.DataFrame, date_col: str, value_col: str,
                         forecast_func, test_size: int = 30) -> Dict[str, Any]:
        """
        Backtest forecasting model.
        
        Args:
            df: Historical dataframe
            date_col: Date column name
            value_col: Value column name
            forecast_func: Forecasting function
            test_size: Number of periods to use for testing
            
        Returns:
            Dict: Backtest results
        """
        logger.info(f"Backtesting forecast with {test_size} test periods")
        
        # Split data
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]
        
        # Generate forecast
        forecast_result = forecast_func(train, date_col, value_col)
        
        if 'error' in forecast_result:
            logger.error(f"Forecast failed during backtesting: {forecast_result['error']}")
            return {'error': forecast_result['error']}
        
        # Get forecast values
        forecast_values = forecast_result['forecast'][:test_size]
        actual_values = test[value_col].values[:test_size]
        
        # Calculate metrics
        metrics = self.calculate_regression_metrics(actual_values, forecast_values)
        
        results = {
            'actual': actual_values,
            'forecast': forecast_values,
            'metrics': metrics,
            'test_size': test_size,
        }
        
        logger.info(f"Backtest complete - MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%")
        return results
    
    def calculate_forecast_accuracy(self, actual: np.ndarray, 
                                   forecast: np.ndarray) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            Dict: Accuracy metrics
        """
        logger.info("Calculating forecast accuracy")
        
        # Basic metrics
        metrics = self.calculate_regression_metrics(actual, forecast)
        
        # Forecast bias
        bias = np.mean(forecast - actual)
        metrics['bias'] = bias
        
        # Theil's U statistic
        mse_forecast = np.mean((forecast - actual) ** 2)
        mse_naive = np.mean((actual[1:] - actual[:-1]) ** 2)
        theils_u = np.sqrt(mse_forecast) / np.sqrt(mse_naive) if mse_naive > 0 else 1.0
        metrics['theils_u'] = theils_u
        
        logger.info(f"Forecast accuracy - Bias: {bias:.4f}, Theil's U: {theils_u:.4f}")
        return metrics
    
    def detect_model_drift(self, recent_metrics: Dict[str, float],
                          baseline_metrics: Dict[str, float],
                          threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect model performance drift.
        
        Args:
            recent_metrics: Recent performance metrics
            baseline_metrics: Baseline performance metrics
            threshold: Threshold for significant drift (10% by default)
            
        Returns:
            Dict: Drift detection results
        """
        logger.info("Detecting model drift")
        
        drift_detected = False
        drift_details = {}
        
        for metric_name in ['mae', 'rmse', 'mape']:
            if metric_name in recent_metrics and metric_name in baseline_metrics:
                recent_val = recent_metrics[metric_name]
                baseline_val = baseline_metrics[metric_name]
                
                if baseline_val > 0:
                    pct_change = (recent_val - baseline_val) / baseline_val
                    
                    if abs(pct_change) > threshold:
                        drift_detected = True
                        drift_details[metric_name] = {
                            'baseline': baseline_val,
                            'recent': recent_val,
                            'pct_change': pct_change * 100,
                        }
        
        result = {
            'drift_detected': drift_detected,
            'details': drift_details,
            'recommendation': 'Retrain model' if drift_detected else 'Model performing well',
        }
        
        if drift_detected:
            logger.warning(f"Model drift detected: {drift_details}")
        else:
            logger.info("No significant model drift detected")
        
        return result
    
    def generate_evaluation_report(self, model_results: Dict[str, Any],
                                  validation_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_results: Model training and prediction results
            validation_results: Cross-validation results (optional)
            
        Returns:
            Dict: Evaluation report
        """
        logger.info("Generating evaluation report")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': model_results.get('model_name', 'Unknown'),
            'training_metrics': model_results.get('metrics', {}),
        }
        
        if validation_results:
            report['cross_validation'] = validation_results
        
        if 'feature_importance' in model_results and model_results['feature_importance'] is not None:
            top_features = model_results['feature_importance'].head(10).to_dict('records')
            report['top_features'] = top_features
        
        return report
