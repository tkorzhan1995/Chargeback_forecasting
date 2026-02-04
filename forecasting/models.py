"""
Forecasting models for chargeback prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils import get_logger
from config.settings import FORECASTING_CONFIG, MODEL_CONFIG

logger = get_logger(__name__)


class ChargebackForecaster:
    """
    Time series forecasting for chargeback volumes.
    """
    
    def __init__(self):
        """Initialize ChargebackForecaster."""
        self.models = {}
        self.forecast_horizon = FORECASTING_CONFIG['forecast_horizon']
        self.confidence_level = FORECASTING_CONFIG['confidence_level']
    
    def prepare_time_series(self, df: pd.DataFrame, date_col: str, 
                           value_col: str, freq: str = 'D') -> pd.DataFrame:
        """
        Prepare time series data for forecasting.
        
        Args:
            df: Input dataframe
            date_col: Date column name
            value_col: Value column name
            freq: Frequency ('D', 'W', 'M')
            
        Returns:
            pd.DataFrame: Time series dataframe
        """
        logger.info("Preparing time series data")
        
        df[date_col] = pd.to_datetime(df[date_col])
        ts = df.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum().reset_index()
        ts = ts.sort_values(date_col)
        
        # Fill missing dates
        date_range = pd.date_range(start=ts[date_col].min(), end=ts[date_col].max(), freq=freq)
        ts_complete = pd.DataFrame({date_col: date_range})
        ts_complete = ts_complete.merge(ts, on=date_col, how='left')
        ts_complete[value_col].fillna(0, inplace=True)
        
        return ts_complete
    
    def arima_forecast(self, ts: pd.DataFrame, date_col: str, value_col: str,
                      order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """
        ARIMA forecasting model.
        
        Args:
            ts: Time series dataframe
            date_col: Date column name
            value_col: Value column name
            order: ARIMA order (p, d, q)
            
        Returns:
            Dict: Forecast results
        """
        logger.info(f"Training ARIMA model with order {order}")
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Prepare data
            y = ts[value_col].values
            
            # Fit model
            model = ARIMA(y, order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=self.forecast_horizon)
            
            # Get confidence intervals
            forecast_df = fitted_model.get_forecast(steps=self.forecast_horizon)
            conf_int = forecast_df.conf_int(alpha=1-self.confidence_level)
            
            # Create forecast dates
            last_date = ts[date_col].max()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                          periods=self.forecast_horizon, freq='D')
            
            results = {
                'model_name': 'ARIMA',
                'forecast': forecast,
                'forecast_dates': forecast_dates,
                'lower_bound': conf_int.iloc[:, 0].values,
                'upper_bound': conf_int.iloc[:, 1].values,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
            }
            
            self.models['arima'] = fitted_model
            logger.info("ARIMA model trained successfully")
            return results
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {str(e)}")
            return {'model_name': 'ARIMA', 'error': str(e)}
    
    def prophet_forecast(self, ts: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Prophet forecasting model.
        
        Args:
            ts: Time series dataframe
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Dict: Forecast results
        """
        logger.info("Training Prophet model")
        
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = ts[[date_col, value_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Initialize and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=self.confidence_level
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=self.forecast_horizon)
            
            # Forecast
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast.tail(self.forecast_horizon)
            
            results = {
                'model_name': 'Prophet',
                'forecast': forecast_values['yhat'].values,
                'forecast_dates': forecast_values['ds'].values,
                'lower_bound': forecast_values['yhat_lower'].values,
                'upper_bound': forecast_values['yhat_upper'].values,
                'components': {
                    'trend': forecast_values['trend'].values,
                    'weekly': forecast_values.get('weekly', pd.Series([0]*len(forecast_values))).values,
                    'yearly': forecast_values.get('yearly', pd.Series([0]*len(forecast_values))).values,
                }
            }
            
            self.models['prophet'] = model
            logger.info("Prophet model trained successfully")
            return results
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            return {'model_name': 'Prophet', 'error': str(e)}
    
    def moving_average_forecast(self, ts: pd.DataFrame, value_col: str, 
                               window: int = 7) -> Dict[str, Any]:
        """
        Simple moving average forecast.
        
        Args:
            ts: Time series dataframe
            value_col: Value column name
            window: Moving average window size
            
        Returns:
            Dict: Forecast results
        """
        logger.info(f"Creating moving average forecast with window={window}")
        
        # Calculate moving average
        ma = ts[value_col].rolling(window=window).mean().iloc[-1]
        
        # Forecast is constant (simple approach)
        forecast = np.full(self.forecast_horizon, ma)
        
        # Estimate confidence interval based on historical std
        std = ts[value_col].rolling(window=window).std().iloc[-1]
        z_score = 1.96  # 95% confidence
        lower_bound = forecast - z_score * std
        upper_bound = forecast + z_score * std
        
        results = {
            'model_name': 'MovingAverage',
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'window': window,
        }
        
        logger.info("Moving average forecast complete")
        return results
    
    def exponential_smoothing_forecast(self, ts: pd.DataFrame, value_col: str) -> Dict[str, Any]:
        """
        Exponential smoothing forecast.
        
        Args:
            ts: Time series dataframe
            value_col: Value column name
            
        Returns:
            Dict: Forecast results
        """
        logger.info("Training Exponential Smoothing model")
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            y = ts[value_col].values
            
            # Fit model
            model = ExponentialSmoothing(y, seasonal_periods=7, trend='add', seasonal='add')
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=self.forecast_horizon)
            
            # Estimate confidence interval (simplified)
            residuals = fitted_model.fittedvalues - y
            std_error = np.std(residuals)
            z_score = 1.96
            lower_bound = forecast - z_score * std_error
            upper_bound = forecast + z_score * std_error
            
            results = {
                'model_name': 'ExponentialSmoothing',
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'aic': fitted_model.aic,
            }
            
            self.models['exp_smoothing'] = fitted_model
            logger.info("Exponential Smoothing model trained successfully")
            return results
            
        except Exception as e:
            logger.error(f"Exponential Smoothing forecasting failed: {str(e)}")
            return {'model_name': 'ExponentialSmoothing', 'error': str(e)}
    
    def ensemble_forecast(self, forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ensemble multiple forecasts.
        
        Args:
            forecasts: List of forecast results from different models
            
        Returns:
            Dict: Ensemble forecast
        """
        logger.info("Creating ensemble forecast")
        
        valid_forecasts = [f for f in forecasts if 'error' not in f]
        
        if not valid_forecasts:
            logger.error("No valid forecasts to ensemble")
            return {'model_name': 'Ensemble', 'error': 'No valid forecasts'}
        
        # Simple average ensemble
        forecast_values = np.array([f['forecast'] for f in valid_forecasts])
        ensemble_forecast = np.mean(forecast_values, axis=0)
        
        # Ensemble confidence intervals
        lower_bounds = np.array([f.get('lower_bound', f['forecast']) for f in valid_forecasts])
        upper_bounds = np.array([f.get('upper_bound', f['forecast']) for f in valid_forecasts])
        
        ensemble_lower = np.mean(lower_bounds, axis=0)
        ensemble_upper = np.mean(upper_bounds, axis=0)
        
        results = {
            'model_name': 'Ensemble',
            'forecast': ensemble_forecast,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper,
            'num_models': len(valid_forecasts),
            'models_used': [f['model_name'] for f in valid_forecasts],
        }
        
        logger.info(f"Ensemble forecast created from {len(valid_forecasts)} models")
        return results
    
    def forecast_chargebacks(self, df: pd.DataFrame, date_col: str = 'chargeback_date',
                            models: List[str] = None) -> Dict[str, Any]:
        """
        Main forecasting function.
        
        Args:
            df: Chargeback dataframe
            date_col: Date column name
            models: List of models to use (if None, use all)
            
        Returns:
            Dict: All forecast results
        """
        logger.info("Starting chargeback forecasting")
        
        # Prepare time series
        ts = self.prepare_time_series(df, date_col, 'chargeback_id', freq='D')
        ts.rename(columns={'chargeback_id': 'count'}, inplace=True)
        
        forecasts = []
        models_to_use = models or ['arima', 'prophet', 'moving_average', 'exp_smoothing']
        
        # Run each model
        if 'arima' in models_to_use:
            arima_result = self.arima_forecast(ts, date_col, 'count')
            forecasts.append(arima_result)
        
        if 'prophet' in models_to_use:
            prophet_result = self.prophet_forecast(ts, date_col, 'count')
            forecasts.append(prophet_result)
        
        if 'moving_average' in models_to_use:
            ma_result = self.moving_average_forecast(ts, 'count')
            forecasts.append(ma_result)
        
        if 'exp_smoothing' in models_to_use:
            es_result = self.exponential_smoothing_forecast(ts, 'count')
            forecasts.append(es_result)
        
        # Create ensemble
        ensemble_result = self.ensemble_forecast(forecasts)
        forecasts.append(ensemble_result)
        
        results = {
            'historical_data': ts,
            'forecasts': forecasts,
            'best_model': self._select_best_model(forecasts),
        }
        
        logger.info("Forecasting complete")
        return results
    
    def _select_best_model(self, forecasts: List[Dict[str, Any]]) -> str:
        """
        Select best model based on AIC/BIC if available, otherwise use ensemble.
        
        Args:
            forecasts: List of forecast results
            
        Returns:
            str: Best model name
        """
        # Prefer models with AIC
        aic_models = [f for f in forecasts if 'aic' in f]
        
        if aic_models:
            best = min(aic_models, key=lambda x: x['aic'])
            return best['model_name']
        
        # Default to ensemble
        return 'Ensemble'


class ChargebackClassifier:
    """
    Classification model to predict chargeback likelihood for individual transactions.
    """
    
    def __init__(self):
        """Initialize ChargebackClassifier."""
        self.model = None
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train classification model.
        
        Args:
            X: Feature dataframe
            y: Target variable (chargeback indicator)
            model_type: Type of model ('random_forest', 'logistic', 'xgboost')
            
        Returns:
            Dict: Training results
        """
        logger.info(f"Training {model_type} classifier")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=MODEL_CONFIG['random_state'], stratify=y
        )
        
        # Select model
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=MODEL_CONFIG['random_state'])
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=MODEL_CONFIG['random_state'], max_iter=1000)
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=MODEL_CONFIG['random_state'])
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=MODEL_CONFIG['random_state'])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = None
        
        self.model = model
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict chargeback likelihood.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
