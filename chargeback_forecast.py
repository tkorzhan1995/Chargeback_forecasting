"""
Chargeback Forecasting System
==============================

This module provides functionality for forecasting chargebacks based on historical data,
win/loss ratios, and key drivers. It uses time series analysis and machine learning
techniques to predict future chargeback volumes and amounts.

Author: Chargeback Forecasting Team
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class ChargebackForecaster:
    """
    Main class for chargeback forecasting using multiple modeling approaches.
    
    Attributes:
        data (pd.DataFrame): Historical chargeback data
        models (dict): Dictionary of trained models
        predictions (dict): Dictionary of predictions from different models
    """
    
    def __init__(self, data=None):
        """
        Initialize the ChargebackForecaster.
        
        Args:
            data (pd.DataFrame, optional): Historical chargeback data
        """
        self.data = data
        self.models = {}
        self.predictions = {}
        self.feature_importance = None
        
    def load_data(self, filepath):
        """
        Load chargeback data from a CSV or Excel file.
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if filepath.endswith('.csv'):
            self.data = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(filepath)
        else:
            raise ValueError("File format not supported. Use CSV or Excel.")
        
        print(f"Loaded {len(self.data)} records from {filepath}")
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the data: handle missing values, create features, etc.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        # Convert date column to datetime
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
        
        # Handle missing values
        self.data = self.data.ffill().bfill()
        
        # Create time-based features if date column exists
        if 'date' in self.data.columns:
            self.data['year'] = self.data['date'].dt.year
            self.data['month'] = self.data['date'].dt.month
            self.data['quarter'] = self.data['date'].dt.quarter
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['week_of_year'] = self.data['date'].dt.isocalendar().week
        
        # Calculate rolling statistics
        if 'chargeback_amount' in self.data.columns:
            self.data['rolling_mean_7'] = self.data['chargeback_amount'].rolling(window=7, min_periods=1).mean()
            self.data['rolling_std_7'] = self.data['chargeback_amount'].rolling(window=7, min_periods=1).std()
            self.data['rolling_mean_30'] = self.data['chargeback_amount'].rolling(window=30, min_periods=1).mean()
        
        print("Data preprocessing completed.")
        return self.data
    
    def create_features(self, target_col='chargeback_amount'):
        """
        Create feature matrix and target vector for machine learning models.
        
        Args:
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        # Define feature columns (exclude target and date)
        exclude_cols = [target_col, 'date']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[feature_cols].select_dtypes(include=[np.number])
        # Fill any remaining NaN values
        X = X.fillna(0)
        y = self.data[target_col]
        
        return X, y
    
    def train_ml_models(self, target_col='chargeback_amount', test_size=0.2):
        """
        Train multiple machine learning models for chargeback forecasting.
        
        Args:
            target_col (str): Name of the target column
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Dictionary containing trained models and their performance metrics
        """
        X, y = self.create_features(target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        results = {}
        
        # Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred),
            'predictions': rf_pred
        }
        
        # Gradient Boosting
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        
        self.models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'r2': r2_score(y_test, gb_pred),
            'predictions': gb_pred
        }
        
        # Store feature importance from the best model
        if results['random_forest']['r2'] > results['gradient_boosting']['r2']:
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Print results
        print("\n=== Model Performance ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  MAE: ${metrics['mae']:,.2f}")
            print(f"  RMSE: ${metrics['rmse']:,.2f}")
            print(f"  R²: {metrics['r2']:.4f}")
        
        return results
    
    def train_time_series_models(self, target_col='chargeback_amount', periods=30):
        """
        Train time series models (ARIMA, Exponential Smoothing).
        
        Args:
            target_col (str): Name of the target column
            periods (int): Number of periods to forecast
            
        Returns:
            dict: Dictionary containing forecasts from time series models
        """
        if 'date' not in self.data.columns:
            raise ValueError("Date column required for time series forecasting.")
        
        # Prepare time series data
        ts_data = self.data.set_index('date')[target_col]
        
        results = {}
        
        # ARIMA Model
        try:
            print("Training ARIMA model...")
            arima_model = ARIMA(ts_data, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=periods)
            
            self.models['arima'] = arima_fit
            results['arima'] = {
                'forecast': arima_forecast,
                'aic': arima_fit.aic,
                'bic': arima_fit.bic
            }
            print(f"ARIMA - AIC: {arima_fit.aic:.2f}, BIC: {arima_fit.bic:.2f}")
        except Exception as e:
            print(f"ARIMA model failed: {e}")
        
        # Exponential Smoothing
        try:
            print("Training Exponential Smoothing model...")
            es_model = ExponentialSmoothing(
                ts_data, 
                seasonal_periods=12, 
                trend='add', 
                seasonal='add'
            )
            es_fit = es_model.fit()
            es_forecast = es_fit.forecast(steps=periods)
            
            self.models['exp_smoothing'] = es_fit
            results['exp_smoothing'] = {
                'forecast': es_forecast,
                'sse': es_fit.sse
            }
            print(f"Exponential Smoothing - SSE: {es_fit.sse:.2f}")
        except Exception as e:
            print(f"Exponential Smoothing model failed: {e}")
        
        return results
    
    def forecast(self, periods=30, method='all'):
        """
        Generate forecasts using trained models.
        
        Args:
            periods (int): Number of periods to forecast
            method (str): 'all', 'ml', or 'time_series'
            
        Returns:
            pd.DataFrame: Forecasted values
        """
        forecasts = {}
        
        if method in ['all', 'ml'] and 'random_forest' in self.models:
            # Generate future features (simplified - in practice, need actual feature values)
            X, _ = self.create_features()
            last_features = X.iloc[-1:].values
            
            # Repeat last features for forecasting (simplified approach)
            future_X = np.tile(last_features, (periods, 1))
            
            rf_forecast = self.models['random_forest'].predict(future_X)
            forecasts['random_forest'] = rf_forecast
        
        if method in ['all', 'time_series'] and 'arima' in self.models:
            arima_forecast = self.models['arima'].forecast(steps=periods)
            forecasts['arima'] = arima_forecast.values
        
        if method in ['all', 'time_series'] and 'exp_smoothing' in self.models:
            es_forecast = self.models['exp_smoothing'].forecast(steps=periods)
            forecasts['exp_smoothing'] = es_forecast.values
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts)
        forecast_df['ensemble'] = forecast_df.mean(axis=1)
        
        # Add dates
        if 'date' in self.data.columns:
            last_date = self.data['date'].max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            forecast_df['date'] = forecast_dates
        
        self.predictions = forecast_df
        return forecast_df
    
    def calculate_win_loss_ratio(self, won_col='chargebacks_won', lost_col='chargebacks_lost'):
        """
        Calculate win/loss ratio for chargebacks.
        
        Args:
            won_col (str): Column name for won chargebacks
            lost_col (str): Column name for lost chargebacks
            
        Returns:
            pd.DataFrame: Win/loss statistics
        """
        if won_col not in self.data.columns or lost_col not in self.data.columns:
            print(f"Warning: {won_col} or {lost_col} not found in data.")
            return None
        
        stats = pd.DataFrame({
            'total_won': [self.data[won_col].sum()],
            'total_lost': [self.data[lost_col].sum()],
            'win_rate': [self.data[won_col].sum() / (self.data[won_col].sum() + self.data[lost_col].sum())],
            'loss_rate': [self.data[lost_col].sum() / (self.data[won_col].sum() + self.data[lost_col].sum())]
        })
        
        print("\n=== Win/Loss Statistics ===")
        print(f"Total Won: {stats['total_won'].values[0]}")
        print(f"Total Lost: {stats['total_lost'].values[0]}")
        print(f"Win Rate: {stats['win_rate'].values[0]:.2%}")
        print(f"Loss Rate: {stats['loss_rate'].values[0]:.2%}")
        
        return stats
    
    def plot_forecast(self, save_path=None):
        """
        Plot historical data and forecasts.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if self.predictions is None or len(self.predictions) == 0:
            print("No predictions to plot. Run forecast() first.")
            return
        
        plt.figure(figsize=(14, 7))
        
        # Plot historical data
        if 'date' in self.data.columns and 'chargeback_amount' in self.data.columns:
            plt.plot(self.data['date'], self.data['chargeback_amount'], 
                    label='Historical', linewidth=2, color='blue')
        
        # Plot forecasts
        if 'date' in self.predictions.columns:
            for col in self.predictions.columns:
                if col != 'date':
                    plt.plot(self.predictions['date'], self.predictions[col],
                            label=col.replace('_', ' ').title(), 
                            linestyle='--', linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Chargeback Amount ($)', fontsize=12)
        plt.title('Chargeback Forecasting', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importance from the best ML model.
        
        Args:
            top_n (int): Number of top features to plot
            save_path (str, optional): Path to save the plot
        """
        if self.feature_importance is None:
            print("No feature importance data. Train ML models first.")
            return
        
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path='chargeback_forecast_report.xlsx'):
        """
        Generate a comprehensive Excel report with forecasts and analysis.
        
        Args:
            output_path (str): Path for the output Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Historical data
            if self.data is not None:
                self.data.to_excel(writer, sheet_name='Historical Data', index=False)
            
            # Forecasts
            if self.predictions is not None and len(self.predictions) > 0:
                self.predictions.to_excel(writer, sheet_name='Forecasts', index=False)
            
            # Feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            
            # Summary statistics
            if self.data is not None:
                summary = self.data.describe()
                summary.to_excel(writer, sheet_name='Summary Statistics')
        
        print(f"Report generated: {output_path}")


def generate_sample_data(n_records=365, output_path='sample_data.csv'):
    """
    Generate sample chargeback data for demonstration purposes.
    
    Args:
        n_records (int): Number of records to generate
        output_path (str): Path to save the sample data
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_records, freq='D')
    
    # Generate synthetic chargeback data with trends and seasonality
    trend = np.linspace(1000, 2000, n_records)
    seasonality = 500 * np.sin(2 * np.pi * np.arange(n_records) / 365)
    noise = np.random.normal(0, 200, n_records)
    chargeback_amount = trend + seasonality + noise
    chargeback_amount = np.maximum(chargeback_amount, 0)  # Ensure non-negative
    
    # Generate related features
    data = pd.DataFrame({
        'date': dates,
        'chargeback_amount': chargeback_amount,
        'chargeback_count': np.random.poisson(15, n_records),
        'transaction_volume': chargeback_amount * np.random.uniform(50, 100, n_records),
        'chargebacks_won': np.random.binomial(15, 0.6, n_records),
        'chargebacks_lost': np.random.binomial(15, 0.4, n_records),
        'win_rate': np.random.uniform(0.5, 0.7, n_records),
        'avg_transaction_amount': chargeback_amount / np.random.uniform(10, 20, n_records),
        'merchant_category': np.random.choice(['Retail', 'E-commerce', 'Services', 'Food'], n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records)
    })
    
    data.to_csv(output_path, index=False)
    print(f"Sample data generated: {output_path} ({n_records} records)")
    return data


def main():
    """
    Main function demonstrating the chargeback forecasting workflow.
    """
    print("=" * 60)
    print("CHARGEBACK FORECASTING SYSTEM")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    sample_data = generate_sample_data(n_records=365, output_path='sample_data.csv')
    
    # Initialize forecaster
    print("\n2. Initializing forecaster...")
    forecaster = ChargebackForecaster()
    
    # Load data
    print("\n3. Loading data...")
    forecaster.load_data('sample_data.csv')
    
    # Preprocess data
    print("\n4. Preprocessing data...")
    forecaster.preprocess_data()
    
    # Train ML models
    print("\n5. Training machine learning models...")
    ml_results = forecaster.train_ml_models(target_col='chargeback_amount')
    
    # Train time series models
    print("\n6. Training time series models...")
    ts_results = forecaster.train_time_series_models(target_col='chargeback_amount', periods=30)
    
    # Generate forecasts
    print("\n7. Generating forecasts...")
    forecasts = forecaster.forecast(periods=30, method='all')
    print("\nForecast Summary (first 5 days):")
    print(forecasts.head())
    
    # Calculate win/loss ratio
    print("\n8. Calculating win/loss statistics...")
    win_loss_stats = forecaster.calculate_win_loss_ratio()
    
    # Generate visualizations
    print("\n9. Generating visualizations...")
    forecaster.plot_forecast(save_path='forecast_plot.png')
    forecaster.plot_feature_importance(top_n=10, save_path='feature_importance.png')
    
    # Generate report
    print("\n10. Generating Excel report...")
    forecaster.generate_report(output_path='chargeback_forecast_report.xlsx')
    
    print("\n" + "=" * 60)
    print("FORECASTING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - sample_data.csv: Sample chargeback data")
    print("  - forecast_plot.png: Forecast visualization")
    print("  - feature_importance.png: Feature importance chart")
    print("  - chargeback_forecast_report.xlsx: Comprehensive Excel report")
    

if __name__ == "__main__":
    main()
