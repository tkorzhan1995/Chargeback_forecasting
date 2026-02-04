# Chargeback Forecasting - Usage Examples

This document provides practical examples for using the Chargeback Forecasting system.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Advanced Usage](#advanced-usage)
3. [Custom Data](#custom-data)
4. [Model Selection](#model-selection)
5. [Batch Processing](#batch-processing)
6. [Integration Examples](#integration-examples)

---

## Basic Usage

### Example 1: Quick Forecast with Sample Data

```python
from chargeback_forecast import ChargebackForecaster, generate_sample_data

# Generate sample data
generate_sample_data(n_records=365, output_path='data.csv')

# Create forecaster and load data
forecaster = ChargebackForecaster()
forecaster.load_data('data.csv')
forecaster.preprocess_data()

# Train and forecast
forecaster.train_ml_models()
forecasts = forecaster.forecast(periods=30)

print(forecasts[['date', 'ensemble']].head())
```

### Example 2: Generate Report Only

```python
from chargeback_forecast import ChargebackForecaster

forecaster = ChargebackForecaster()
forecaster.load_data('your_data.csv')
forecaster.preprocess_data()
forecaster.train_ml_models()
forecaster.forecast(periods=30)

# Generate Excel report
forecaster.generate_report(output_path='monthly_report.xlsx')
```

---

## Advanced Usage

### Example 3: Custom Model Training

```python
from chargeback_forecast import ChargebackForecaster

forecaster = ChargebackForecaster()
forecaster.load_data('chargeback_data.csv')
forecaster.preprocess_data()

# Train ML models with custom test split
ml_results = forecaster.train_ml_models(
    target_col='chargeback_amount',
    test_size=0.3  # 30% test data
)

# Train time series models for 60 days
ts_results = forecaster.train_time_series_models(
    target_col='chargeback_amount',
    periods=60
)

# Print model performance
for model_name, metrics in ml_results.items():
    print(f"{model_name}: R² = {metrics['r2']:.4f}")
```

### Example 4: Comparing Different Forecast Horizons

```python
from chargeback_forecast import ChargebackForecaster
import pandas as pd

forecaster = ChargebackForecaster()
forecaster.load_data('data.csv')
forecaster.preprocess_data()
forecaster.train_ml_models()
forecaster.train_time_series_models()

# Generate forecasts for different periods
forecasts_7d = forecaster.forecast(periods=7, method='all')
forecasts_30d = forecaster.forecast(periods=30, method='all')
forecasts_90d = forecaster.forecast(periods=90, method='all')

print("7-day forecast mean:", forecasts_7d['ensemble'].mean())
print("30-day forecast mean:", forecasts_30d['ensemble'].mean())
print("90-day forecast mean:", forecasts_90d['ensemble'].mean())
```

### Example 5: Feature Importance Analysis

```python
from chargeback_forecast import ChargebackForecaster

forecaster = ChargebackForecaster()
forecaster.load_data('data.csv')
forecaster.preprocess_data()
forecaster.train_ml_models()

# Get top features
top_features = forecaster.feature_importance.head(10)
print("\nTop 10 Most Important Features:")
print(top_features.to_string(index=False))

# Save feature importance plot
forecaster.plot_feature_importance(top_n=15, save_path='top_features.png')
```

---

## Custom Data

### Example 6: Loading Your Own Data

```python
from chargeback_forecast import ChargebackForecaster
import pandas as pd

# Prepare your data
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365),
    'chargeback_amount': [1000, 1100, 950, ...],  # Your actual data
    'chargeback_count': [10, 12, 8, ...],
    'transaction_volume': [50000, 52000, 48000, ...],
    'chargebacks_won': [6, 7, 5, ...],
    'chargebacks_lost': [4, 5, 3, ...],
    # Add any other relevant columns
})

# Save to CSV
data.to_csv('my_chargeback_data.csv', index=False)

# Use with forecaster
forecaster = ChargebackForecaster()
forecaster.load_data('my_chargeback_data.csv')
forecaster.preprocess_data()
forecaster.train_ml_models()
forecasts = forecaster.forecast(periods=30)
```

### Example 7: Handling Missing Data

```python
from chargeback_forecast import ChargebackForecaster
import pandas as pd

# Load data with missing values
forecaster = ChargebackForecaster()
data = pd.read_csv('incomplete_data.csv')

# Check missing values before
print("Missing values before:")
print(data.isnull().sum())

forecaster.data = data
forecaster.preprocess_data()

# Check after preprocessing (should be handled)
print("\nMissing values after:")
print(forecaster.data.isnull().sum())
```

---

## Model Selection

### Example 8: Using Only Time Series Models

```python
from chargeback_forecast import ChargebackForecaster

forecaster = ChargebackForecaster()
forecaster.load_data('data.csv')
forecaster.preprocess_data()

# Train only time series models
forecaster.train_time_series_models(target_col='chargeback_amount', periods=30)

# Forecast with time series only
forecasts = forecaster.forecast(periods=30, method='time_series')

print(forecasts[['date', 'arima', 'exp_smoothing']].head())
```

### Example 9: Using Only ML Models

```python
from chargeback_forecast import ChargebackForecaster

forecaster = ChargebackForecaster()
forecaster.load_data('data.csv')
forecaster.preprocess_data()

# Train only ML models
forecaster.train_ml_models(target_col='chargeback_amount')

# Forecast with ML only
forecasts = forecaster.forecast(periods=30, method='ml')

print(forecasts[['date', 'random_forest']].head())
```

---

## Batch Processing

### Example 10: Process Multiple Files

```python
from chargeback_forecast import ChargebackForecaster
import os
import glob

# Process all CSV files in a directory
data_files = glob.glob('data/*.csv')

for file in data_files:
    print(f"\nProcessing {file}...")
    
    forecaster = ChargebackForecaster()
    forecaster.load_data(file)
    forecaster.preprocess_data()
    forecaster.train_ml_models()
    forecaster.train_time_series_models()
    forecaster.forecast(periods=30)
    
    # Generate report with unique name
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_name = f'reports/{base_name}_forecast.xlsx'
    forecaster.generate_report(output_path=output_name)
    
    print(f"Report saved: {output_name}")
```

### Example 11: Automated Monthly Forecasting

```python
from chargeback_forecast import ChargebackForecaster
from datetime import datetime
import schedule
import time

def run_monthly_forecast():
    """Run forecast and generate report"""
    print(f"Running forecast at {datetime.now()}")
    
    forecaster = ChargebackForecaster()
    forecaster.load_data('current_data.csv')
    forecaster.preprocess_data()
    forecaster.train_ml_models()
    forecaster.train_time_series_models()
    forecaster.forecast(periods=30)
    
    # Generate report with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    forecaster.generate_report(output_path=f'reports/forecast_{timestamp}.xlsx')
    
    print(f"Forecast complete: forecast_{timestamp}.xlsx")

# Schedule monthly execution
schedule.every().month.at("09:00").do(run_monthly_forecast)

# Or run immediately for testing
run_monthly_forecast()

# Keep running (for production)
# while True:
#     schedule.run_pending()
#     time.sleep(3600)  # Check every hour
```

---

## Integration Examples

### Example 12: API Integration (Flask)

```python
from flask import Flask, jsonify, request
from chargeback_forecast import ChargebackForecaster
import pandas as pd

app = Flask(__name__)
forecaster = None

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the forecasting model"""
    global forecaster
    
    data = request.get_json()
    df = pd.DataFrame(data)
    
    forecaster = ChargebackForecaster()
    forecaster.data = df
    forecaster.preprocess_data()
    forecaster.train_ml_models()
    
    return jsonify({'status': 'success', 'message': 'Model trained'})

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get forecast for specified periods"""
    if forecaster is None:
        return jsonify({'error': 'Model not trained'}), 400
    
    periods = int(request.args.get('periods', 30))
    forecasts = forecaster.forecast(periods=periods)
    
    return jsonify(forecasts.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Example 13: Database Integration

```python
from chargeback_forecast import ChargebackForecaster
import pandas as pd
import sqlalchemy

# Connect to database
engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/db')

# Load data from database
query = """
    SELECT 
        transaction_date as date,
        SUM(chargeback_amount) as chargeback_amount,
        COUNT(*) as chargeback_count,
        SUM(transaction_amount) as transaction_volume,
        SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) as chargebacks_won,
        SUM(CASE WHEN outcome = 'lost' THEN 1 ELSE 0 END) as chargebacks_lost
    FROM chargebacks
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY transaction_date
    ORDER BY transaction_date
"""

data = pd.read_sql(query, engine)
data.to_csv('db_export.csv', index=False)

# Use with forecaster
forecaster = ChargebackForecaster()
forecaster.data = data
forecaster.preprocess_data()
forecaster.train_ml_models()
forecasts = forecaster.forecast(periods=30)

# Save forecasts back to database
forecasts.to_sql('chargeback_forecasts', engine, if_exists='replace', index=False)
print("Forecasts saved to database")
```

### Example 14: Email Report Automation

```python
from chargeback_forecast import ChargebackForecaster
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

def send_forecast_report():
    """Generate forecast and email report"""
    
    # Generate forecast
    forecaster = ChargebackForecaster()
    forecaster.load_data('data.csv')
    forecaster.preprocess_data()
    forecaster.train_ml_models()
    forecaster.forecast(periods=30)
    forecaster.generate_report(output_path='weekly_forecast.xlsx')
    
    # Prepare email
    msg = MIMEMultipart()
    msg['From'] = 'forecast@company.com'
    msg['To'] = 'manager@company.com'
    msg['Subject'] = 'Weekly Chargeback Forecast Report'
    
    body = """
    Please find attached the weekly chargeback forecast report.
    
    Key Highlights:
    - 30-day forecast generated
    - Model performance: R² > 0.64
    - Win rate analysis included
    
    Best regards,
    Forecasting System
    """
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach report
    with open('weekly_forecast.xlsx', 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename=weekly_forecast.xlsx')
        msg.attach(part)
    
    # Send email
    server = smtplib.SMTP('smtp.company.com', 587)
    server.starttls()
    server.login('forecast@company.com', 'password')
    server.send_message(msg)
    server.quit()
    
    print("Report emailed successfully")

# Run weekly
import schedule
schedule.every().monday.at("08:00").do(send_forecast_report)
```

---

## Tips and Best Practices

1. **Data Quality**: Ensure your data has consistent date formats and no excessive missing values
2. **Regular Retraining**: Retrain models monthly or when significant business changes occur
3. **Validation**: Always validate forecasts against domain knowledge
4. **Feature Engineering**: Add business-specific features for better predictions
5. **Model Selection**: Use ensemble predictions for more robust forecasts
6. **Performance Monitoring**: Track forecast accuracy over time and adjust as needed

## Common Issues and Solutions

### Issue: Poor Model Performance
**Solution**: 
- Check data quality and completeness
- Add more relevant features
- Increase training data size
- Try different forecasting horizons

### Issue: Forecasts Don't Match Business Reality
**Solution**:
- Validate input data
- Consider seasonal factors
- Add business rules or constraints
- Consult domain experts

### Issue: Long Training Time
**Solution**:
- Reduce data volume (focus on recent data)
- Use fewer models
- Optimize feature engineering
- Consider sampling large datasets

---

For more information, see:
- `README.md` - Project overview and installation
- `PowerBI_Dashboard_Guide.md` - Dashboard creation
- `chargeback_forecast.py` - Full API documentation
