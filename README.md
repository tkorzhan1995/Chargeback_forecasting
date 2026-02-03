# Chargeback Forecasting System

A comprehensive machine learning and time series forecasting system for predicting chargebacks based on historical rates, win/loss ratios, and key business drivers.

## 📊 Overview

This project provides an end-to-end solution for chargeback forecasting using:
- **Machine Learning Models**: Random Forest and Gradient Boosting algorithms
- **Time Series Models**: ARIMA and Exponential Smoothing for temporal patterns
- **Win/Loss Analysis**: Track and analyze chargeback dispute outcomes
- **Feature Engineering**: Automated creation of predictive features
- **Power BI Integration**: Dashboard templates for visualization

## 🚀 Features

- ✅ Multiple forecasting algorithms with ensemble predictions
- ✅ Automated data preprocessing and feature engineering
- ✅ Win/loss ratio tracking and analysis
- ✅ Feature importance analysis
- ✅ Comprehensive Excel reports
- ✅ Visualization of forecasts and trends
- ✅ Power BI dashboard template
- ✅ Sample data generation for testing

## 📋 Requirements

- Python 3.8 or higher
- Power BI Desktop (for dashboard visualization)

### Python Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.14.0
openpyxl>=3.1.0
python-dateutil>=2.8.0
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/tkorzhan1995/Chargeback_forecasting.git
cd Chargeback_forecasting
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Quick Start

Run the complete forecasting pipeline with sample data:

```bash
python chargeback_forecast.py
```

This will:
1. Generate sample chargeback data
2. Preprocess and engineer features
3. Train ML and time series models
4. Generate 30-day forecasts
5. Create visualizations and reports

### Using Your Own Data

```python
from chargeback_forecast import ChargebackForecaster

# Initialize forecaster
forecaster = ChargebackForecaster()

# Load your data (CSV or Excel)
forecaster.load_data('your_data.csv')

# Preprocess
forecaster.preprocess_data()

# Train models
ml_results = forecaster.train_ml_models(target_col='chargeback_amount')
ts_results = forecaster.train_time_series_models(target_col='chargeback_amount')

# Generate forecasts
forecasts = forecaster.forecast(periods=30)

# Create visualizations
forecaster.plot_forecast(save_path='forecast.png')
forecaster.plot_feature_importance(save_path='importance.png')

# Generate Excel report
forecaster.generate_report(output_path='report.xlsx')
```

### Data Format

Your data should include the following columns:

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Transaction/chargeback date |
| chargeback_amount | float | Total chargeback amount |
| chargeback_count | int | Number of chargebacks |
| transaction_volume | float | Total transaction volume |
| chargebacks_won | int | Number of disputes won |
| chargebacks_lost | int | Number of disputes lost |

Additional columns will be automatically used as features for the ML models.

## 📊 Power BI Dashboard

### Setup Instructions

1. **Open Power BI Desktop**

2. **Import Data**
   - Click "Get Data" → "Excel"
   - Select the generated `chargeback_forecast_report.xlsx`
   - Load all sheets (Historical Data, Forecasts, Feature Importance)

3. **Create Visualizations**

   **Page 1: Overview Dashboard**
   - **KPI Cards**: Total Chargebacks, Win Rate, Loss Rate, Average Amount
   - **Line Chart**: Historical chargeback amounts over time
   - **Column Chart**: Monthly chargeback comparison
   - **Pie Chart**: Chargeback distribution by category/region

   **Page 2: Forecast Dashboard**
   - **Line Chart**: Historical data + forecast predictions (with multiple models)
   - **Area Chart**: Confidence intervals for forecasts
   - **Table**: Detailed forecast values by date
   - **KPI Cards**: Forecast summary metrics

   **Page 3: Analytics Dashboard**
   - **Bar Chart**: Feature importance rankings
   - **Line Chart**: Win/loss ratio trends over time
   - **Scatter Plot**: Correlation analysis
   - **Table**: Model performance metrics

4. **Add Filters**
   - Date range slicer
   - Model type filter
   - Category/Region filters

5. **Save the Dashboard**
   - File → Save As → `Chargeback_Forecasting_Dashboard.pbix`

### Dashboard Features

- **Interactive Filtering**: Filter by date, category, region, or model type
- **Drill-down Capabilities**: Click on charts to drill into details
- **Automatic Refresh**: Connect to live data sources for real-time updates
- **Export Options**: Export visuals and data to Excel or PDF

### Sample Power BI Measures (DAX)

```dax
Total Chargebacks = SUM('Historical Data'[chargeback_amount])

Win Rate = 
DIVIDE(
    SUM('Historical Data'[chargebacks_won]),
    SUM('Historical Data'[chargebacks_won]) + SUM('Historical Data'[chargebacks_lost])
)

Forecast Accuracy = 
1 - ABS(
    DIVIDE(
        [Actual] - [Forecast],
        [Actual]
    )
)

YoY Growth = 
VAR CurrentYear = SUM('Historical Data'[chargeback_amount])
VAR PreviousYear = CALCULATE(
    SUM('Historical Data'[chargeback_amount]),
    SAMEPERIODLASTYEAR('Historical Data'[date])
)
RETURN
DIVIDE(CurrentYear - PreviousYear, PreviousYear)
```

## 📈 Model Performance

The system uses multiple forecasting approaches:

- **Random Forest**: Captures non-linear relationships and feature interactions
- **Gradient Boosting**: Sequential learning for improved accuracy
- **ARIMA**: Time series patterns and autocorrelation
- **Exponential Smoothing**: Trend and seasonality decomposition
- **Ensemble**: Combined predictions for robust forecasting

Performance metrics include:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- AIC/BIC for time series models

## 📁 Project Structure

```
Chargeback_forecasting/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── chargeback_forecast.py            # Main forecasting script
├── sample_data.csv                   # Generated sample data
├── forecast_plot.png                 # Forecast visualization
├── feature_importance.png            # Feature importance chart
├── chargeback_forecast_report.xlsx   # Excel report
└── Chargeback_Forecasting_Dashboard.pbix  # Power BI dashboard (create using instructions above)
```

## 🔍 Key Features Explained

### Win/Loss Analysis
Track the outcome of chargeback disputes to understand:
- Overall win rate percentage
- Trends in dispute outcomes over time
- Impact of win/loss ratio on forecasting

### Feature Engineering
Automatically creates predictive features:
- Time-based features (year, month, quarter, day of week)
- Rolling statistics (7-day and 30-day moving averages)
- Lag features for time series patterns

### Ensemble Forecasting
Combines predictions from multiple models:
- Reduces individual model biases
- Provides more robust predictions
- Offers confidence intervals

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🎯 Future Enhancements

- [ ] Real-time forecasting with streaming data
- [ ] Deep learning models (LSTM, GRU)
- [ ] Automated hyperparameter tuning
- [ ] REST API for model serving
- [ ] Web-based dashboard interface
- [ ] Integration with payment processors
- [ ] Anomaly detection for fraud patterns

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- Statsmodels Documentation: https://www.statsmodels.org/
- Power BI Documentation: https://docs.microsoft.com/power-bi/

---

**Note**: This is a forecasting tool and should be used as part of a comprehensive chargeback management strategy. Always validate predictions with domain expertise and business context.
