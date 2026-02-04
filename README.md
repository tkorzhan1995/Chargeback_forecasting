# Chargeback Data Intake, Reconciliation, and Forecasting System

A comprehensive end-to-end chargeback management system that handles data intake, reconciliation, forecasting, and visualization through an integrated Power BI dashboard.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Power BI Integration](#power-bi-integration)
- [Sample Data](#sample-data)
- [Testing](#testing)
- [Contributing](#contributing)

## Overview

This system provides a complete solution for managing chargebacks with the following capabilities:

1. **Data Intake**: Ingest data from multiple sources (CSV, JSON, API, databases)
2. **Reconciliation**: Match chargebacks to original transactions using advanced algorithms
3. **Forecasting**: Predict future chargeback volumes and identify high-risk transactions
4. **Power BI Integration**: Export data and create visualizations in Power BI

## Features

### Data Intake Module
- Multi-source data ingestion (CSV, JSON, Excel, Parquet, API, databases)
- Comprehensive data validation and cleaning
- Support for batch and real-time processing
- Error handling and logging

### Reconciliation Engine
- **Exact matching** on transaction IDs
- **Fuzzy matching** for incomplete data
- **Amount and time-based matching**
- **Customer-based matching**
- **Partial amount matching** for split transactions
- Configurable confidence thresholds
- Comprehensive reconciliation reports

### Forecasting Models
- **Time series models**: ARIMA, Prophet, Exponential Smoothing
- **Classification models**: Random Forest, Logistic Regression, XGBoost
- **Ensemble methods** for improved accuracy
- Historical rate calculation and trend analysis
- Win/loss ratio tracking
- Key driver identification

### Power BI Integration
- Data export in CSV and Parquet formats
- Pre-aggregated metrics and KPIs
- Python visual scripts for custom visualizations
- Automated data refresh capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) PostgreSQL or SQL Server for database storage
- (Optional) Power BI Desktop for dashboard development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tkorzhan1995/Chargeback_forecasting.git
cd Chargeback_forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration (optional):
```bash
cp config/settings.py config/settings_local.py
# Edit settings_local.py with your configuration
```

## Quick Start

### Generate Sample Data

```python
from sample_data.generate_data import SampleDataGenerator
from config.settings import DATA_DIR

# Generate sample data
generator = SampleDataGenerator()
datasets = generator.generate_complete_dataset(DATA_DIR)
```

### Run Data Ingestion

```python
from data_intake import DataIngestion, DataValidator, DataTransformer

# Initialize components
ingestion = DataIngestion()
validator = DataValidator()
transformer = DataTransformer()

# Ingest data
transactions = ingestion.ingest_csv('sample_data/transactions.csv')
chargebacks = ingestion.ingest_csv('sample_data/chargebacks.csv')

# Validate and transform
transactions = validator.validate_transactions(transactions)
transactions = transformer.transform_transactions(transactions)

chargebacks = validator.validate_chargebacks(chargebacks)
chargebacks = transformer.transform_chargebacks(chargebacks)
```

### Run Reconciliation

```python
from reconciliation import MatchingEngine, ReconciliationReports

# Initialize matching engine
matcher = MatchingEngine()

# Reconcile chargebacks with transactions
matched, unmatched = matcher.reconcile(
    chargebacks=chargebacks,
    transactions=transactions
)

# Generate reports
reporter = ReconciliationReports()
reporter.save_reports(matched, unmatched, len(chargebacks))
```

### Generate Forecasts

```python
from forecasting import ChargebackForecaster, FeatureEngineer

# Initialize forecaster
forecaster = ChargebackForecaster()

# Generate forecast
forecast_results = forecaster.forecast_chargebacks(chargebacks)

# Get best forecast
best_model = forecast_results['best_model']
print(f"Best model: {best_model}")
```

### Export for Power BI

```python
from powerbi_integration import PowerBIExporter, DataAggregator

# Initialize exporter
exporter = PowerBIExporter()

# Export data
exports = exporter.export_chargeback_data(
    chargebacks=chargebacks,
    transactions=transactions,
    matched=matched
)

# Export forecasts
from forecasting import PredictionEngine
pred_engine = PredictionEngine()
predictions = pred_engine.generate_volume_predictions(forecast_results)
exporter.export_forecast_data(predictions)
```

## Project Structure

```
Chargeback_forecasting/
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── settings.py             # Main settings
│   └── database_config.py      # Database configuration
├── data_intake/                 # Data ingestion module
│   ├── __init__.py
│   ├── ingestion.py            # Data ingestion from multiple sources
│   ├── validation.py           # Data validation
│   └── transformations.py      # Data transformations
├── reconciliation/              # Reconciliation module
│   ├── __init__.py
│   ├── matching_engine.py      # Main matching logic
│   ├── linkage_algorithms.py  # Advanced matching algorithms
│   └── reconciliation_reports.py  # Report generation
├── forecasting/                 # Forecasting module
│   ├── __init__.py
│   ├── feature_engineering.py  # Feature creation
│   ├── models.py               # Forecasting models
│   ├── model_evaluation.py     # Model evaluation
│   └── predictions.py          # Prediction generation
├── powerbi_integration/         # Power BI integration
│   ├── __init__.py
│   ├── data_export.py          # Data export utilities
│   ├── aggregations.py         # Data aggregation
│   └── python_visuals.py       # Python visual scripts
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── logging_config.py       # Logging configuration
│   └── helpers.py              # Helper functions
├── sample_data/                 # Sample data
│   └── generate_data.py        # Sample data generator
├── tests/                       # Unit tests
├── dashboard/                   # Power BI dashboard files
├── output/                      # Output directory for results
├── logs/                        # Log files
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Module Documentation

### Data Intake

The data intake module handles ingestion from multiple sources:

- **DataIngestion**: Ingest from CSV, JSON, Excel, Parquet, APIs, and databases
- **DataValidator**: Validate schema, required fields, data types, and ranges
- **DataTransformer**: Clean, normalize, and transform data

### Reconciliation

The reconciliation module matches chargebacks to transactions:

- **MatchingEngine**: Main reconciliation logic with multiple matching strategies
- **LinkageAlgorithms**: Advanced fuzzy matching and probabilistic linkage
- **ReconciliationReports**: Generate comprehensive reports

### Forecasting

The forecasting module predicts future chargebacks:

- **FeatureEngineer**: Create predictive features from raw data
- **ChargebackForecaster**: Time series forecasting models
- **ChargebackClassifier**: Transaction-level risk prediction
- **ModelEvaluator**: Evaluate and monitor model performance
- **PredictionEngine**: Generate and manage predictions

### Power BI Integration

Export data and create visualizations:

- **PowerBIExporter**: Export data in Power BI-compatible formats
- **DataAggregator**: Pre-aggregate data for dashboard performance
- **Python Visuals**: Example scripts for Power BI Python visuals

## Configuration

### Settings

Edit `config/settings.py` to configure:

- Data directories
- Reconciliation thresholds
- Forecasting parameters
- Model settings
- Logging configuration

### Database

Edit `config/database_config.py` to configure database connections:

```python
# Set environment variables
export DB_TYPE=postgresql
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=chargeback_db
export DB_USER=your_username
export DB_PASSWORD=your_password
```

## Power BI Integration

### Data Export

Data is automatically exported in CSV and Parquet formats to the `output/powerbi/` directory.

### Dashboard Setup

1. Open Power BI Desktop
2. Get Data → CSV/Parquet and import the exported files
3. Create relationships between tables:
   - transactions ← chargebacks (transaction_id)
   - transactions ← products (product_id)
   - transactions ← customers (customer_id)
   - transactions ← channels (channel_id)

### Python Visuals

Enable Python visuals in Power BI:
1. File → Options → Python scripting
2. Set Python home directory
3. Use example scripts from `powerbi_integration/python_visuals.py`

### Dashboard Pages

Create the following pages:

**Overview Page:**
- Total chargeback volume and amount cards
- Chargeback rate trending line chart
- Win/loss ratio gauge
- Forecasted chargebacks line chart

**Reconciliation Page:**
- Match rate card
- Matched vs unmatched bar chart
- Data quality metrics table
- Unmatched records table

**Analysis Page:**
- Chargeback by product category (bar chart)
- Chargeback by channel (pie chart)
- Chargeback by reason code (bar chart)
- Product-Channel heatmap

**Forecasting Page:**
- Forecast vs actual line chart
- Confidence intervals area chart
- Feature importance bar chart
- Model performance metrics

## Sample Data

Generate sample data for testing:

```bash
python sample_data/generate_data.py
```

This creates:
- 50 products
- 1,000 customers
- 5 channels
- 10,000 transactions
- ~200 chargebacks (2% rate)

All files are saved to the `sample_data/` directory.

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

## Success Metrics

- ✅ Successfully ingests and validates chargeback and transaction data
- ✅ Achieves >85% automatic matching rate in reconciliation (configurable)
- ✅ Links disputes to transactions, products, customers, and channels accurately
- ✅ Forecasting models with comprehensive evaluation metrics
- ✅ Power BI compatible data exports
- ✅ Modular, documented, and maintainable code

## Technical Stack

- **Python 3.8+**: Primary development language
- **Pandas, NumPy**: Data manipulation
- **Scikit-learn**: Machine learning
- **Statsmodels, Prophet**: Time series forecasting
- **SQLAlchemy**: Database connectivity
- **Power BI Desktop**: Dashboard development
- **Pytest**: Testing framework

## Best Practices

1. **Data Privacy**: Ensure sensitive data is properly secured
2. **Error Handling**: All modules include comprehensive error handling
3. **Logging**: Detailed logging throughout the pipeline
4. **Configuration**: Use environment variables for sensitive settings
5. **Testing**: Write unit tests for all new functionality
6. **Documentation**: Document all functions and modules

## Troubleshooting

### Common Issues

**Issue: Module not found**
```bash
# Ensure you're in the project root and virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Issue: Database connection failed**
```bash
# Check database configuration
echo $DB_HOST
echo $DB_NAME
# Test connection manually
```

**Issue: Power BI can't find Python**
- Verify Python is installed and in PATH
- Set Python home directory in Power BI Options
- Ensure required packages are installed in Python environment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is provided as-is for educational and commercial use.

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review logs in the `logs/` directory

## Roadmap

Future enhancements:
- [ ] Real-time streaming data ingestion
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Automated model retraining pipeline
- [ ] REST API for predictions
- [ ] Web-based dashboard (alternative to Power BI)
- [ ] Multi-currency support
- [ ] Advanced fraud detection algorithms

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-03
