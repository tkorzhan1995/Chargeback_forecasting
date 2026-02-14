# Chargeback Forecasting Framework - Structure Summary

This document provides a complete overview of the repository structure, emphasizing enterprise architecture, auditability, and decision support.

## 🏗️ Repository Structure

```
Chargeback_forecasting/
├── data_pipeline/          # Data processing modules
│   ├── data_cleaning.py    # Data validation and cleaning
│   └── feature_engineering.py  # Feature creation and transformation
├── modeling/               # Forecasting models
│   ├── baseline.py         # Simple baseline models
│   ├── ml_models.py        # ML-based models (RF, XGBoost, LightGBM)
│   └── eval_metrics.py     # Evaluation metrics
├── config/                 # Configuration files
│   └── settings.yml        # Central configuration
├── tests/                  # Unit tests
│   └── test_models.py      # Model tests
└── docs/                   # Documentation
    └── architecture.md     # Architecture documentation
```

## 📋 Module Descriptions

### Data Pipeline

#### `data_pipeline/data_cleaning.py`
Handles data quality and preparation:
- **Data Validation**: Checks for required columns and data types
- **Missing Value Handling**: Multiple strategies (mean, median, forward fill, drop)
- **Outlier Detection**: Statistical methods to identify anomalies
- **Duplicate Removal**: Ensures data integrity
- **Date Parsing**: Handles various date formats

**Usage Example:**
```python
from data_pipeline.data_cleaning import DataCleaner

cleaner = DataCleaner(config={'missing_threshold': 0.5})
cleaned_data = cleaner.clean(raw_data)
```

#### `data_pipeline/feature_engineering.py`
Creates features for ML models:
- **Time Features**: Year, month, day, day of week, quarter, weekend indicator
- **Lag Features**: Historical values at specified lags
- **Rolling Statistics**: Moving averages, std dev, min, max
- **Interaction Features**: Combined feature effects

**Usage Example:**
```python
from data_pipeline.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(config={'lag_periods': [1, 7, 30]})
featured_data = engineer.create_features(cleaned_data, date_col='date')
```

### Modeling

#### `modeling/baseline.py`
Simple, interpretable baseline models:
- **Moving Average**: Simple average of recent values
- **Weighted Moving Average**: Recent values weighted more heavily
- **Naive Forecast**: Last observed value
- **Seasonal Naive**: Value from same period last season
- **Trend Forecast**: Linear trend projection

**Usage Example:**
```python
from modeling.baseline import BaselineForecaster

forecaster = BaselineForecaster()
forecast = forecaster.forecast(historical_data, method='moving_average', window=7)
```

#### `modeling/ml_models.py`
Advanced machine learning models:
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast gradient boosting framework
- **Ensemble**: Combines multiple models

**Usage Example:**
```python
from modeling.ml_models import MLForecaster

ml = MLForecaster(config=settings['models']['ml'])
X, y = ml.prepare_data(df, target_col='amount')
models = ml.train_all(X, y)
predictions = ml.ensemble_predict(X_test, method='mean')
```

#### `modeling/eval_metrics.py`
Comprehensive evaluation framework:
- **Standard Metrics**: MAE, RMSE, MAPE, SMAPE, R², MASE
- **Accuracy Metrics**: Percentage within threshold
- **Business Metrics**: Cost estimation, bias analysis
- **Model Comparison**: Ranking and selection tools

**Usage Example:**
```python
from modeling.eval_metrics import MetricsCalculator

metrics = MetricsCalculator()
results = metrics.calculate_all_metrics(y_true, y_pred, y_train)
comparison = metrics.compare_models({'model1': results1, 'model2': results2})
```

### Configuration

#### `config/settings.yml`
Centralized configuration for all components:
- Data pipeline parameters
- Model hyperparameters
- Evaluation thresholds
- Paths and logging settings

**Usage Example:**
```python
import yaml

with open('config/settings.yml', 'r') as f:
    config = yaml.safe_load(f)

cleaner = DataCleaner(config=config['data_pipeline']['cleaning'])
```

## 🔄 Complete Workflow Example

```python
import pandas as pd
import yaml
from data_pipeline.data_cleaning import DataCleaner
from data_pipeline.feature_engineering import FeatureEngineer
from modeling.baseline import BaselineForecaster
from modeling.ml_models import MLForecaster
from modeling.eval_metrics import MetricsCalculator

# 1. Load configuration
with open('config/settings.yml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Load and clean data
raw_data = pd.read_csv('data/raw_chargebacks.csv')
cleaner = DataCleaner(config=config['data_pipeline']['cleaning'])
cleaned_data = cleaner.clean(raw_data)

# 3. Engineer features
engineer = FeatureEngineer(config=config['data_pipeline']['feature_engineering'])
featured_data = engineer.create_features(cleaned_data, date_col='date')

# 4. Split data
train_size = int(len(featured_data) * 0.8)
train_data = featured_data[:train_size]
test_data = featured_data[train_size:]

# 5. Train baseline models
baseline = BaselineForecaster()
baseline_results = baseline.evaluate_baselines(train_data['amount'], test_size=30)

# 6. Train ML models
ml = MLForecaster(config=config['models']['ml'])
X_train, y_train = ml.prepare_data(train_data, target_col='amount')
X_test, y_test = ml.prepare_data(test_data, target_col='amount')
models = ml.train_all(X_train, y_train)

# 7. Generate predictions
predictions = ml.ensemble_predict(X_test, method='mean')

# 8. Evaluate
metrics_calc = MetricsCalculator()
results = metrics_calc.calculate_all_metrics(y_test, predictions, y_train)
business_metrics = metrics_calc.business_metrics(y_test, predictions, cost_per_error=1.0)

# 9. Compare all models
all_results = {
    'baseline_ma': baseline_results.iloc[0].to_dict(),
    'ensemble_ml': results
}
comparison = metrics_calc.compare_models(all_results)
print(comparison)
```

## 🎯 Key Features

### Architecture
- **Modular Design**: Each component is independent and reusable
- **Separation of Concerns**: Data, modeling, and evaluation are distinct
- **Configuration-Driven**: Easy to adjust parameters without code changes

### Auditability
- **Comprehensive Logging**: Track all operations and decisions
- **Metric Tracking**: Multiple evaluation metrics for transparency
- **Version Control**: All changes tracked in Git

### Decision Support
- **Multiple Models**: Compare baseline and ML approaches
- **Business Metrics**: Understand cost implications
- **Feature Importance**: Identify key drivers
- **Model Comparison**: Rank models by performance

### Reliability
- **Error Handling**: Robust validation and error checking
- **Data Quality**: Multiple cleaning and validation steps
- **Unit Tests**: Ensure code correctness

### Interpretability
- **Baseline Models**: Simple, explainable forecasts
- **Feature Engineering**: Understand what drives predictions
- **Metric Explanations**: Clear interpretation guidelines

### Scalability
- **Parallel Processing**: Utilize multiple cores
- **Efficient Algorithms**: Optimized implementations
- **Batch Processing**: Handle large datasets

## 📊 Model Selection Guidelines

1. **Start with Baselines**: Establish performance floor
2. **Compare ML Models**: Identify best performing approach
3. **Consider Business Context**: Balance accuracy with interpretability
4. **Monitor Performance**: Track metrics over time
5. **Update Regularly**: Retrain with new data

## 🔍 Next Steps

1. **Customize Configuration**: Adjust `config/settings.yml` for your data
2. **Add Domain Logic**: Incorporate business rules
3. **Expand Features**: Add domain-specific features
4. **Enhance Models**: Tune hyperparameters
5. **Build Dashboard**: Visualize results and metrics

## 📝 Testing

Run unit tests:
```bash
python -m pytest tests/
```

## 🚀 Deployment Considerations

- **Model Persistence**: Save trained models using pickle or joblib
- **API Development**: Wrap models in REST API
- **Monitoring**: Track prediction accuracy in production
- **Retraining**: Schedule periodic model updates
- **Version Control**: Track model versions and performance

## 📚 Additional Resources

- See `docs/architecture.md` for detailed architecture
- Check `IMPLEMENTATION_SUMMARY.md` for implementation details
- Review `USAGE_EXAMPLES.md` for more examples

---

**Note**: This framework prioritizes enterprise requirements: architecture, auditability, reliability, interpretability, and scalability. It provides a foundation for production-ready chargeback forecasting systems.