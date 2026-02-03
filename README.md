# Chargeback Forecasting System

A comprehensive Python-based system for forecasting chargebacks based on historical rates, win/loss ratios, and key drivers.

## Features

- **Historical Rate Analysis**: Calculate and analyze chargeback rates over time (monthly, quarterly)
- **Win/Loss Ratio Analysis**: Track and analyze dispute resolution outcomes
- **Key Driver Identification**: Identify factors that influence chargeback rates
- **Multiple Forecasting Methods**: Support for simple average, weighted average, and trend-based forecasting
- **Confidence Intervals**: Provide uncertainty estimates with 95% confidence intervals
- **Flexible Data Import**: Support for JSON and CSV data formats
- **Export Capabilities**: Export forecasts to CSV or JSON for further analysis

## Installation

### Requirements

- Python 3.8 or higher
- No external dependencies required (uses Python standard library only)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tkorzhan1995/Chargeback_forecasting.git
cd Chargeback_forecasting
```

2. (Optional) Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. (Optional) Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the forecasting system with sample data:

```bash
python3 main.py
```

This will:
1. Load sample chargeback and transaction data
2. Perform historical analysis
3. Generate forecasts for the next 3 periods
4. Display results in the console

## Usage

### Basic Usage

```bash
python3 main.py --chargebacks data/sample/chargebacks.json \
                --transactions data/sample/transactions.json \
                --forecast-periods 3
```

### Advanced Options

```bash
python3 main.py \
    --chargebacks path/to/chargebacks.json \
    --transactions path/to/transactions.json \
    --forecast-periods 6 \
    --expected-transactions 1000 \
    --method weighted_average \
    --output-csv forecast_results.csv \
    --output-json forecast_results.json
```

### Command Line Options

- `--chargebacks`: Path to chargebacks data file (JSON format)
- `--transactions`: Path to transactions data file (JSON format)
- `--forecast-periods`: Number of periods to forecast (default: 3)
- `--expected-transactions`: Expected transactions per period (default: 600)
- `--method`: Forecasting method - `simple_average`, `weighted_average`, or `trend` (default: weighted_average)
- `--output-csv`: Export forecast results to CSV file
- `--output-json`: Export forecast results to JSON file

## Data Format

### Chargebacks Data Format (JSON)

```json
[
  {
    "transaction_id": "TXN001",
    "amount": 150.50,
    "date": "2023-01-15",
    "reason_code": "FRAUD",
    "merchant_id": "MERCH_001",
    "status": "won",
    "resolved_date": "2023-02-01",
    "resolution_amount": 150.50
  }
]
```

### Transactions Data Format (JSON)

```json
[
  {
    "transaction_id": "TXN000001",
    "amount": 250.00,
    "date": "2023-01-01",
    "merchant_id": "MERCH_001",
    "category": "electronics"
  }
]
```

## Architecture

The system is organized into the following modules:

### Core Modules

- **models**: Data models for chargebacks, historical rates, win/loss ratios, and key drivers
- **analysis**: Analysis modules for historical rates, win/loss ratios, and key drivers
- **forecasting**: Main forecasting engine with multiple forecasting methods
- **utils**: Utility functions for data loading, processing, and export

### Key Components

1. **HistoricalRateCalculator**: Analyzes historical chargeback rates
2. **WinLossAnalyzer**: Tracks dispute resolution outcomes
3. **KeyDriverAnalyzer**: Identifies factors affecting chargeback rates
4. **ChargebackForecaster**: Main forecasting engine that combines all analyses

## Forecasting Methods

### Simple Average
Uses the simple average of all historical rates to predict future rates.

### Weighted Average (Default)
Gives more weight to recent periods, making the forecast more responsive to recent trends.

### Trend-Based
Uses linear regression to identify trends in historical data and project them forward.

## Example Output

```
======================================================================
HISTORICAL ANALYSIS
======================================================================

Monthly Chargeback Rates:
Period          Transactions    Chargebacks     Rate      
----------------------------------------------------------------------
2023-01         620             2               0.32%
2023-02         580             2               0.34%
2023-03         610             2               0.33%

Average Chargeback Rate: 0.3319%

Win/Loss Ratios:
Period          Total      Won        Lost       Win Rate  
----------------------------------------------------------------------
2023-01         2          1          1          50.00%
2023-02         2          1          1          50.00%
2023-03         2          1          1          50.00%

Overall Win Rate: 58.33%
Overall Loss Rate: 41.67%

Top 5 Key Drivers:
Driver                         Type            Impact          Correlation    
----------------------------------------------------------------------
Reason: FRAUD                  categorical     0.3333          0.3333
Reason: NOT_RECEIVED           categorical     0.2500          0.2500
Reason: DEFECTIVE              categorical     0.2500          0.2500

======================================================================
FORECAST RESULTS
======================================================================

--- Forecast #1 ---

Period:                        2023-07
Expected Chargebacks:          2.01
Expected Rate:                 0.3351%
95% Confidence Interval:       1.93 - 2.09
Expected Win Rate:             58.33%
Expected Loss Rate:            41.67%
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
