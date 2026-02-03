# Power BI Dashboard Setup Guide

## Prerequisites

1. **Power BI Desktop**
   - Download and install from [Microsoft Power BI](https://powerbi.microsoft.com/desktop/)
   - Version: Latest stable release

2. **Python Environment**
   - Python 3.8+
   - Required packages installed (see requirements.txt)
   - Python visible in system PATH

3. **Data Files**
   - Generated sample data or production data
   - Exported files in output/powerbi/ directory

## Initial Setup

### 1. Enable Python in Power BI

1. Open Power BI Desktop
2. Go to File → Options and settings → Options
3. Select "Python scripting" from the list
4. Set Python home directory (e.g., C:\Python38\ or path to your virtual environment)
5. Click OK

### 2. Import Data

#### Import from CSV Files

1. Click "Get Data" → "Text/CSV"
2. Navigate to output/powerbi/ directory
3. Import the following files:
   - transactions_YYYYMMDD.csv
   - chargebacks_YYYYMMDD.csv
   - reconciliation_YYYYMMDD.csv
   - products_YYYYMMDD.csv
   - customers_YYYYMMDD.csv
   - channels_YYYYMMDD.csv
   - predictions_YYYYMMDD.csv

#### Import from Parquet Files (Alternative)

1. Click "Get Data" → "More"
2. Search for "Parquet"
3. Navigate to output/powerbi/ directory
4. Select files with .parquet extension

### 3. Create Relationships

1. Go to Model view (left sidebar icon)
2. Create the following relationships:

**Primary Relationships:**
```
transactions.transaction_id (1) → (*) chargebacks.transaction_id
transactions.product_id (*) → (1) products.product_id
transactions.customer_id (*) → (1) customers.customer_id
transactions.channel_id (*) → (1) channels.channel_id
chargebacks.chargeback_id (1) → (1) reconciliation.chargeback_id
```

**Relationship Settings:**
- Cardinality: One-to-many or Many-to-one as specified
- Cross filter direction: Single (or Both where needed)
- Make relationship active: Yes

### 4. Create Date Table

1. Go to "Modeling" tab
2. Click "New Table"
3. Enter DAX formula:
```dax
DateTable = 
ADDCOLUMNS (
    CALENDAR (DATE(2025, 1, 1), DATE(2026, 12, 31)),
    "Year", YEAR([Date]),
    "Month", FORMAT([Date], "MMMM"),
    "MonthNumber", MONTH([Date]),
    "Quarter", "Q" & FORMAT([Date], "Q"),
    "DayOfWeek", FORMAT([Date], "dddd"),
    "DayOfWeekNumber", WEEKDAY([Date])
)
```

4. Mark as date table: Right-click DateTable → Mark as date table

## Dashboard Pages

### Page 1: Overview

**Visuals to Create:**

1. **KPI Cards** (4 cards in top row)
   - Total Chargebacks: `COUNT(chargebacks[chargeback_id])`
   - Total Amount: `SUM(chargebacks[amount])`
   - Match Rate: `DIVIDE(COUNT(reconciliation[chargeback_id]), COUNT(chargebacks[chargeback_id])) * 100`
   - Win Rate: `DIVIDE(COUNTROWS(FILTER(chargebacks, chargebacks[status]="won")), COUNTROWS(chargebacks)) * 100`

2. **Chargeback Trend** (Line Chart)
   - X-Axis: chargebacks[chargeback_date] (by Month)
   - Y-Axis: Count of chargeback_id
   - Legend: None

3. **Forecast Chart** (Line Chart)
   - X-Axis: predictions[date]
   - Y-Axis: predictions[predicted_volume]
   - Add confidence intervals as separate series

4. **Chargeback Rate Gauge**
   - Value: `DIVIDE(COUNT(chargebacks[chargeback_id]), COUNT(transactions[transaction_id])) * 100`
   - Target: 2.0
   - Maximum: 5.0

### Page 2: Reconciliation

**Visuals to Create:**

1. **Match Status Card**
   - Matched: `COUNT(reconciliation[chargeback_id])`
   - Unmatched: `COUNT(chargebacks[chargeback_id]) - COUNT(reconciliation[chargeback_id])`

2. **Match Method Distribution** (Pie Chart)
   - Legend: reconciliation[match_method]
   - Values: Count of chargeback_id

3. **Confidence Score Distribution** (Histogram)
   - X-Axis: reconciliation[match_confidence] (binned)
   - Y-Axis: Count

4. **Unmatched Records Table**
   - Columns: chargeback_id, chargeback_date, amount, reason_code
   - Filter: WHERE chargeback_id NOT IN reconciliation

5. **Data Quality Metrics Table**
   - Custom measures for completeness, accuracy

### Page 3: Analysis

**Visuals to Create:**

1. **Chargebacks by Product Category** (Bar Chart)
   - X-Axis: products[category]
   - Y-Axis: Count of chargebacks
   - Sort: Descending

2. **Chargebacks by Channel** (Pie/Donut Chart)
   - Legend: channels[channel_name]
   - Values: Count of chargebacks

3. **Chargebacks by Reason Code** (Bar Chart)
   - X-Axis: chargebacks[reason_code]
   - Y-Axis: Count
   - Sort: Descending

4. **Product-Channel Heatmap** (Python Visual)
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   pivot = dataset.pivot_table(
       index='product_category',
       columns='channel_name',
       values='chargeback_id',
       aggfunc='count',
       fill_value=0
   )
   
   plt.figure(figsize=(10, 6))
   sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
   plt.title('Chargeback Count by Product and Channel')
   plt.tight_layout()
   plt.show()
   ```

5. **Time-based Analysis** (Line Chart)
   - X-Axis: Date (by Week/Month)
   - Y-Axis: Multiple metrics (Amount, Count, Rate)

### Page 4: Forecasting

**Visuals to Create:**

1. **Forecast vs Actual** (Line Chart with Python)
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Separate historical and forecast
   historical = dataset[dataset['type'] == 'actual']
   forecast = dataset[dataset['type'] == 'forecast']
   
   plt.figure(figsize=(12, 6))
   plt.plot(historical['date'], historical['volume'], label='Actual', linewidth=2)
   plt.plot(forecast['date'], forecast['volume'], label='Forecast', linestyle='--', linewidth=2)
   
   # Add confidence intervals
   plt.fill_between(forecast['date'], forecast['lower_bound'], 
                    forecast['upper_bound'], alpha=0.2)
   
   plt.xlabel('Date')
   plt.ylabel('Chargeback Volume')
   plt.title('Chargeback Forecast')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()
   ```

2. **Model Performance Metrics** (Table/Cards)
   - RMSE, MAE, MAPE
   - R-squared
   - Model name

3. **Prediction Intervals** (Area Chart)
   - X-Axis: predictions[date]
   - Y-Axis: predicted_volume, lower_bound, upper_bound

4. **Feature Importance** (Bar Chart)
   - From model training results
   - Top 10 features

## DAX Measures

Create these custom measures:

```dax
// Chargeback Rate
ChargebackRate = 
DIVIDE(
    COUNT(chargebacks[chargeback_id]),
    COUNT(transactions[transaction_id])
) * 100

// Win Rate
WinRate = 
DIVIDE(
    COUNTROWS(FILTER(chargebacks, chargebacks[status] = "won")),
    COUNTROWS(FILTER(chargebacks, chargebacks[status] IN {"won", "lost"}))
) * 100

// Match Rate
MatchRate = 
DIVIDE(
    COUNT(reconciliation[chargeback_id]),
    COUNT(chargebacks[chargeback_id])
) * 100

// Average Chargeback Amount
AvgChargebackAmount = AVERAGE(chargebacks[amount])

// Total Chargeback Amount
TotalChargebackAmount = SUM(chargebacks[amount])

// Chargebacks This Month
ChargebacksThisMonth = 
CALCULATE(
    COUNT(chargebacks[chargeback_id]),
    DATESMTD(DateTable[Date])
)

// MoM Change
MoMChange = 
VAR CurrentMonth = [TotalChargebackAmount]
VAR PreviousMonth = 
    CALCULATE(
        [TotalChargebackAmount],
        DATEADD(DateTable[Date], -1, MONTH)
    )
RETURN
    DIVIDE(CurrentMonth - PreviousMonth, PreviousMonth) * 100
```

## Filters and Slicers

Add the following slicers to appropriate pages:

1. **Date Range Slicer**
   - Type: Between
   - Field: DateTable[Date]
   - Apply to: All pages

2. **Product Category Slicer**
   - Type: List
   - Field: products[category]

3. **Channel Slicer**
   - Type: Dropdown
   - Field: channels[channel_name]

4. **Status Slicer** (Reconciliation page)
   - Type: List
   - Field: chargebacks[status]

## Data Refresh

### Manual Refresh

1. Click "Refresh" button in Home tab
2. Data will reload from source files

### Scheduled Refresh (Power BI Service)

1. Publish report to Power BI Service
2. Go to dataset settings
3. Configure scheduled refresh:
   - Frequency: Daily
   - Time: Early morning (e.g., 6:00 AM)
4. Configure gateway if using on-premises data

### Incremental Refresh

For large datasets:

1. Right-click table → Incremental refresh
2. Set parameters:
   - Archive data older than: 2 years
   - Refresh data in the last: 7 days
3. Apply policy

## Formatting and Design

### Theme

Apply consistent colors:
- Primary: #0078D4 (Blue)
- Secondary: #106EBE (Dark Blue)
- Accent: #D83B01 (Orange for alerts)
- Background: #F3F2F1 (Light Gray)

### Fonts

- Headers: Segoe UI, 14pt, Bold
- Body: Segoe UI, 11pt, Regular
- Labels: Segoe UI, 9pt, Regular

### Page Layout

- Use consistent padding: 10px
- Align visuals to grid
- Group related visuals
- Use white space effectively

## Best Practices

1. **Performance**
   - Use DirectQuery for real-time data
   - Import mode for better performance with smaller datasets
   - Create aggregations for large tables

2. **Usability**
   - Add tooltips to explain metrics
   - Use bookmarks for different views
   - Add drill-through pages for details

3. **Maintenance**
   - Document data sources
   - Version control .pbix files
   - Test refresh before deployment

## Troubleshooting

### Python Visual Not Working

1. Check Python is installed and in PATH
2. Verify Python home directory in Options
3. Install required packages: `pip install pandas matplotlib seaborn`
4. Check Python console output for errors

### Data Not Refreshing

1. Verify source files exist and are accessible
2. Check file paths are correct
3. Ensure no file locks exist
4. Review refresh history in Power BI Service

### Slow Performance

1. Reduce number of visuals per page
2. Use aggregated data where possible
3. Filter data at source
4. Use star schema design
5. Disable auto date/time tables

## Additional Resources

- [Power BI Documentation](https://docs.microsoft.com/power-bi/)
- [DAX Guide](https://dax.guide/)
- [Python Visuals in Power BI](https://docs.microsoft.com/power-bi/connect-data/desktop-python-visuals)

## Support

For issues with this dashboard:
1. Check logs in logs/ directory
2. Review data quality reports
3. Contact system administrator
