# Power BI Dashboard Setup Guide for Chargeback Forecasting

## Overview
This guide provides step-by-step instructions for creating a comprehensive Power BI dashboard for chargeback forecasting visualization and analysis.

## Prerequisites
- Power BI Desktop (Download from: https://powerbi.microsoft.com/desktop/)
- Excel report generated from chargeback_forecast.py (`chargeback_forecast_report.xlsx`)

## Data Import

### Step 1: Import Excel Data
1. Open Power BI Desktop
2. Click **Get Data** → **Excel**
3. Browse to `chargeback_forecast_report.xlsx`
4. Select the following sheets:
   - ✅ Historical Data
   - ✅ Forecasts
   - ✅ Feature Importance
   - ✅ Summary Statistics
5. Click **Load**

### Step 2: Data Relationships
Power BI should automatically detect relationships. Verify:
- Historical Data and Forecasts are linked by date (if applicable)

## Dashboard Pages

### Page 1: Executive Overview

**Purpose**: High-level KPIs and trends for executive decision-making

**Visualizations**:

1. **KPI Cards** (Top row)
   - Total Chargebacks
     - Visual: Card
     - Value: SUM(Historical Data[chargeback_amount])
     - Format: Currency, $
   
   - Win Rate
     - Visual: Card
     - Value: SUM(chargebacks_won) / (SUM(chargebacks_won) + SUM(chargebacks_lost))
     - Format: Percentage, 2 decimals
   
   - Average Chargeback
     - Visual: Card
     - Value: AVERAGE(Historical Data[chargeback_amount])
     - Format: Currency, $
   
   - Total Count
     - Visual: Card
     - Value: SUM(Historical Data[chargeback_count])

2. **Line Chart: Chargeback Trend**
   - X-axis: date (from Historical Data)
   - Y-axis: chargeback_amount
   - Legend: None
   - Title: "Chargeback Amount Over Time"

3. **Column Chart: Monthly Chargebacks**
   - X-axis: Month-Year (date hierarchy)
   - Y-axis: SUM(chargeback_amount)
   - Title: "Monthly Chargeback Totals"

4. **Donut Chart: Distribution by Category**
   - Legend: merchant_category
   - Values: SUM(chargeback_amount)
   - Title: "Chargebacks by Merchant Category"

5. **Map Visual: Regional Distribution**
   - Location: region
   - Size: SUM(chargeback_amount)
   - Title: "Geographic Distribution"

**Filters** (Page level):
- Date range slicer
- Category filter
- Region filter

---

### Page 2: Forecast Analysis

**Purpose**: Detailed forecast visualization and model comparison

**Visualizations**:

1. **Line Chart: Historical vs Forecast**
   - X-axis: date (from both tables)
   - Y-axis: 
     - Historical Data[chargeback_amount] (Solid line)
     - Forecasts[ensemble] (Dashed line)
     - Forecasts[random_forest] (Optional)
     - Forecasts[arima] (Optional)
   - Legend: Model Type
   - Title: "Historical Data and Forecasts"

2. **Area Chart: Forecast Confidence**
   - X-axis: Forecasts[date]
   - Y-axis: 
     - Upper bound: ensemble * 1.1
     - Lower bound: ensemble * 0.9
     - Ensemble (Line overlay)
   - Title: "Forecast with Confidence Interval"

3. **Table: Forecast Details**
   - Columns:
     - date
     - ensemble (Predicted Amount)
     - random_forest
     - gradient_boosting (if available)
     - arima
   - Title: "Detailed Forecast Values"
   - Sort by: date (ascending)

4. **Clustered Column Chart: Model Comparison**
   - X-axis: First 7 days of forecast
   - Y-axis: Values from each model
   - Legend: Model names
   - Title: "7-Day Model Comparison"

**Filters** (Page level):
- Date range for forecast
- Model selection (multi-select)

---

### Page 3: Feature Analytics

**Purpose**: Understanding key drivers and feature importance

**Visualizations**:

1. **Horizontal Bar Chart: Top Features**
   - Y-axis: Feature Importance[feature]
   - X-axis: Feature Importance[importance]
   - Sort: Descending by importance
   - Show top 10
   - Title: "Top 10 Predictive Features"

2. **Line Chart: Win/Loss Trends**
   - X-axis: date
   - Y-axis: 
     - chargebacks_won (Line 1)
     - chargebacks_lost (Line 2)
   - Legend: Outcome type
   - Title: "Win/Loss Trends Over Time"

3. **Gauge: Current Win Rate**
   - Value: Current win rate
   - Minimum: 0%
   - Maximum: 100%
   - Target: 70%
   - Title: "Current Win Rate"

4. **Scatter Plot: Correlation Analysis**
   - X-axis: transaction_volume
   - Y-axis: chargeback_amount
   - Details: date
   - Size: chargeback_count
   - Title: "Transaction Volume vs Chargeback Amount"

5. **Table: Summary Statistics**
   - Data: Summary Statistics sheet
   - All columns visible
   - Title: "Statistical Summary"

**Filters** (Page level):
- Date range
- Top N features selector

---

### Page 4: Performance Metrics

**Purpose**: Model performance and accuracy tracking

**Visualizations**:

1. **Card Visuals: Model Metrics**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score
   - Format appropriately

2. **Waterfall Chart: Forecast Breakdown**
   - Categories: Components of forecast
   - Values: Contribution to total
   - Title: "Forecast Components"

3. **Line and Clustered Column Chart: Accuracy Over Time**
   - X-axis: date
   - Column: Actual values
   - Line: Predicted values
   - Title: "Actual vs Predicted"

---

## Custom DAX Measures

Create these measures in Power BI:

```dax
// Total Chargebacks
Total Chargebacks = SUM('Historical Data'[chargeback_amount])

// Win Rate
Win Rate = 
DIVIDE(
    SUM('Historical Data'[chargebacks_won]),
    SUM('Historical Data'[chargebacks_won]) + SUM('Historical Data'[chargebacks_lost]),
    0
)

// Loss Rate
Loss Rate = 1 - [Win Rate]

// Average Chargeback
Average Chargeback = AVERAGE('Historical Data'[chargeback_amount])

// Month-over-Month Growth
MoM Growth = 
VAR CurrentMonth = [Total Chargebacks]
VAR PreviousMonth = CALCULATE(
    [Total Chargebacks],
    DATEADD('Historical Data'[date], -1, MONTH)
)
RETURN
DIVIDE(CurrentMonth - PreviousMonth, PreviousMonth, 0)

// Year-over-Year Growth
YoY Growth = 
VAR CurrentYear = [Total Chargebacks]
VAR PreviousYear = CALCULATE(
    [Total Chargebacks],
    SAMEPERIODLASTYEAR('Historical Data'[date])
)
RETURN
DIVIDE(CurrentYear - PreviousYear, PreviousYear, 0)

// Forecast Accuracy (when actual data becomes available)
Forecast Accuracy = 
VAR Actual = SUM('Historical Data'[chargeback_amount])
VAR Forecast = SUM('Forecasts'[ensemble])
RETURN
1 - ABS(DIVIDE(Actual - Forecast, Actual, 0))

// Rolling 7-Day Average
Rolling 7 Day Avg = 
AVERAGEX(
    DATESINPERIOD('Historical Data'[date], LASTDATE('Historical Data'[date]), -7, DAY),
    [Total Chargebacks]
)

// Rolling 30-Day Average
Rolling 30 Day Avg = 
AVERAGEX(
    DATESINPERIOD('Historical Data'[date], LASTDATE('Historical Data'[date]), -30, DAY),
    [Total Chargebacks]
)

// Chargeback Count
Total Count = SUM('Historical Data'[chargeback_count])

// Average Transaction Amount
Avg Transaction = AVERAGE('Historical Data'[avg_transaction_amount])
```

## Formatting and Design

### Color Scheme
Use consistent colors:
- **Primary**: #1f77b4 (Blue) - Historical data
- **Secondary**: #ff7f0e (Orange) - Forecasts
- **Accent 1**: #2ca02c (Green) - Positive metrics
- **Accent 2**: #d62728 (Red) - Negative metrics
- **Neutral**: #7f7f7f (Gray) - Supporting data

### Typography
- **Title**: Segoe UI, 16pt, Bold
- **Labels**: Segoe UI, 11pt, Regular
- **Values**: Segoe UI, 14pt, Semibold

### Layout Tips
1. Use consistent spacing (10-15px margins)
2. Align visuals to a grid
3. Group related visuals together
4. Use white space effectively
5. Add subtle borders to separate sections

## Interactive Features

### Slicers
Add these slicers for interactivity:
1. **Date Range**: Use "Between" style slicer
2. **Category**: List style with multi-select
3. **Region**: Dropdown with multi-select
4. **Model Type**: Buttons for easy selection

### Drill-Through
Configure drill-through from:
- Executive Overview → Forecast Analysis (on date)
- Forecast Analysis → Feature Analytics (on prediction)

### Bookmarks
Create bookmarks for:
1. Default View
2. Historical Only View
3. Forecast Only View
4. Comparison View

### Tooltips
Enable custom tooltips showing:
- Date details
- Win/loss information
- Model confidence
- Related metrics

## Data Refresh

### Manual Refresh
1. Click **Refresh** in the Home ribbon
2. Power BI will reload data from the Excel file

### Scheduled Refresh (Power BI Service)
1. Publish the dashboard to Power BI Service
2. Configure data source credentials
3. Set up refresh schedule (e.g., daily at 6 AM)

## Publishing

### Publish to Power BI Service
1. Click **Publish** in the Home ribbon
2. Select your workspace
3. Share with stakeholders
4. Configure row-level security if needed

### Export Options
- PDF: File → Export to PDF
- PowerPoint: File → Export to PowerPoint
- Excel: Right-click visual → Export data

## Best Practices

1. **Performance Optimization**
   - Minimize calculated columns
   - Use measures instead of calculated columns when possible
   - Reduce data granularity if dataset is large
   - Remove unnecessary columns

2. **User Experience**
   - Keep dashboards simple and focused
   - Limit to 3-5 key visuals per page
   - Use consistent color coding
   - Provide clear titles and labels

3. **Maintenance**
   - Document all custom measures
   - Version control your .pbix file
   - Test with updated data regularly
   - Gather user feedback and iterate

## Troubleshooting

### Common Issues

**Issue**: Date relationships not working
- **Solution**: Ensure date formats are consistent across tables

**Issue**: Measures showing incorrect values
- **Solution**: Check DAX syntax and context filters

**Issue**: Visuals loading slowly
- **Solution**: Reduce data volume or optimize DAX measures

**Issue**: Can't see forecast data
- **Solution**: Verify the Forecasts sheet was imported correctly

## Additional Resources

- Power BI Documentation: https://docs.microsoft.com/power-bi/
- DAX Guide: https://dax.guide/
- Power BI Community: https://community.powerbi.com/
- Video Tutorials: https://www.youtube.com/powerbi

## Support

For issues specific to this dashboard:
1. Check the data source (chargeback_forecast_report.xlsx)
2. Verify Python script completed successfully
3. Review error messages in Power BI
4. Open an issue on the GitHub repository

---

**Last Updated**: 2024
**Version**: 1.0
