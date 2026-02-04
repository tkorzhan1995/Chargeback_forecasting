# Creating a Power BI Dashboard Template (.pbit)

## Overview
This document explains how to create and share a Power BI Template (.pbit) file for the Chargeback Forecasting dashboard.

## What is a .pbit file?
A Power BI Template (.pbit) file is a reusable Power BI report template that contains:
- Report layout and visualizations
- DAX measures and calculations
- Page structure and formatting
- Data source connections (parameterized)

**Note:** .pbit files do NOT contain actual data, making them perfect for sharing and version control.

## Creating the Template

### Step 1: Build Your Dashboard
1. Follow the instructions in `PowerBI_Dashboard_Guide.md`
2. Create all visualizations, measures, and pages
3. Test the dashboard with your data

### Step 2: Save as Template
1. In Power BI Desktop, go to **File → Export → Power BI template**
2. Provide a name: `Chargeback_Forecasting_Template`
3. Add a description:
   ```
   Chargeback Forecasting Dashboard Template
   
   Features:
   - Executive Overview with KPIs
   - Forecast Analysis with multiple models
   - Feature Importance and Analytics
   - Win/Loss Ratio tracking
   
   Data Requirements:
   - Historical Data with columns: date, chargeback_amount, chargeback_count, etc.
   - Forecasts with columns: date, ensemble, random_forest, arima, etc.
   - Feature Importance with columns: feature, importance
   
   Usage:
   1. Generate data using chargeback_forecast.py
   2. Open this template
   3. Point to your chargeback_forecast_report.xlsx
   4. Click Load
   ```
4. Click **OK** to save

### Step 3: Test the Template
1. Close Power BI Desktop
2. Double-click the .pbit file
3. Enter data source parameters when prompted
4. Verify all visualizations load correctly

## Using the Template

### For End Users
1. Double-click `Chargeback_Forecasting_Template.pbit`
2. When prompted, provide the path to your Excel file:
   - File path: `/path/to/chargeback_forecast_report.xlsx`
3. Click **Load**
4. The dashboard will populate with your data

### Updating Data
1. Run `python chargeback_forecast.py` to generate new data
2. In Power BI, click **Refresh** in the Home ribbon
3. The dashboard will update with the latest data

## Alternative: Manual Dashboard Creation

If you prefer not to create a .pbit file, users can:
1. Follow `PowerBI_Dashboard_Guide.md` to build the dashboard from scratch
2. Use the provided DAX measures and visualization specifications
3. Save as .pbix file for personal use

## Parameterizing Data Sources

To make the template flexible, you can parameterize the data source:

### Creating Parameters
1. In Power BI Desktop, go to **Transform Data**
2. Go to **Manage Parameters → New Parameter**
3. Create a parameter:
   - Name: `DataFilePath`
   - Type: Text
   - Current Value: `chargeback_forecast_report.xlsx`
4. Click **OK**

### Using Parameters in Data Source
1. Go to **Transform Data**
2. Right-click on your data source → **Advanced Editor**
3. Replace the file path with: `#"DataFilePath"`
4. Click **Done**

Now when users open the template, they'll be prompted for the file path.

## Sharing the Template

### GitHub Repository
The .pbit file can be added to the repository, but note:
- Keep file size under 100MB (GitHub limit)
- .pbit files are binary and not diff-friendly
- Include version in filename: `Chargeback_Dashboard_v1.0.pbit`

### Documentation
When sharing, include:
1. This README section
2. `PowerBI_Dashboard_Guide.md` for detailed instructions
3. Sample data or link to generate it
4. List of required columns

## Troubleshooting

### Issue: "Can't connect to data source"
**Solution**: Verify the Excel file path is correct and accessible

### Issue: "Couldn't load data"
**Solution**: Ensure the Excel file has the required sheets:
- Historical Data
- Forecasts
- Feature Importance
- Summary Statistics

### Issue: "Measures showing errors"
**Solution**: Check that column names match exactly in DAX measures

### Issue: "Template looks different from guide"
**Solution**: Ensure you're using the latest version of Power BI Desktop

## Best Practices

1. **Version Control**
   - Include version number in template filename
   - Document changes in release notes
   - Keep a changelog

2. **Documentation**
   - Include a "Help" page in the dashboard
   - Add tooltips to visuals explaining what they show
   - Document required data format

3. **Testing**
   - Test template with multiple data sources
   - Verify all interactions work
   - Check performance with large datasets

4. **Maintenance**
   - Update template when adding new features
   - Remove deprecated visuals
   - Keep DAX measures optimized

## Example: Complete Workflow

```bash
# 1. Generate data
python chargeback_forecast.py

# 2. Open Power BI template
# (Double-click Chargeback_Forecasting_Template.pbit)

# 3. Enter data path when prompted
# File: /path/to/chargeback_forecast_report.xlsx

# 4. Dashboard loads automatically

# 5. Explore and analyze

# 6. Save as .pbix for your records
# File → Save As → my_chargeback_dashboard.pbix
```

## Additional Resources

- **Power BI Template Documentation**: https://docs.microsoft.com/power-bi/create-reports/desktop-templates
- **DAX Guide**: https://dax.guide/
- **Power BI Best Practices**: https://docs.microsoft.com/power-bi/guidance/

---

**Note**: Since .pbit files are binary and require Power BI Desktop, this repository focuses on providing comprehensive documentation to build the dashboard from scratch. Users can create their own templates following the guide.
