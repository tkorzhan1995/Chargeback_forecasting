"""
Quick Start Example for Chargeback Forecasting

This script demonstrates basic usage of the chargeback forecasting system.
"""

from chargeback_forecast import ChargebackForecaster, generate_sample_data

def main():
    print("=" * 60)
    print("QUICK START: Chargeback Forecasting")
    print("=" * 60)
    
    # Step 1: Generate sample data (or use your own)
    print("\n1. Generating sample data...")
    generate_sample_data(n_records=365, output_path='sample_data.csv')
    
    # Step 2: Initialize the forecaster
    print("\n2. Initializing forecaster...")
    forecaster = ChargebackForecaster()
    
    # Step 3: Load data
    print("\n3. Loading data...")
    forecaster.load_data('sample_data.csv')
    
    # Step 4: Preprocess
    print("\n4. Preprocessing data...")
    forecaster.preprocess_data()
    
    # Step 5: Train models
    print("\n5. Training models (this may take a minute)...")
    forecaster.train_ml_models(target_col='chargeback_amount')
    forecaster.train_time_series_models(target_col='chargeback_amount', periods=30)
    
    # Step 6: Generate forecasts
    print("\n6. Generating 30-day forecast...")
    forecasts = forecaster.forecast(periods=30)
    print("\nFirst 5 forecast days:")
    print(forecasts[['date', 'ensemble']].head())
    
    # Step 7: Win/Loss Analysis
    print("\n7. Analyzing win/loss ratios...")
    forecaster.calculate_win_loss_ratio()
    
    # Step 8: Create visualizations
    print("\n8. Creating visualizations...")
    forecaster.plot_forecast(save_path='quick_start_forecast.png')
    forecaster.plot_feature_importance(top_n=10, save_path='quick_start_features.png')
    
    # Step 9: Generate report
    print("\n9. Generating Excel report...")
    forecaster.generate_report(output_path='quick_start_report.xlsx')
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - quick_start_forecast.png")
    print("  - quick_start_features.png")
    print("  - quick_start_report.xlsx")
    print("\nNext steps:")
    print("  1. Open quick_start_report.xlsx to see detailed results")
    print("  2. Use the Excel file with Power BI (see PowerBI_Dashboard_Guide.md)")
    print("  3. Replace sample data with your own chargeback data")
    

if __name__ == "__main__":
    main()
