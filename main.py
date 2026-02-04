"""
Main execution script for the Chargeback Management System.
Demonstrates end-to-end workflow.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_intake import DataIngestion, DataValidator, DataTransformer
from reconciliation import MatchingEngine, ReconciliationReports
from forecasting import ChargebackForecaster, FeatureEngineer, PredictionEngine
from powerbi_integration import PowerBIExporter, DataAggregator
from sample_data.generate_data import SampleDataGenerator
from utils import setup_logging, get_logger
from config.settings import DATA_DIR

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Starting Chargeback Management System")
    logger.info("=" * 60)
    
    # Step 1: Generate Sample Data
    logger.info("\n[1/6] Generating sample data...")
    generator = SampleDataGenerator()
    datasets = generator.generate_complete_dataset(DATA_DIR)
    logger.info(f"✓ Generated {len(datasets)} datasets")
    
    # Step 2: Data Intake and Validation
    logger.info("\n[2/6] Ingesting and validating data...")
    ingestion = DataIngestion()
    validator = DataValidator()
    transformer = DataTransformer()
    
    # Load data
    transactions = datasets['transactions']
    chargebacks = datasets['chargebacks']
    products = datasets['products']
    customers = datasets['customers']
    channels = datasets['channels']
    
    # Validate
    transactions = validator.validate_transactions(transactions)
    chargebacks = validator.validate_chargebacks(chargebacks)
    
    # Transform
    transactions = transformer.transform_transactions(transactions)
    chargebacks = transformer.transform_chargebacks(chargebacks)
    
    logger.info(f"✓ Processed {len(transactions)} transactions")
    logger.info(f"✓ Processed {len(chargebacks)} chargebacks")
    
    # Step 3: Reconciliation
    logger.info("\n[3/6] Running reconciliation...")
    matcher = MatchingEngine()
    
    matched, unmatched = matcher.reconcile(
        chargebacks=chargebacks,
        transactions=transactions,
        products=products,
        customers=customers,
        channels=channels
    )
    
    match_rate = len(matched) / len(chargebacks) * 100 if len(chargebacks) > 0 else 0
    logger.info(f"✓ Match rate: {match_rate:.2f}% ({len(matched)}/{len(chargebacks)})")
    
    # Generate reconciliation reports
    reporter = ReconciliationReports()
    reporter.save_reports(
        matched=matched,
        unmatched=unmatched,
        total_chargebacks=len(chargebacks),
        chargebacks=chargebacks,
        transactions=transactions
    )
    logger.info("✓ Reconciliation reports generated")
    
    # Step 4: Feature Engineering
    logger.info("\n[4/6] Engineering features...")
    engineer = FeatureEngineer()
    
    # Create features for transactions
    transactions_with_features = engineer.build_feature_set(
        transactions=transactions,
        chargebacks=chargebacks,
        include_lag=False,  # Skip lag features for demo
        include_rolling=False  # Skip rolling features for demo
    )
    
    logger.info(f"✓ Created {len(transactions_with_features.columns)} features")
    
    # Step 5: Forecasting
    logger.info("\n[5/6] Generating forecasts...")
    forecaster = ChargebackForecaster()
    
    try:
        # Generate volume forecast
        forecast_results = forecaster.forecast_chargebacks(
            df=chargebacks,
            date_col='chargeback_date',
            models=['moving_average', 'exp_smoothing']  # Use simpler models for demo
        )
        
        best_model = forecast_results.get('best_model', 'Ensemble')
        logger.info(f"✓ Forecasts generated using {best_model} model")
        
        # Generate predictions
        pred_engine = PredictionEngine()
        predictions = pred_engine.generate_volume_predictions(forecast_results)
        pred_engine.save_predictions(predictions)
        logger.info(f"✓ Generated {len(predictions)} prediction records")
        
    except Exception as e:
        logger.warning(f"Forecasting encountered an error: {str(e)}")
        logger.warning("Continuing with remaining steps...")
        predictions = None
    
    # Step 6: Export for Power BI
    logger.info("\n[6/6] Exporting data for Power BI...")
    exporter = PowerBIExporter()
    
    # Export main datasets
    exports = exporter.export_chargeback_data(
        chargebacks=chargebacks,
        transactions=transactions,
        matched=matched
    )
    
    # Export additional reference data
    exporter.export_dataframe(products, 'products')
    exporter.export_dataframe(customers, 'customers')
    exporter.export_dataframe(channels, 'channels')
    
    # Export predictions if available
    if predictions is not None:
        exporter.export_forecast_data(predictions)
    
    # Create aggregated metrics
    aggregator = DataAggregator()
    kpis = aggregator.create_kpi_summary(
        chargebacks=chargebacks,
        transactions=transactions,
        matched=matched
    )
    exporter.export_aggregated_metrics(kpis)
    
    # Create manifest
    exporter.create_powerbi_dataset_manifest(exports)
    
    logger.info("✓ All data exported for Power BI")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Transactions processed: {len(transactions):,}")
    logger.info(f"Chargebacks processed: {len(chargebacks):,}")
    logger.info(f"Match rate: {match_rate:.2f}%")
    logger.info(f"Unmatched chargebacks: {len(unmatched)}")
    logger.info(f"Chargeback rate: {len(chargebacks)/len(transactions)*100:.2f}%")
    
    if kpis:
        logger.info(f"\nKey Metrics:")
        logger.info(f"  Total chargeback amount: ${kpis.get('total_chargeback_amount', 0):,.2f}")
        logger.info(f"  Average chargeback amount: ${kpis.get('avg_chargeback_amount', 0):,.2f}")
    
    logger.info("\n✓ All tasks completed successfully!")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
