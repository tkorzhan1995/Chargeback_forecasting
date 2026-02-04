"""
Data export module for Power BI integration.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from utils import get_logger
from config.settings import OUTPUT_DIR, POWERBI_CONFIG

logger = get_logger(__name__)


class PowerBIExporter:
    """
    Export data in formats compatible with Power BI.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize PowerBIExporter.
        
        Args:
            output_dir: Directory to save exports
        """
        self.output_dir = output_dir or OUTPUT_DIR / 'powerbi'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.export_formats = POWERBI_CONFIG['export_formats']
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Export dataframe to CSV.
        
        Args:
            df: Dataframe to export
            filename: Output filename
            
        Returns:
            Path: Output file path
        """
        logger.info(f"Exporting to CSV: {filename}")
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV export complete: {output_path}")
        return output_path
    
    def export_to_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Export dataframe to Parquet format.
        
        Args:
            df: Dataframe to export
            filename: Output filename
            
        Returns:
            Path: Output file path
        """
        logger.info(f"Exporting to Parquet: {filename}")
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Parquet export complete: {output_path}")
        return output_path
    
    def export_dataframe(self, df: pd.DataFrame, base_filename: str) -> List[Path]:
        """
        Export dataframe in all configured formats.
        
        Args:
            df: Dataframe to export
            base_filename: Base filename (without extension)
            
        Returns:
            List[Path]: List of exported file paths
        """
        logger.info(f"Exporting dataframe: {base_filename}")
        exported_files = []
        
        if 'csv' in self.export_formats:
            csv_path = self.export_to_csv(df, f"{base_filename}.csv")
            exported_files.append(csv_path)
        
        if 'parquet' in self.export_formats:
            parquet_path = self.export_to_parquet(df, f"{base_filename}.parquet")
            exported_files.append(parquet_path)
        
        return exported_files
    
    def export_chargeback_data(self, chargebacks: pd.DataFrame,
                              transactions: pd.DataFrame,
                              matched: pd.DataFrame) -> Dict[str, List[Path]]:
        """
        Export all chargeback-related data for Power BI.
        
        Args:
            chargebacks: Chargebacks dataframe
            transactions: Transactions dataframe
            matched: Matched/reconciled records
            
        Returns:
            Dict: Dictionary of exported file paths by dataset name
        """
        logger.info("Exporting chargeback data for Power BI")
        
        timestamp = datetime.now().strftime('%Y%m%d')
        exports = {}
        
        # Export chargebacks
        exports['chargebacks'] = self.export_dataframe(
            chargebacks, f'chargebacks_{timestamp}'
        )
        
        # Export transactions
        exports['transactions'] = self.export_dataframe(
            transactions, f'transactions_{timestamp}'
        )
        
        # Export matched records
        exports['reconciliation'] = self.export_dataframe(
            matched, f'reconciliation_{timestamp}'
        )
        
        logger.info("All chargeback data exported")
        return exports
    
    def export_forecast_data(self, predictions: pd.DataFrame,
                            historical: pd.DataFrame = None) -> Dict[str, List[Path]]:
        """
        Export forecast data for Power BI.
        
        Args:
            predictions: Predictions dataframe
            historical: Historical data (optional)
            
        Returns:
            Dict: Dictionary of exported file paths
        """
        logger.info("Exporting forecast data for Power BI")
        
        timestamp = datetime.now().strftime('%Y%m%d')
        exports = {}
        
        # Export predictions
        exports['predictions'] = self.export_dataframe(
            predictions, f'predictions_{timestamp}'
        )
        
        # Export historical if provided
        if historical is not None:
            exports['historical'] = self.export_dataframe(
                historical, f'historical_{timestamp}'
            )
        
        logger.info("Forecast data exported")
        return exports
    
    def export_aggregated_metrics(self, metrics: Dict[str, Any]) -> Path:
        """
        Export aggregated metrics as CSV.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Path: Output file path
        """
        logger.info("Exporting aggregated metrics")
        
        # Convert metrics to dataframe
        metrics_list = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_list.append({'metric_name': key, 'metric_value': value})
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        metrics_list.append({
                            'metric_name': f'{key}_{subkey}',
                            'metric_value': subvalue
                        })
        
        metrics_df = pd.DataFrame(metrics_list)
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = self.export_to_csv(metrics_df, f'metrics_{timestamp}.csv')
        
        logger.info("Metrics exported")
        return output_path
    
    def create_powerbi_dataset_manifest(self, exports: Dict[str, List[Path]]) -> Path:
        """
        Create a manifest file listing all exported datasets.
        
        Args:
            exports: Dictionary of exported file paths
            
        Returns:
            Path: Manifest file path
        """
        logger.info("Creating Power BI dataset manifest")
        
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for dataset_name, file_paths in exports.items():
            manifest['datasets'][dataset_name] = [str(p) for p in file_paths]
        
        manifest_path = self.output_dir / 'dataset_manifest.json'
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest created: {manifest_path}")
        return manifest_path
