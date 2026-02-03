"""
Data ingestion module for handling multiple data sources.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Union, Dict, Any, List
import requests
from utils import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Handle data ingestion from multiple sources including CSV, JSON, API, and databases.
    """
    
    def __init__(self):
        """Initialize DataIngestion instance."""
        self.supported_formats = ['csv', 'json', 'excel', 'parquet']
    
    def ingest_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Ingest data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Ingested data
        """
        logger.info(f"Ingesting CSV file: {file_path}")
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully ingested {len(df)} rows from CSV")
            return df
        except Exception as e:
            logger.error(f"Error ingesting CSV: {str(e)}")
            raise
    
    def ingest_json(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Ingest data from JSON file.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json
            
        Returns:
            pd.DataFrame: Ingested data
        """
        logger.info(f"Ingesting JSON file: {file_path}")
        try:
            df = pd.read_json(file_path, **kwargs)
            logger.info(f"Successfully ingested {len(df)} rows from JSON")
            return df
        except Exception as e:
            logger.error(f"Error ingesting JSON: {str(e)}")
            raise
    
    def ingest_excel(self, file_path: Union[str, Path], sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Ingest data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to read
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            pd.DataFrame: Ingested data
        """
        logger.info(f"Ingesting Excel file: {file_path}, sheet: {sheet_name}")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully ingested {len(df)} rows from Excel")
            return df
        except Exception as e:
            logger.error(f"Error ingesting Excel: {str(e)}")
            raise
    
    def ingest_parquet(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Ingest data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            pd.DataFrame: Ingested data
        """
        logger.info(f"Ingesting Parquet file: {file_path}")
        try:
            df = pd.read_parquet(file_path, **kwargs)
            logger.info(f"Successfully ingested {len(df)} rows from Parquet")
            return df
        except Exception as e:
            logger.error(f"Error ingesting Parquet: {str(e)}")
            raise
    
    def ingest_from_api(self, url: str, method: str = 'GET', 
                       headers: Dict = None, params: Dict = None,
                       json_path: str = None) -> pd.DataFrame:
        """
        Ingest data from REST API.
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST)
            headers: Request headers
            params: Request parameters
            json_path: Path to data in JSON response (e.g., 'data.items')
            
        Returns:
            pd.DataFrame: Ingested data
        """
        logger.info(f"Ingesting data from API: {url}")
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            data = response.json()
            
            # Navigate to nested data if json_path provided
            if json_path:
                for key in json_path.split('.'):
                    data = data[key]
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully ingested {len(df)} rows from API")
            return df
        except Exception as e:
            logger.error(f"Error ingesting from API: {str(e)}")
            raise
    
    def ingest_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Ingest data from database using SQL query.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            
        Returns:
            pd.DataFrame: Ingested data
        """
        logger.info("Ingesting data from database")
        try:
            from sqlalchemy import create_engine
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Successfully ingested {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Error ingesting from database: {str(e)}")
            raise
    
    def ingest_batch(self, file_paths: List[Union[str, Path]], 
                    file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """
        Ingest multiple files in batch and combine them.
        
        Args:
            file_paths: List of file paths
            file_type: Type of files (csv, json, excel, parquet)
            **kwargs: Additional arguments for ingestion method
            
        Returns:
            pd.DataFrame: Combined data from all files
        """
        logger.info(f"Batch ingesting {len(file_paths)} files")
        dfs = []
        
        ingest_method = {
            'csv': self.ingest_csv,
            'json': self.ingest_json,
            'excel': self.ingest_excel,
            'parquet': self.ingest_parquet,
        }.get(file_type)
        
        if not ingest_method:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        for file_path in file_paths:
            try:
                df = ingest_method(file_path, **kwargs)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to ingest {file_path}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No files were successfully ingested")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Successfully combined {len(combined_df)} total rows")
        return combined_df
    
    def auto_ingest(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Automatically detect format and ingest data.
        
        Args:
            source: File path or URL
            **kwargs: Additional arguments
            
        Returns:
            pd.DataFrame: Ingested data
        """
        source_str = str(source)
        
        if source_str.startswith('http://') or source_str.startswith('https://'):
            return self.ingest_from_api(source_str, **kwargs)
        
        path = Path(source)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return self.ingest_csv(path, **kwargs)
        elif suffix == '.json':
            return self.ingest_json(path, **kwargs)
        elif suffix in ['.xls', '.xlsx']:
            return self.ingest_excel(path, **kwargs)
        elif suffix == '.parquet':
            return self.ingest_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unable to auto-detect format for: {source}")
