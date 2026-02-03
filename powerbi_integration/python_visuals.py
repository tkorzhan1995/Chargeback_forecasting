"""
Python visuals module for Power BI integration.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from utils import get_logger

logger = get_logger(__name__)


def prepare_data_for_powerbi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for Power BI consumption.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe for Power BI
    """
    logger.info("Preparing data for Power BI")
    
    # Convert datetime columns
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle NaN values
    df = df.fillna('')
    
    # Ensure all column names are strings
    df.columns = df.columns.astype(str)
    
    return df


def create_forecast_visual(historical: pd.DataFrame, 
                          forecast: pd.DataFrame,
                          date_col: str = 'date',
                          value_col: str = 'value') -> Dict[str, Any]:
    """
    Create forecast visualization for Power BI Python visual.
    
    Args:
        historical: Historical data
        forecast: Forecast data
        date_col: Date column name
        value_col: Value column name
        
    Returns:
        Dict: Visualization configuration
    """
    logger.info("Creating forecast visual")
    
    plt.figure(figsize=(12, 6))
    
    # Plot historical
    plt.plot(historical[date_col], historical[value_col], 
            label='Historical', color='blue', linewidth=2)
    
    # Plot forecast
    plt.plot(forecast[date_col], forecast[value_col], 
            label='Forecast', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence intervals if available
    if 'lower_bound' in forecast.columns and 'upper_bound' in forecast.columns:
        plt.fill_between(forecast[date_col], 
                        forecast['lower_bound'], 
                        forecast['upper_bound'],
                        alpha=0.2, color='red')
    
    plt.xlabel('Date')
    plt.ylabel(value_col.replace('_', ' ').title())
    plt.title('Chargeback Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {'plot': plt}


def create_heatmap_visual(df: pd.DataFrame, 
                         x_col: str, 
                         y_col: str,
                         value_col: str) -> Dict[str, Any]:
    """
    Create heatmap visualization for Power BI Python visual.
    
    Args:
        df: Input dataframe
        x_col: X-axis column
        y_col: Y-axis column
        value_col: Value column for heatmap
        
    Returns:
        Dict: Visualization configuration
    """
    logger.info("Creating heatmap visual")
    
    # Pivot data for heatmap
    pivot_data = df.pivot_table(index=y_col, columns=x_col, 
                                values=value_col, aggfunc='sum')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title(f'{value_col.replace("_", " ").title()} Heatmap')
    plt.tight_layout()
    
    return {'plot': plt}


def create_feature_importance_visual(feature_importance: pd.DataFrame,
                                    top_n: int = 10) -> Dict[str, Any]:
    """
    Create feature importance visualization.
    
    Args:
        feature_importance: Feature importance dataframe
        top_n: Number of top features to display
        
    Returns:
        Dict: Visualization configuration
    """
    logger.info("Creating feature importance visual")
    
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    return {'plot': plt}


# Example Power BI Python script
POWERBI_PYTHON_SCRIPT_EXAMPLE = """
# Power BI Python Visual Script Example
# This script can be used in Power BI Python visual

import pandas as pd
import matplotlib.pyplot as plt

# dataset is automatically available in Power BI Python visual
# Ensure your data has been passed to the Python visual

# Example: Plot forecast
if 'date' in dataset.columns and 'predicted_volume' in dataset.columns:
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.plot(dataset['date'], dataset['predicted_volume'], 
            label='Forecast', color='red', linewidth=2)
    
    # Add confidence intervals if available
    if 'lower_bound' in dataset.columns and 'upper_bound' in dataset.columns:
        plt.fill_between(dataset['date'], 
                        dataset['lower_bound'], 
                        dataset['upper_bound'],
                        alpha=0.2, color='red')
    
    plt.xlabel('Date')
    plt.ylabel('Predicted Chargeback Volume')
    plt.title('Chargeback Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
"""


def generate_powerbi_python_scripts() -> Dict[str, str]:
    """
    Generate example Python scripts for Power BI visuals.
    
    Returns:
        Dict: Dictionary of script names and code
    """
    scripts = {
        'forecast_visual': POWERBI_PYTHON_SCRIPT_EXAMPLE,
        'heatmap_visual': """
# Power BI Heatmap Script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pivot data for heatmap
pivot_data = dataset.pivot_table(
    index='product_category', 
    columns='channel', 
    values='chargeback_count', 
    aggfunc='sum'
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Chargeback Count by Product and Channel')
plt.tight_layout()
plt.show()
""",
        'trend_analysis': """
# Power BI Trend Analysis Script
import pandas as pd
import matplotlib.pyplot as plt

dataset['date'] = pd.to_datetime(dataset['date'])
dataset = dataset.sort_values('date')

plt.figure(figsize=(14, 6))

# Plot actual vs forecast
plt.plot(dataset[dataset['type']=='actual']['date'], 
        dataset[dataset['type']=='actual']['value'], 
        label='Actual', color='blue', linewidth=2)

plt.plot(dataset[dataset['type']=='forecast']['date'], 
        dataset[dataset['type']=='forecast']['value'], 
        label='Forecast', color='red', linewidth=2, linestyle='--')

plt.xlabel('Date')
plt.ylabel('Chargeback Volume')
plt.title('Actual vs Forecast Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    }
    
    return scripts
