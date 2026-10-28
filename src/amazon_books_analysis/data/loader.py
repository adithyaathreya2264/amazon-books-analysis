"""
Data loading module for Amazon Books Analysis.
"""

import pandas as pd
from typing import Optional
import logging

from ..config.loader import load_config, Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and provides access to the Amazon books dataset."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.data_config = self.config.data
        self._df: Optional[pd.DataFrame] = None
    
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Args:
            path: Optional path to CSV file. Uses config default if not provided.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
            pd.errors.EmptyDataError: If the file is empty.
        """
        file_path = path or self.data_config.path
        logger.info(f"Loading data from {file_path}")
        
        try:
            self._df = pd.read_csv(file_path, encoding=self.data_config.encoding)
            logger.info(f"Successfully loaded {len(self._df)} records")
            return self._df
        except FileNotFoundError:
            logger.error(f"Data file not found at {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Data file is empty: {file_path}")
            raise
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data. Loads if not already loaded.
        
        Returns:
            DataFrame with the data.
        """
        if self._df is None:
            return self.load_data()
        return self._df
    
    def get_shape(self) -> tuple:
        """Get the shape of the loaded data."""
        df = self.get_data()
        return df.shape
    
    def get_columns(self) -> list:
        """Get the column names of the loaded data."""
        df = self.get_data()
        return list(df.columns)
    
    def get_dtypes(self) -> pd.Series:
        """Get the data types of the loaded data."""
        df = self.get_data()
        return df.dtypes
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """Get a sample of the data."""
        df = self.get_data()
        return df.head(n)
    
    def get_info(self) -> dict:
        """Get basic information about the dataset."""
        df = self.get_data()
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
        }