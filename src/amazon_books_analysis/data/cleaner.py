"""
Data cleaning module for Amazon Books Analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

from ..config.loader import load_config, Config

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and preprocesses the Amazon books dataset."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.data_config = self.config.data
        self.column_mapping = self.data_config.column_mapping
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset.
        
        Args:
            df: Raw DataFrame to clean.
            
        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting data cleaning process")
        df_clean = df.copy()
        
        # 1. Rename columns to snake_case
        df_clean = self._rename_columns(df_clean)
        
        # 2. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 3. Convert data types
        df_clean = self._convert_types(df_clean)
        
        # 4. Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # 5. Standardize text fields
        df_clean = self._standardize_text(df_clean)
        
        logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
        return df_clean
    
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to snake_case."""
        logger.debug("Renaming columns")
        df = df.rename(columns=self.column_mapping)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.debug("Handling missing values")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.debug(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                        logger.debug(f"Filled missing values in {col} with mode: {mode_val.iloc[0]}")
        
        return df
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        logger.debug("Converting data types")
        
        # Ensure numeric columns are proper types
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        if 'user_rating' in df.columns:
            df['user_rating'] = pd.to_numeric(df['user_rating'], errors='coerce')
        
        if 'reviews' in df.columns:
            df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
        
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Report outliers using the IQR method for price and reviews.

        Values are intentionally left untouched: this pipeline's analysis
        and insight modules perform their own outlier detection, and
        clipping here would silently censor the extreme values before
        that analysis ever sees them.
        """
        logger.debug("Checking for outliers")

        for col in ['price', 'reviews']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    logger.warning(
                        f"Found {len(outliers)} potential outliers in {col} "
                        f"(outside [{lower_bound:.2f}, {upper_bound:.2f}]); left unmodified"
                    )

        return df
    
    def _standardize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text fields."""
        logger.debug("Standardizing text fields")
        
        # Standardize genre
        if 'genre' in df.columns:
            df['genre'] = df['genre'].str.strip().str.title()
            # Standardize genre values
            genre_mapping = {
                'Fiction': 'Fiction',
                'Non Fiction': 'Non Fiction',
                'Non-Fiction': 'Non Fiction',
                'Nonfiction': 'Non Fiction',
            }
            df['genre'] = df['genre'].map(genre_mapping).fillna(df['genre'])
        
        # Standardize author names
        if 'author' in df.columns:
            df['author'] = df['author'].str.strip()
        
        # Standardize book names
        if 'name' in df.columns:
            df['name'] = df['name'].str.strip()
        
        return df
    
    def get_cleaning_report(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> Dict[str, Any]:
        """Generate a report of cleaning operations performed."""
        return {
            "original_shape": df_raw.shape,
            "cleaned_shape": df_clean.shape,
            "rows_removed": df_raw.shape[0] - df_clean.shape[0],
            "columns_renamed": list(self.column_mapping.keys()),
            "missing_values_filled": df_raw.isnull().sum().sum() - df_clean.isnull().sum().sum(),
            "dtypes_changed": {
                col: str(df_raw[col].dtype) + " -> " + str(df_clean[col].dtype)
                for col in df_raw.columns
                if col in df_clean.columns and df_raw[col].dtype != df_clean[col].dtype
            }
        }