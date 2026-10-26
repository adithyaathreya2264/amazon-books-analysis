"""
Correlation analysis module for Amazon Books Analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from ..config.loader import load_config, Config
from .statistical import CorrelationAnalyzer

logger = logging.getLogger(__name__)


class CorrelationAnalyzerExtended(CorrelationAnalyzer):
    """Extended correlation analyzer with additional methods for book analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
    
    def analyze_reviews_vs_rating(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between reviews and ratings.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary with correlation analysis results.
        """
        logger.info("Analyzing reviews vs ratings correlation")
        
        result = self.correlation_with_significance(df, 'reviews', 'user_rating')
        result['analysis_type'] = 'reviews_vs_rating'
        
        # Add additional insights
        # Correlation by genre
        fiction_corr = self.correlation_with_significance(
            df[df['genre'] == 'Fiction'], 'reviews', 'user_rating'
        )
        nonfiction_corr = self.correlation_with_significance(
            df[df['genre'] == 'Non Fiction'], 'reviews', 'user_rating'
        )
        
        result['by_genre'] = {
            'Fiction': fiction_corr,
            'Non Fiction': nonfiction_corr
        }
        
        return result
    
    def analyze_price_vs_rating(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between price and rating.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary with correlation analysis results.
        """
        logger.info("Analyzing price vs rating correlation")
        
        result = self.correlation_with_significance(df, 'price', 'user_rating')
        result['analysis_type'] = 'price_vs_rating'
        
        # Correlation by genre
        fiction_corr = self.correlation_with_significance(
            df[df['genre'] == 'Fiction'], 'price', 'user_rating'
        )
        nonfiction_corr = self.correlation_with_significance(
            df[df['genre'] == 'Non Fiction'], 'price', 'user_rating'
        )
        
        result['by_genre'] = {
            'Fiction': fiction_corr,
            'Non Fiction': nonfiction_corr
        }
        
        return result
    
    def analyze_price_vs_reviews(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between price and number of reviews.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary with correlation analysis results.
        """
        logger.info("Analyzing price vs reviews correlation")
        
        result = self.correlation_with_significance(df, 'price', 'reviews')
        result['analysis_type'] = 'price_vs_reviews'
        
        # Correlation by genre
        fiction_corr = self.correlation_with_significance(
            df[df['genre'] == 'Fiction'], 'price', 'reviews'
        )
        nonfiction_corr = self.correlation_with_significance(
            df[df['genre'] == 'Non Fiction'], 'price', 'reviews'
        )
        
        result['by_genre'] = {
            'Fiction': fiction_corr,
            'Non Fiction': nonfiction_corr
        }
        
        return result
    
    def analyze_yearly_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how correlations change over years vs other variables.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary with yearly correlation analysis.
        """
        logger.info("Analyzing yearly correlations")
        
        years = sorted(df['year'].unique())
        results = {}
        
        for year in years:
            year_data = df[df['year'] == year]
            if len(year_data) > 3:
                results[int(year)] = {
                    'reviews_vs_rating': self.correlation_with_significance(
                        year_data, 'reviews', 'user_rating'
                    ),
                    'price_vs_rating': self.correlation_with_significance(
                        year_data, 'price', 'user_rating'
                    ),
                    'price_vs_reviews': self.correlation_with_significance(
                        year_data, 'price', 'reviews'
                    )
                }
        
        return {
            'analysis_type': 'yearly_correlations',
            'yearly_results': results
        }
    
    def get_top_correlations(self, df: pd.DataFrame, 
                            target: str = 'user_rating',
                            top_n: int = 5) -> Dict[str, Any]:
        """
        Get top correlations with a target variable.
        
        Args:
            df: Cleaned DataFrame.
            target: Target variable to correlate with.
            top_n: Number of top correlations to return.
            
        Returns:
            Dictionary with top correlations.
        """
        logger.info(f"Finding top {top_n} correlations with {target}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target not in numeric_cols:
            raise ValueError(f"Target column '{target}' not found in numeric columns")
        
        correlations = []
        for col in numeric_cols:
            if col != target:
                result = self.correlation_with_significance(df, col, target)
                correlations.append({
                    'variable': col,
                    'correlation': result['correlation'],
                    'p_value': result['p_value'],
                    'significant': result['significant']
                })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'target': target,
            'top_correlations': correlations[:top_n],
            'all_correlations': correlations
        }