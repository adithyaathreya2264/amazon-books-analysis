"""
Trend analysis module for Amazon Books Analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from ..config.loader import load_config, Config

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyzes trends over time in the book data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the trend analyzer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.analysis_config = self.config.analysis
        self.year_range = self.analysis_config.year_range

    def _warn_if_outside_configured_range(self, years: List[int]) -> None:
        """Warn when the data contains years outside the configured analysis.year_range."""
        if not years:
            return
        out_of_range = [y for y in years if y < self.year_range[0] or y > self.year_range[1]]
        if out_of_range:
            logger.warning(
                f"Data contains years {out_of_range} outside the configured "
                f"year_range {self.year_range}; analyzing all available years anyway."
            )

    def analyze_yearly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze yearly trends for key metrics.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary with yearly trend analysis.
        """
        logger.info("Analyzing yearly trends")

        years = sorted(df['year'].unique())
        self._warn_if_outside_configured_range(years)
        trends = {}
        
        for year in years:
            year_data = df[df['year'] == year]
            trends[int(year)] = {
                'count': len(year_data),
                'avg_rating': float(year_data['user_rating'].mean()),
                'median_rating': float(year_data['user_rating'].median()),
                'avg_price': float(year_data['price'].mean()),
                'median_price': float(year_data['price'].median()),
                'avg_reviews': float(year_data['reviews'].mean()),
                'median_reviews': float(year_data['reviews'].median()),
                'fiction_count': int((year_data['genre'] == 'Fiction').sum()),
                'nonfiction_count': int((year_data['genre'] == 'Non Fiction').sum()),
                'fiction_pct': float((year_data['genre'] == 'Fiction').mean() * 100),
                'nonfiction_pct': float((year_data['genre'] == 'Non Fiction').mean() * 100)
            }
        
        # Compute year-over-year changes
        yoy_changes = {}
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            yoy_changes[f"{prev_year}_to_{curr_year}"] = {
                'count_change': trends[curr_year]['count'] - trends[prev_year]['count'],
                'count_pct_change': ((trends[curr_year]['count'] - trends[prev_year]['count']) 
                                     / trends[prev_year]['count'] * 100) if trends[prev_year]['count'] > 0 else 0,
                'rating_change': trends[curr_year]['avg_rating'] - trends[prev_year]['avg_rating'],
                'price_change': trends[curr_year]['avg_price'] - trends[prev_year]['avg_price'],
                'reviews_change': trends[curr_year]['avg_reviews'] - trends[prev_year]['avg_reviews'],
                'fiction_pct_change': trends[curr_year]['fiction_pct'] - trends[prev_year]['fiction_pct']
            }
        
        return {
            'yearly_metrics': trends,
            'year_over_year_changes': yoy_changes,
            'year_range': [int(years[0]), int(years[-1])],
            'total_years': len(years)
        }
    
    def analyze_genre_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trends by genre over time.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary with genre trend analysis.
        """
        logger.info("Analyzing genre trends over time")
        
        years = sorted(df['year'].unique())
        genres = df['genre'].unique()
        
        genre_trends = {}
        for genre in genres:
            genre_data = df[df['genre'] == genre]
            genre_trends[genre] = {}
            
            for year in years:
                year_data = genre_data[genre_data['year'] == year]
                if len(year_data) > 0:
                    genre_trends[genre][int(year)] = {
                        'count': len(year_data),
                        'avg_rating': float(year_data['user_rating'].mean()),
                        'avg_price': float(year_data['price'].mean()),
                        'avg_reviews': float(year_data['reviews'].mean()),
                        'pct_of_year': float(len(year_data) / len(df[df['year'] == year]) * 100)
                    }
        
        return {
            'genre_trends': genre_trends,
            'years': [int(y) for y in years],
            'genres': list(genres)
        }
    
    def analyze_top_authors_trends(self, df: pd.DataFrame, 
                                   top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze trends for top authors over time.
        
        Args:
            df: Cleaned DataFrame.
            top_n: Number of top authors to analyze.
            
        Returns:
            Dictionary with top authors trend analysis.
        """
        logger.info(f"Analyzing trends for top {top_n} authors")
        
        # Get top authors by total appearances
        author_counts = df['author'].value_counts()
        top_authors = author_counts.head(top_n).index.tolist()
        
        years = sorted(df['year'].unique())
        author_trends = {}
        
        for author in top_authors:
            author_data = df[df['author'] == author]
            author_trends[author] = {}
            
            for year in years:
                year_data = author_data[author_data['year'] == year]
                if len(year_data) > 0:
                    author_trends[author][int(year)] = {
                        'count': len(year_data),
                        'avg_rating': float(year_data['user_rating'].mean()),
                        'avg_price': float(year_data['price'].mean()),
                        'books': year_data['name'].tolist()
                    }
        
        return {
            'top_authors': top_authors,
            'author_trends': author_trends,
            'years': [int(y) for y in years]
        }
    
    def analyze_rating_distribution_trends(self, df: pd.DataFrame, 
                                           bins: int = 10) -> Dict[str, Any]:
        """
        Analyze how rating distribution changes over time.
        
        Args:
            df: Cleaned DataFrame.
            bins: Number of bins for histogram.
            
        Returns:
            Dictionary with rating distribution trends.
        """
        logger.info("Analyzing rating distribution trends")
        
        years = sorted(df['year'].unique())
        distribution_trends = {}
        
        for year in years:
            year_data = df[df['year'] == year]['user_rating']
            hist, bin_edges = np.histogram(year_data, bins=bins, range=(0, 5))
            
            distribution_trends[int(year)] = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mean': float(year_data.mean()),
                'std': float(year_data.std()),
                'skewness': float(year_data.skew()),
                'kurtosis': float(year_data.kurtosis())
            }
        
        return {
            'yearly_distributions': distribution_trends,
            'bins': bins,
            'range': [0, 5]
        }
    
    def analyze_price_distribution_trends(self, df: pd.DataFrame, 
                                          bins: int = 20) -> Dict[str, Any]:
        """
        Analyze how price distribution changes over time.
        
        Args:
            df: Cleaned DataFrame.
            bins: Number of bins for histogram.
            
        Returns:
            Dictionary with price distribution trends.
        """
        logger.info("Analyzing price distribution trends")
        
        years = sorted(df['year'].unique())
        max_price = df['price'].max()
        
        distribution_trends = {}
        
        for year in years:
            year_data = df[df['year'] == year]['price']
            hist, bin_edges = np.histogram(year_data, bins=bins, range=(0, max_price))
            
            distribution_trends[int(year)] = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mean': float(year_data.mean()),
                'median': float(year_data.median()),
                'std': float(year_data.std()),
                'skewness': float(year_data.skew()),
                'kurtosis': float(year_data.kurtosis())
            }
        
        return {
            'yearly_distributions': distribution_trends,
            'bins': bins,
            'range': [0, float(max_price)]
        }
    
    def detect_trend_changes(self, df: pd.DataFrame, 
                            metric: str = 'user_rating',
                            window: int = 3) -> Dict[str, Any]:
        """
        Detect significant changes in trends using rolling statistics.
        
        Args:
            df: Cleaned DataFrame.
            metric: Metric to analyze ('user_rating', 'price', 'reviews').
            window: Rolling window size.
            
        Returns:
            Dictionary with trend change detection results.
        """
        logger.info(f"Detecting trend changes for {metric}")
        
        yearly_avg = df.groupby('year')[metric].mean().reset_index()
        yearly_avg = yearly_avg.sort_values('year')
        
        # Rolling statistics
        yearly_avg['rolling_mean'] = yearly_avg[metric].rolling(window=window, center=True).mean()
        yearly_avg['rolling_std'] = yearly_avg[metric].rolling(window=window, center=True).std()
        yearly_avg['pct_change'] = yearly_avg[metric].pct_change() * 100
        
        # Detect anomalies (values outside 2 std from rolling mean)
        yearly_avg['anomaly'] = np.abs(yearly_avg[metric] - yearly_avg['rolling_mean']) > 2 * yearly_avg['rolling_std']
        
        anomalies = yearly_avg[yearly_avg['anomaly']][['year', metric, 'rolling_mean', 'rolling_std']].to_dict('records')
        
        return {
            'metric': metric,
            'yearly_data': yearly_avg.to_dict('records'),
            'anomalies': anomalies,
            'window': window
        }