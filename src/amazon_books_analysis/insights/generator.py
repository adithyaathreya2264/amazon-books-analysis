"""
Insight generation module for Amazon Books Analysis.
"""

import pandas as pd
from typing import Any, Dict, List, Optional, cast
import logging
from datetime import datetime

from ..config.loader import Config, load_config
from ..analysis.statistical import StatisticalAnalyzer
from ..analysis.correlation import CorrelationAnalyzerExtended
from ..analysis.trends import TrendAnalyzer

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Generates insights and findings from book data analysis."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the insight generator.

        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.analysis_config = self.config.analysis

        # Initialize analyzers
        self.stat_analyzer = StatisticalAnalyzer(self.config)
        self.corr_analyzer = CorrelationAnalyzerExtended(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
    
    def generate_all_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate all insights from the data.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Dictionary containing all insights.
        """
        logger.info("Generating all insights")
        
        insights = {
            'generated_at': datetime.now().isoformat(),
            'data_summary': self._get_data_summary(df),
            'descriptive_statistics': self._get_descriptive_stats(df),
            'genre_insights': self._get_genre_insights(df),
            'author_insights': self._get_author_insights(df, top_n=self.analysis_config.top_authors_count),
            'correlation_insights': self._get_correlation_insights(df),
            'trend_insights': self._get_trend_insights(df),
            'outlier_insights': self._get_outlier_insights(df),
            'statistical_test_insights': self._get_statistical_test_insights(df),
            'key_findings': []
        }
        
        # Generate key findings summary
        insights['key_findings'] = self._generate_key_findings(insights)
        
        return insights
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic data summary."""
        return {
            'total_books': len(df),
            'year_range': [int(df['year'].min()), int(df['year'].max())],
            'genres': df['genre'].unique().tolist(),
            'unique_authors': df['author'].nunique(),
            'price_range': [float(df['price'].min()), float(df['price'].max())],
            'rating_range': [float(df['user_rating'].min()), float(df['user_rating'].max())],
            'reviews_range': [int(df['reviews'].min()), int(df['reviews'].max())]
        }
    
    def _get_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get descriptive statistics for numeric columns."""
        numeric_cols = ['price', 'user_rating', 'reviews']
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        return stats
    
    def _get_genre_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get insights by genre."""
        insights = {}
        
        for genre in df['genre'].unique():
            genre_data = df[df['genre'] == genre]
            insights[genre] = {
                'count': len(genre_data),
                'percentage': float(len(genre_data) / len(df) * 100),
                'avg_rating': float(genre_data['user_rating'].mean()),
                'median_rating': float(genre_data['user_rating'].median()),
                'avg_price': float(genre_data['price'].mean()),
                'median_price': float(genre_data['price'].median()),
                'avg_reviews': float(genre_data['reviews'].mean()),
                'median_reviews': float(genre_data['reviews'].median()),
                'top_authors': genre_data['author'].value_counts().head(5).to_dict(),
                'price_range': [float(genre_data['price'].min()), float(genre_data['price'].max())],
                'rating_range': [float(genre_data['user_rating'].min()), float(genre_data['user_rating'].max())]
            }
        
        # Compare the two configured genres
        genre_a, genre_b = self.analysis_config.genre_order[0], self.analysis_config.genre_order[1]
        group_a = df[df['genre'] == genre_a]
        group_b = df[df['genre'] == genre_b]

        insights['comparison'] = {
            'genres_compared': [genre_a, genre_b],
            'rating_diff': float(group_a['user_rating'].mean() - group_b['user_rating'].mean()),
            'price_diff': float(group_a['price'].mean() - group_b['price'].mean()),
            'reviews_diff': float(group_a['reviews'].mean() - group_b['reviews'].mean()),
            'count_diff': len(group_a) - len(group_b)
        }

        return insights

    def _get_author_insights(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        """Get insights about authors."""
        author_stats = df.groupby('author').agg(
            book_count=('name', 'count'),
            avg_rating=('user_rating', 'mean'),
            avg_price=('price', 'mean'),
            total_reviews=('reviews', 'sum'),
            avg_reviews=('reviews', 'mean')
        ).reset_index()

        # Top authors by count
        top_by_count = author_stats.nlargest(top_n, 'book_count')

        # Top authors by rating (min 2 books)
        top_by_rating = author_stats[author_stats['book_count'] >= 2].nlargest(top_n, 'avg_rating')

        # Most reviewed authors
        top_by_reviews = author_stats.nlargest(top_n, 'total_reviews')

        # Authors appearing in multiple years
        author_years = df.groupby('author')['year'].nunique().reset_index()
        author_years.columns = ['author', 'years_active']
        multi_year_authors = author_years[author_years['years_active'] > 1].sort_values('years_active', ascending=False)

        return {
            'total_unique_authors': df['author'].nunique(),
            'top_by_count': top_by_count[['author', 'book_count', 'avg_rating']].to_dict('records'),
            'top_by_rating': top_by_rating[['author', 'book_count', 'avg_rating']].to_dict('records'),
            'top_by_reviews': top_by_reviews[['author', 'total_reviews', 'avg_rating']].to_dict('records'),
            'multi_year_authors': multi_year_authors.head(top_n).to_dict('records'),
            'avg_books_per_author': float(df['author'].value_counts().mean()),
            'max_books_by_author': int(df['author'].value_counts().max())
        }
    
    def _get_correlation_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation insights."""
        insights = {}
        
        # Overall correlations
        insights['overall'] = self.corr_analyzer.get_top_correlations(df, 'user_rating', top_n=5)
        
        # Specific analyses
        insights['reviews_vs_rating'] = self.corr_analyzer.analyze_reviews_vs_rating(df)
        insights['price_vs_rating'] = self.corr_analyzer.analyze_price_vs_rating(df)
        insights['price_vs_reviews'] = self.corr_analyzer.analyze_price_vs_reviews(df)
        insights['yearly_correlations'] = self.corr_analyzer.analyze_yearly_correlations(df)
        
        return insights
    
    def _get_trend_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get trend insights."""
        insights = {}
        
        insights['yearly_trends'] = self.trend_analyzer.analyze_yearly_trends(df)
        insights['genre_trends'] = self.trend_analyzer.analyze_genre_trends(df)
        insights['top_authors_trends'] = self.trend_analyzer.analyze_top_authors_trends(df, top_n=10)
        insights['rating_distribution_trends'] = self.trend_analyzer.analyze_rating_distribution_trends(df)
        insights['price_distribution_trends'] = self.trend_analyzer.analyze_price_distribution_trends(df)
        
        # Detect trend changes
        for metric in ['user_rating', 'price', 'reviews']:
            insights[f'{metric}_trend_changes'] = self.trend_analyzer.detect_trend_changes(df, metric)
        
        return insights
    
    def _get_outlier_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get outlier insights."""
        insights = {}
        
        for col in ['price', 'user_rating', 'reviews']:
            outliers = self.stat_analyzer.detect_outliers_iqr(df, col)
            zscore_outliers = self.stat_analyzer.detect_outliers_zscore(df, col)
            
            insights[col] = {
                'iqr_outliers': {
                    'count': len(outliers),
                    'percentage': float(len(outliers) / len(df) * 100),
                    'min': float(outliers[col].min()) if len(outliers) > 0 else None,
                    'max': float(outliers[col].max()) if len(outliers) > 0 else None,
                    'examples': outliers.nlargest(5, col)[['name', 'author', col]].to_dict('records')
                },
                'zscore_outliers': {
                    'count': len(zscore_outliers),
                    'percentage': float(len(zscore_outliers) / len(df) * 100),
                    'examples': zscore_outliers.nlargest(5, col)[['name', 'author', col]].to_dict('records')
                }
            }
        
        return insights

    def _get_statistical_test_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get significance-test insights: genre comparisons and yearly trend tests."""
        insights: Dict[str, Any] = {'genre_comparison': {}, 'yearly_trend': {}}

        genre_a, genre_b = self.analysis_config.genre_order[0], self.analysis_config.genre_order[1]
        group_a = df[df['genre'] == genre_a]
        group_b = df[df['genre'] == genre_b]

        for col in ['user_rating', 'price', 'reviews']:
            insights['genre_comparison'][col] = self.stat_analyzer.compare_groups_mannwhitney(
                group_a[col], group_b[col], group1_name=genre_a, group2_name=genre_b
            )

        for col in ['user_rating', 'price', 'reviews']:
            insights['yearly_trend'][col] = self.stat_analyzer.test_trend_significance(df, 'year', col)

        return insights

    def _generate_key_findings(self, insights: Dict[str, Any]) -> List[str]:
        """Generate key findings from all insights."""
        findings = []
        
        # Data summary findings
        summary = insights['data_summary']
        findings.append(f"Dataset contains {summary['total_books']:,} books from {summary['year_range'][0]} to {summary['year_range'][1]}")
        findings.append(f"Genres: {', '.join(summary['genres'])} ({summary['unique_authors']} unique authors)")
        
        # Genre comparison
        genre_comp = insights['genre_insights']['comparison']
        if genre_comp['rating_diff'] > 0:
            findings.append(f"Fiction books have higher average ratings ({genre_comp['rating_diff']:.2f} points higher)")
        else:
            findings.append(f"Non-Fiction books have higher average ratings ({abs(genre_comp['rating_diff']):.2f} points higher)")
        
        if genre_comp['price_diff'] > 0:
            findings.append(f"Fiction books are more expensive on average (${genre_comp['price_diff']:.2f} more)")
        else:
            findings.append(f"Non-Fiction books are more expensive on average (${abs(genre_comp['price_diff']):.2f} more)")
        
        # Correlation findings
        corr = insights['correlation_insights']['overall']
        top_corr = corr['top_correlations'][0] if corr['top_correlations'] else None
        if top_corr and top_corr['significant']:
            findings.append(f"Strongest correlation with rating: {top_corr['variable']} (r={top_corr['correlation']:.3f}, p={top_corr['p_value']:.4f})")
        
        # Reviews vs rating
        rv = insights['correlation_insights']['reviews_vs_rating']
        if rv['significant']:
            findings.append(f"Reviews vs Rating correlation: {rv['correlation']:.3f} (p={rv['p_value']:.4f}) - {'positive' if rv['correlation'] > 0 else 'negative'} relationship")
        
        # Price vs rating
        pv = insights['correlation_insights']['price_vs_rating']
        if pv['significant']:
            findings.append(f"Price vs Rating correlation: {pv['correlation']:.3f} (p={pv['p_value']:.4f}) - {'positive' if pv['correlation'] > 0 else 'negative'} relationship")
        
        # Trend findings
        yearly = insights['trend_insights']['yearly_trends']
        if yearly['year_over_year_changes']:
            latest_change = list(yearly['year_over_year_changes'].values())[-1]
            findings.append(f"Year-over-year change ({list(yearly['year_over_year_changes'].keys())[-1]}): "
                          f"Books: {latest_change['count_pct_change']:+.1f}%, "
                          f"Rating: {latest_change['rating_change']:+.3f}, "
                          f"Price: ${latest_change['price_change']:+.2f}")
        
        # Author findings
        author = insights['author_insights']
        top_author = author['top_by_count'][0] if author['top_by_count'] else None
        if top_author:
            findings.append(f"Most prolific author: {top_author['author']} with {top_author['book_count']} books (avg rating: {top_author['avg_rating']:.2f})")
        
        # Outlier findings
        for col in ['price', 'user_rating', 'reviews']:
            outlier_info = insights['outlier_insights'][col]['iqr_outliers']
            if outlier_info['count'] > 0:
                findings.append(f"{col.capitalize()} has {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%) using IQR method")
        
        return findings
    
    def get_genre_comparison_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a focused genre comparison summary."""
        comparison = self._get_genre_insights(df)['comparison']
        return cast(Dict[str, Any], comparison)
    
    def get_top_correlations_summary(self, df: pd.DataFrame, target: str = 'user_rating', top_n: int = 5) -> Dict[str, Any]:
        """Get top correlations summary."""
        return self.corr_analyzer.get_top_correlations(df, target, top_n)
    
    def get_trend_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get trend summary."""
        trends = self.trend_analyzer.analyze_yearly_trends(df)
        return {
            'year_range': trends['year_range'],
            'total_years': trends['total_years'],
            'latest_year_metrics': trends['yearly_metrics'][trends['year_range'][1]],
            'key_changes': trends['year_over_year_changes']
        }
