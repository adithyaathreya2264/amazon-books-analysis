"""
High-level analysis pipeline for Amazon Books Analysis.
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from .config.loader import load_config, Config
from .data.loader import DataLoader
from .data.cleaner import DataCleaner
from .data.validator import DataValidator
from .analysis.statistical import StatisticalAnalyzer
from .analysis.correlation import CorrelationAnalyzerExtended
from .analysis.trends import TrendAnalyzer
from .insights.generator import InsightGenerator
from .insights.reporter import ReportGenerator
from .visualization.plots import PlotGenerator
from .visualization.saver import FigureSaver

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    High-level pipeline for running the complete Amazon Books analysis.
    
    This class provides a simple interface to run the full analysis pipeline
    including data loading, cleaning, analysis, visualization, and reporting.
    """
    
    def __init__(self, config_path: Optional[Union[str, Config]] = None):
        """
        Initialize the analysis pipeline.

        Args:
            config_path: Optional path to a configuration file, or an
                already-built Config object (useful when the caller needs
                to override config values, e.g. output directories, before
                the pipeline's components are constructed).
        """
        self.config = config_path if isinstance(config_path, Config) else load_config(config_path)
        self.data_config = self.config.data
        self.viz_config = self.config.visualization
        
        # Initialize components
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.validator = DataValidator(self.config)
        self.stat_analyzer = StatisticalAnalyzer(self.config)
        self.corr_analyzer = CorrelationAnalyzerExtended(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.insight_generator = InsightGenerator(self.config)
        self.plot_generator = PlotGenerator(self.config)
        self.figure_saver = FigureSaver(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        self.df: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and validate the dataset.
        
        Args:
            data_path: Optional path to data file. If None, uses config path.
            
        Returns:
            Cleaned and validated DataFrame.
        """
        if data_path is None:
            data_path = self.data_config.path
        
        logger.info(f"Loading data from {data_path}")
        
        # Load raw data
        df_raw = self.loader.load_data(data_path)
        
        # Validate raw data
        validation_result = self.validator.validate_raw_data(df_raw)
        if not validation_result.is_valid:
            logger.warning(f"Raw data validation warnings: {validation_result.errors}")
        
        # Clean data
        self.df = self.cleaner.clean_data(df_raw)
        
        # Validate cleaned data
        validation_result = self.validator.validate_cleaned_data(self.df)
        if not validation_result.is_valid:
            logger.error(f"Cleaned data validation failed: {validation_result.errors}")
            raise ValueError("Cleaned data validation failed")
        
        logger.info(f"Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def run_analysis(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run the complete statistical analysis.
        
        Args:
            df: Optional DataFrame. If None, uses loaded data.
            
        Returns:
            Dictionary with analysis results.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_data() first or provide df.")
            df = self.df
        
        logger.info("Running statistical analysis")
        
        self.results = {
            'descriptive_stats': self.stat_analyzer.get_descriptive_statistics(df),
            'genre_analysis': self.stat_analyzer.analyze_by_genre(df),
            'author_analysis': self.stat_analyzer.analyze_top_authors(df, top_n=20),
            'correlations': self.corr_analyzer.get_top_correlations(df, 'user_rating', top_n=10),
            'reviews_vs_rating': self.corr_analyzer.analyze_reviews_vs_rating(df),
            'price_vs_rating': self.corr_analyzer.analyze_price_vs_rating(df),
            'price_vs_reviews': self.corr_analyzer.analyze_price_vs_reviews(df),
            'yearly_correlations': self.corr_analyzer.analyze_yearly_correlations(df),
            'yearly_trends': self.trend_analyzer.analyze_yearly_trends(df),
            'genre_trends': self.trend_analyzer.analyze_genre_trends(df),
            'top_authors_trends': self.trend_analyzer.analyze_top_authors_trends(df, top_n=10),
            'rating_distribution_trends': self.trend_analyzer.analyze_rating_distribution_trends(df),
            'price_distribution_trends': self.trend_analyzer.analyze_price_distribution_trends(df),
        }
        
        # Outlier detection
        self.results['outliers'] = {}
        for col in ['price', 'user_rating', 'reviews']:
            self.results['outliers'][col] = {
                'iqr': self.stat_analyzer.detect_outliers_iqr(df, col),
                'zscore': self.stat_analyzer.detect_outliers_zscore(df, col)
            }
        
        # Statistical tests
        self.results['statistical_tests'] = {}
        genre_order = self.config.analysis.genre_order or ["Fiction", "Non Fiction"]
        genre_a, genre_b = genre_order[0], genre_order[1]
        group_a = df[df['genre'] == genre_a]
        group_b = df[df['genre'] == genre_b]

        for col in ['user_rating', 'price', 'reviews']:
            test_result = self.stat_analyzer.compare_groups_mannwhitney(
                group_a[col], group_b[col], group1_name=genre_a, group2_name=genre_b
            )
            self.results['statistical_tests'][f'genre_{col}'] = test_result
        
        for col in ['user_rating', 'price', 'reviews']:
            trend_result = self.stat_analyzer.test_trend_significance(df, 'year', col)
            self.results['statistical_tests'][f'trend_{col}'] = trend_result
        
        logger.info("Statistical analysis completed")
        return self.results
    
    def generate_insights(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate insights from the analysis.
        
        Args:
            df: Optional DataFrame. If None, uses loaded data.
            
        Returns:
            Dictionary with insights.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_data() first or provide df.")
            df = self.df
        
        logger.info("Generating insights")
        insights = self.insight_generator.generate_all_insights(df)
        self.results['insights'] = insights
        return insights
    
    def generate_visualizations(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate and save all visualizations.
        
        Args:
            df: Optional DataFrame. If None, uses loaded data.
            
        Returns:
            Dictionary with saved figure paths.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_data() first or provide df.")
            df = self.df
        
        logger.info("Generating visualizations")
        figures = self.plot_generator.generate_all_plots(df)
        saved_files = self.figure_saver.save_all_figures(figures)
        self.results['figures'] = saved_files
        
        total_files = sum(len(v) for v in saved_files.values())
        logger.info(f"Saved {total_files} figure files")
        return saved_files
    
    def generate_reports(self, insights: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Generate analysis reports in multiple formats.
        
        Args:
            insights: Optional insights dictionary. If None, uses generated insights.
            
        Returns:
            Dictionary mapping format to file path.
        """
        if insights is None:
            if 'insights' not in self.results:
                raise ValueError("No insights available. Call generate_insights() first.")
            insights = self.results['insights']

        if insights is None:
            raise ValueError("Insights cannot be None.")
        
        logger.info("Generating reports")
        reports = self.report_generator.generate_all_reports(insights)
        self.results['reports'] = reports
        logger.info(f"Generated reports: {list(reports.keys())}")
        return reports
    
    def run_full_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            data_path: Optional path to data file.
            
        Returns:
            Dictionary with all results.
        """
        logger.info("Starting full analysis pipeline")
        
        # Load data
        self.load_data(data_path)
        
        # Run analysis
        self.run_analysis()
        
        # Generate insights
        self.generate_insights()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate reports
        self.generate_reports()
        
        logger.info("Full analysis pipeline completed")
        return self.results
    
    def get_summary(self) -> str:
        """Get a text summary of the analysis results."""
        if 'insights' not in self.results:
            return "No analysis results available. Run the pipeline first."
        
        insights = self.results['insights']
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("ANALYSIS PIPELINE SUMMARY")
        lines.append("=" * 60)
        
        summary = insights.get('data_summary', {})
        lines.append(f"\nDataset: {summary.get('total_books', 'N/A'):,} books")
        lines.append(f"Years: {summary.get('year_range', ['N/A', 'N/A'])[0]} - {summary.get('year_range', ['N/A', 'N/A'])[1]}")
        lines.append(f"Genres: {', '.join(summary.get('genres', []))}")
        lines.append(f"Authors: {summary.get('unique_authors', 'N/A'):,}")
        
        lines.append("\nKey Findings:")
        for i, finding in enumerate(insights.get('key_findings', []), 1):
            lines.append(f"  {i}. {finding}")
        
        if 'reports' in self.results:
            lines.append("\nReports Generated:")
            for fmt, path in self.results['reports'].items():
                lines.append(f"  {fmt.upper()}: {path}")
        
        if 'figures' in self.results:
            total_figs = sum(len(v) for v in self.results['figures'].values())
            lines.append(f"\nFigures Saved: {total_figs} files")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
