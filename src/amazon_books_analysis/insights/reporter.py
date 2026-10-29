"""
Report generation module for Amazon Books Analysis.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config.loader import Config, load_config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports in various formats."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the report generator.

        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.output_config = self.config.output
        self.output_dir = Path(self.output_config.reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment for HTML reports
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    def generate_json_report(self, insights: Dict[str, Any], 
                             filename: Optional[str] = None) -> Path:
        """
        Generate a JSON report from insights.
        
        Args:
            insights: Insights dictionary from InsightGenerator.
            filename: Optional filename. If None, generates timestamped name.
            
        Returns:
            Path to the generated report.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insights_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        serializable_insights = self._make_serializable(insights)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_insights, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {filepath}")
        return filepath
    
    def generate_text_report(self, insights: Dict[str, Any],
                             filename: Optional[str] = None) -> Path:
        """
        Generate a human-readable text report.
        
        Args:
            insights: Insights dictionary from InsightGenerator.
            filename: Optional filename. If None, generates timestamped name.
            
        Returns:
            Path to the generated report.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insights_report_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._format_text_report(insights))
        
        logger.info(f"Text report saved to {filepath}")
        return filepath
    
    def generate_html_report(self, insights: Dict[str, Any],
                             filename: Optional[str] = None,
                             template_name: str = "report.html") -> Path:
        """
        Generate an HTML report using Jinja2 template.
        
        Args:
            insights: Insights dictionary from InsightGenerator.
            filename: Optional filename. If None, generates timestamped name.
            template_name: Name of the Jinja2 template to use.
            
        Returns:
            Path to the generated report.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insights_report_{timestamp}.html"
        
        filepath = self.output_dir / filename

        template = self.jinja_env.get_template(template_name)

        html_content = template.render(
            insights=insights,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_summary=insights.get('data_summary', {}),
            key_findings=insights.get('key_findings', [])
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {filepath}")
        return filepath
    
    def generate_markdown_report(self, insights: Dict[str, Any],
                                 filename: Optional[str] = None) -> Path:
        """
        Generate a Markdown report.
        
        Args:
            insights: Insights dictionary from InsightGenerator.
            filename: Optional filename. If None, generates timestamped name.
            
        Returns:
            Path to the generated report.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insights_report_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._format_markdown_report(insights))
        
        logger.info(f"Markdown report saved to {filepath}")
        return filepath
    
    def generate_all_reports(self, insights: Dict[str, Any],
                             base_name: Optional[str] = None) -> Dict[str, Path]:
        """
        Generate all report formats.
        
        Args:
            insights: Insights dictionary from InsightGenerator.
            base_name: Base name for files (without extension).
            
        Returns:
            Dictionary mapping format to file path.
        """
        if base_name is None:
            base_name = f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        reports = {}
        reports['json'] = self.generate_json_report(insights, f"{base_name}.json")
        reports['text'] = self.generate_text_report(insights, f"{base_name}.txt")
        reports['markdown'] = self.generate_markdown_report(insights, f"{base_name}.md")
        reports['html'] = self.generate_html_report(insights, f"{base_name}.html")
        
        return reports
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to Python native types for JSON serialization."""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(v) for v in obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _format_text_report(self, insights: Dict[str, Any]) -> str:
        """Format insights as a text report."""
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("AMAZON BOOKS ANALYSIS - INSIGHTS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {insights.get('generated_at', 'Unknown')}")
        lines.append("")
        
        # Data Summary
        lines.append("-" * 80)
        lines.append("DATA SUMMARY")
        lines.append("-" * 80)
        summary = insights.get('data_summary', {})
        lines.append(f"Total Books: {summary.get('total_books', 'N/A'):,}")
        lines.append(f"Year Range: {summary.get('year_range', ['N/A', 'N/A'])[0]} - {summary.get('year_range', ['N/A', 'N/A'])[1]}")
        lines.append(f"Genres: {', '.join(summary.get('genres', []))}")
        lines.append(f"Unique Authors: {summary.get('unique_authors', 'N/A'):,}")
        lines.append(f"Price Range: ${summary.get('price_range', [0, 0])[0]:.2f} - ${summary.get('price_range', [0, 0])[1]:.2f}")
        lines.append(f"Rating Range: {summary.get('rating_range', [0, 0])[0]:.2f} - {summary.get('rating_range', [0, 0])[1]:.2f}")
        lines.append(f"Reviews Range: {summary.get('reviews_range', [0, 0])[0]:,} - {summary.get('reviews_range', [0, 0])[1]:,}")
        lines.append("")
        
        # Key Findings
        lines.append("-" * 80)
        lines.append("KEY FINDINGS")
        lines.append("-" * 80)
        for i, finding in enumerate(insights.get('key_findings', []), 1):
            lines.append(f"{i}. {finding}")
        lines.append("")
        
        # Genre Insights
        lines.append("-" * 80)
        lines.append("GENRE INSIGHTS")
        lines.append("-" * 80)
        genre_insights = insights.get('genre_insights', {})
        for genre in ['Fiction', 'Non Fiction']:
            if genre in genre_insights:
                g = genre_insights[genre]
                lines.append(f"\n{genre}:")
                lines.append(f"  Count: {g['count']:,} ({g['percentage']:.1f}%)")
                lines.append(f"  Avg Rating: {g['avg_rating']:.2f} (Median: {g['median_rating']:.2f})")
                lines.append(f"  Avg Price: ${g['avg_price']:.2f} (Median: ${g['median_price']:.2f})")
                lines.append(f"  Avg Reviews: {g['avg_reviews']:.0f} (Median: {g['median_reviews']:.0f})")
        
        comp = genre_insights.get('comparison', {})
        lines.append(f"\nComparison (Fiction - Non Fiction):")
        lines.append(f"  Rating Difference: {comp.get('rating_diff', 0):+.2f}")
        lines.append(f"  Price Difference: ${comp.get('price_diff', 0):+.2f}")
        lines.append(f"  Reviews Difference: {comp.get('reviews_diff', 0):+.0f}")
        lines.append(f"  Count Difference: {comp.get('count_diff', 0):+,}")
        lines.append("")
        
        # Author Insights
        lines.append("-" * 80)
        lines.append("TOP AUTHORS")
        lines.append("-" * 80)
        author_insights = insights.get('author_insights', {})
        lines.append(f"Total Unique Authors: {author_insights.get('total_unique_authors', 'N/A'):,}")
        lines.append(f"Avg Books per Author: {author_insights.get('avg_books_per_author', 0):.1f}")
        lines.append(f"Max Books by Author: {author_insights.get('max_books_by_author', 0)}")
        lines.append("")
        
        lines.append("Top 10 by Book Count:")
        for i, author in enumerate(author_insights.get('top_by_count', [])[:10], 1):
            lines.append(f"  {i}. {author['author']}: {author['book_count']} books (avg rating: {author['avg_rating']:.2f})")
        lines.append("")
        
        lines.append("Top 10 by Average Rating (min 2 books):")
        for i, author in enumerate(author_insights.get('top_by_rating', [])[:10], 1):
            lines.append(f"  {i}. {author['author']}: {author['avg_rating']:.2f} avg rating ({author['book_count']} books)")
        lines.append("")
        
        # Correlation Insights
        lines.append("-" * 80)
        lines.append("CORRELATION INSIGHTS")
        lines.append("-" * 80)
        corr = insights.get('correlation_insights', {})
        
        rv = corr.get('reviews_vs_rating', {})
        if rv:
            lines.append(f"Reviews vs Rating: r={rv.get('correlation', 0):.3f}, p={rv.get('p_value', 1):.4f} {'*' if rv.get('significant') else ''}")
        
        pv = corr.get('price_vs_rating', {})
        if pv:
            lines.append(f"Price vs Rating: r={pv.get('correlation', 0):.3f}, p={pv.get('p_value', 1):.4f} {'*' if pv.get('significant') else ''}")
        
        pr = corr.get('price_vs_reviews', {})
        if pr:
            lines.append(f"Price vs Reviews: r={pr.get('correlation', 0):.3f}, p={pr.get('p_value', 1):.4f} {'*' if pr.get('significant') else ''}")
        lines.append("")
        
        # Top correlations
        overall = corr.get('overall', {})
        top_corrs = overall.get('top_correlations', [])
        if top_corrs:
            lines.append("Top Correlations with Rating:")
            for i, c in enumerate(top_corrs[:5], 1):
                sig = " *" if c.get('significant') else ""
                lines.append(f"  {i}. {c['variable']}: r={c['correlation']:.3f}, p={c['p_value']:.4f}{sig}")
        lines.append("")
        
        # Trend Insights
        lines.append("-" * 80)
        lines.append("TREND INSIGHTS")
        lines.append("-" * 80)
        trends = insights.get('trend_insights', {})
        yearly = trends.get('yearly_trends', {})
        
        if yearly.get('year_over_year_changes'):
            lines.append("Year-over-Year Changes:")
            for period, change in yearly['year_over_year_changes'].items():
                lines.append(f"  {period}:")
                lines.append(f"    Books: {change['count_pct_change']:+.1f}%")
                lines.append(f"    Rating: {change['rating_change']:+.3f}")
                lines.append(f"    Price: ${change['price_change']:+.2f}")
                lines.append(f"    Reviews: {change['reviews_change']:+.0f}")
        lines.append("")
        
        # Outlier Insights
        lines.append("-" * 80)
        lines.append("OUTLIER ANALYSIS")
        lines.append("-" * 80)
        outliers = insights.get('outlier_insights', {})
        for col in ['price', 'user_rating', 'reviews']:
            o = outliers.get(col, {})
            iqr = o.get('iqr_outliers', {})
            zscore = o.get('zscore_outliers', {})
            lines.append(f"{col.capitalize()}:")
            lines.append(f"  IQR Outliers: {iqr.get('count', 0)} ({iqr.get('percentage', 0):.1f}%)")
            lines.append(f"  Z-Score Outliers: {zscore.get('count', 0)} ({zscore.get('percentage', 0):.1f}%)")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_markdown_report(self, insights: Dict[str, Any]) -> str:
        """Format insights as a Markdown report."""
        lines: List[str] = []
        lines.append("# Amazon Books Analysis - Insights Report")
        lines.append(f"\n*Generated: {insights.get('generated_at', 'Unknown')}*\n")
        
        # Data Summary
        lines.append("## Data Summary")
        summary = insights.get('data_summary', {})
        lines.append(f"- **Total Books**: {summary.get('total_books', 'N/A'):,}")
        lines.append(f"- **Year Range**: {summary.get('year_range', ['N/A', 'N/A'])[0]} - {summary.get('year_range', ['N/A', 'N/A'])[1]}")
        lines.append(f"- **Genres**: {', '.join(summary.get('genres', []))}")
        lines.append(f"- **Unique Authors**: {summary.get('unique_authors', 'N/A'):,}")
        lines.append(f"- **Price Range**: ${summary.get('price_range', [0, 0])[0]:.2f} - ${summary.get('price_range', [0, 0])[1]:.2f}")
        lines.append(f"- **Rating Range**: {summary.get('rating_range', [0, 0])[0]:.2f} - {summary.get('rating_range', [0, 0])[1]:.2f}")
        lines.append(f"- **Reviews Range**: {summary.get('reviews_range', [0, 0])[0]:,} - {summary.get('reviews_range', [0, 0])[1]:,}")
        lines.append("")
        
        # Key Findings
        lines.append("## Key Findings")
        for i, finding in enumerate(insights.get('key_findings', []), 1):
            lines.append(f"{i}. {finding}")
        lines.append("")
        
        # Genre Insights
        lines.append("## Genre Insights")
        genre_insights = insights.get('genre_insights', {})
        
        # Table for genre comparison
        lines.append("| Genre | Count | % | Avg Rating | Median Rating | Avg Price | Median Price | Avg Reviews |")
        lines.append("|-------|-------|---|------------|---------------|-----------|--------------|-------------|")
        for genre in ['Fiction', 'Non Fiction']:
            if genre in genre_insights:
                g = genre_insights[genre]
                lines.append(f"| {genre} | {g['count']:,} | {g['percentage']:.1f}% | {g['avg_rating']:.2f} | {g['median_rating']:.2f} | ${g['avg_price']:.2f} | ${g['median_price']:.2f} | {g['avg_reviews']:.0f} |")
        lines.append("")
        
        comp = genre_insights.get('comparison', {})
        lines.append("### Genre Comparison (Fiction - Non Fiction)")
        lines.append(f"- **Rating Difference**: {comp.get('rating_diff', 0):+.2f}")
        lines.append(f"- **Price Difference**: ${comp.get('price_diff', 0):+.2f}")
        lines.append(f"- **Reviews Difference**: {comp.get('reviews_diff', 0):+.0f}")
        lines.append(f"- **Count Difference**: {comp.get('count_diff', 0):+,}")
        lines.append("")
        
        # Author Insights
        lines.append("## Top Authors")
        author_insights = insights.get('author_insights', {})
        lines.append(f"- **Total Unique Authors**: {author_insights.get('total_unique_authors', 'N/A'):,}")
        lines.append(f"- **Avg Books per Author**: {author_insights.get('avg_books_per_author', 0):.1f}")
        lines.append(f"- **Max Books by Author**: {author_insights.get('max_books_by_author', 0)}")
        lines.append("")
        
        lines.append("### Top 10 by Book Count")
        lines.append("| Rank | Author | Books | Avg Rating |")
        lines.append("|------|--------|-------|------------|")
        for i, author in enumerate(author_insights.get('top_by_count', [])[:10], 1):
            lines.append(f"| {i} | {author['author']} | {author['book_count']} | {author['avg_rating']:.2f} |")
        lines.append("")
        
        lines.append("### Top 10 by Average Rating (min 2 books)")
        lines.append("| Rank | Author | Books | Avg Rating |")
        lines.append("|------|--------|-------|------------|")
        for i, author in enumerate(author_insights.get('top_by_rating', [])[:10], 1):
            lines.append(f"| {i} | {author['author']} | {author['book_count']} | {author['avg_rating']:.2f} |")
        lines.append("")
        
        # Correlation Insights
        lines.append("## Correlation Insights")
        corr = insights.get('correlation_insights', {})
        
        rv = corr.get('reviews_vs_rating', {})
        if rv:
            sig = " *" if rv.get('significant') else ""
            lines.append(f"- **Reviews vs Rating**: r={rv.get('correlation', 0):.3f}, p={rv.get('p_value', 1):.4f}{sig}")
        
        pv = corr.get('price_vs_rating', {})
        if pv:
            sig = " *" if pv.get('significant') else ""
            lines.append(f"- **Price vs Rating**: r={pv.get('correlation', 0):.3f}, p={pv.get('p_value', 1):.4f}{sig}")
        
        pr = corr.get('price_vs_reviews', {})
        if pr:
            sig = " *" if pr.get('significant') else ""
            lines.append(f"- **Price vs Reviews**: r={pr.get('correlation', 0):.3f}, p={pr.get('p_value', 1):.4f}{sig}")
        lines.append("")
        
        overall = corr.get('overall', {})
        top_corrs = overall.get('top_correlations', [])
        if top_corrs:
            lines.append("### Top Correlations with Rating")
            lines.append("| Rank | Variable | Correlation | P-Value | Significant |")
            lines.append("|------|----------|-------------|---------|-------------|")
            for i, c in enumerate(top_corrs[:5], 1):
                sig = "✓" if c.get('significant') else "✗"
                lines.append(f"| {i} | {c['variable']} | {c['correlation']:.3f} | {c['p_value']:.4f} | {sig} |")
            lines.append("")
        
        # Trend Insights
        lines.append("## Trend Insights")
        trends = insights.get('trend_insights', {})
        yearly = trends.get('yearly_trends', {})
        
        if yearly.get('year_over_year_changes'):
            lines.append("### Year-over-Year Changes")
            lines.append("| Period | Books % Change | Rating Change | Price Change | Reviews Change |")
            lines.append("|--------|----------------|---------------|--------------|----------------|")
            for period, change in yearly['year_over_year_changes'].items():
                lines.append(f"| {period} | {change['count_pct_change']:+.1f}% | {change['rating_change']:+.3f} | ${change['price_change']:+.2f} | {change['reviews_change']:+.0f} |")
            lines.append("")
        
        # Outlier Insights
        lines.append("## Outlier Analysis")
        outliers = insights.get('outlier_insights', {})
        lines.append("| Variable | IQR Outliers | IQR % | Z-Score Outliers | Z-Score % |")
        lines.append("|----------|--------------|-------|------------------|-----------|")
        for col in ['price', 'user_rating', 'reviews']:
            o = outliers.get(col, {})
            iqr = o.get('iqr_outliers', {})
            zscore = o.get('zscore_outliers', {})
            lines.append(f"| {col.capitalize()} | {iqr.get('count', 0)} | {iqr.get('percentage', 0):.1f}% | {zscore.get('count', 0)} | {zscore.get('percentage', 0):.1f}% |")
        lines.append("")
        
        lines.append("---")
        lines.append("*End of Report*")

        return "\n".join(lines)

