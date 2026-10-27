"""
Main CLI entry point for Amazon Books Analysis.

This module is a thin argument-parsing wrapper around AnalysisPipeline;
all orchestration logic lives there so the CLI and the library API can't
drift out of sync with each other.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from ..config.loader import load_config
from ..pipeline import AnalysisPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def print_summary(results: Dict[str, Any], quiet: bool) -> None:
    """Print a summary of the analysis results."""
    if quiet:
        return

    insights = results.get('insights', {})

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    summary = insights.get('data_summary', {})
    if summary:
        print(f"\nDataset: {summary.get('total_books', 'N/A'):,} books")
        print(f"Years: {summary.get('year_range', ['N/A', 'N/A'])[0]} - {summary.get('year_range', ['N/A', 'N/A'])[1]}")
        print(f"Genres: {', '.join(summary.get('genres', []))}")
        print(f"Authors: {summary.get('unique_authors', 'N/A'):,}")

    key_findings = insights.get('key_findings', [])
    if key_findings:
        print("\nKey Findings:")
        for i, finding in enumerate(key_findings, 1):
            print(f"  {i}. {finding}")

    top_corrs = insights.get('correlation_insights', {}).get('overall', {}).get('top_correlations', [])
    if top_corrs:
        print("\nTop Correlations with Rating:")
        for i, c in enumerate(top_corrs[:3], 1):
            sig = " *" if c.get('significant') else ""
            print(f"  {i}. {c['variable']}: r={c['correlation']:.3f}, p={c['p_value']:.4f}{sig}")

    if 'reports' in results:
        print("\nReports Generated:")
        for fmt, path in results['reports'].items():
            print(f"  {fmt.upper()}: {path}")

    if 'figures' in results:
        total_figs = sum(len(v) for v in results['figures'].values())
        print(f"\nFigures Saved: {total_figs} files")

    print("\n" + "=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Amazon Books Analysis - Analyze Amazon bestseller books data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default settings
  python -m amazon_books_analysis.cli

  # Run with custom data file
  python -m amazon_books_analysis.cli --data data/my_books.csv

  # Run with custom config
  python -m amazon_books_analysis.cli --config config.yaml

  # Generate only plots
  python -m amazon_books_analysis.cli --plots-only

  # Generate only reports
  python -m amazon_books_analysis.cli --reports-only

  # Verbose output
  python -m amazon_books_analysis.cli -v

  # Quiet mode (warnings only)
  python -m amazon_books_analysis.cli -q
        """
    )

    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                         help='Path to configuration file (default: config.yaml)')
    parser.add_argument('-d', '--data', type=str,
                         help='Path to input data CSV file (overrides config)')
    parser.add_argument('-o', '--output', type=str,
                         help='Output directory (overrides config)')

    parser.add_argument('--plots-only', action='store_true',
                         help='Generate only visualizations, skip statistical analysis')
    parser.add_argument('--reports-only', action='store_true',
                         help='Generate only reports (runs analysis and insights first)')
    parser.add_argument('--no-plots', action='store_true',
                         help='Skip visualization generation')
    parser.add_argument('--no-reports', action='store_true',
                         help='Skip report generation')

    parser.add_argument('-v', '--verbose', action='store_true',
                         help='Enable verbose (DEBUG) logging')
    parser.add_argument('-q', '--quiet', action='store_true',
                         help='Suppress INFO logs, show only WARNING and above')

    parser.add_argument('--correlations-only', action='store_true',
                         help='Run only correlation analysis')
    parser.add_argument('--trends-only', action='store_true',
                         help='Run only trend analysis')
    parser.add_argument('--stats-only', action='store_true',
                         help='Run only statistical analysis')

    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the pipeline according to the parsed CLI flags."""
    config = load_config(args.config)

    if args.output:
        output_root = Path(args.output)
        config.output.output_dir = str(output_root)
        config.output.figures_dir = str(output_root / "figures")
        config.output.reports_dir = str(output_root / "reports")

    pipeline = AnalysisPipeline(config)
    df = pipeline.load_data(args.data)

    if args.correlations_only:
        logger.info("Running correlation analysis only")
        corr = pipeline.corr_analyzer
        return {
            'correlations': corr.get_top_correlations(df, 'user_rating', top_n=10),
            'reviews_vs_rating': corr.analyze_reviews_vs_rating(df),
            'price_vs_rating': corr.analyze_price_vs_rating(df),
            'price_vs_reviews': corr.analyze_price_vs_reviews(df),
            'yearly_correlations': corr.analyze_yearly_correlations(df),
        }

    if args.trends_only:
        logger.info("Running trend analysis only")
        trend = pipeline.trend_analyzer
        return {
            'yearly_trends': trend.analyze_yearly_trends(df),
            'genre_trends': trend.analyze_genre_trends(df),
            'top_authors_trends': trend.analyze_top_authors_trends(df, top_n=10),
            'rating_distribution_trends': trend.analyze_rating_distribution_trends(df),
            'price_distribution_trends': trend.analyze_price_distribution_trends(df),
        }

    if args.stats_only:
        logger.info("Running statistical analysis only")
        stat = pipeline.stat_analyzer
        return {
            'descriptive_stats': stat.get_descriptive_statistics(df),
            'genre_analysis': stat.analyze_by_genre(df),
            'author_analysis': stat.analyze_top_authors(df, top_n=20),
        }

    if args.plots_only:
        logger.info("Generating plots only")
        return {'figures': pipeline.generate_visualizations()}

    if args.reports_only:
        logger.info("Generating reports only")
        insights = pipeline.generate_insights()
        reports = pipeline.generate_reports(insights)
        return {'insights': insights, 'reports': reports}

    # Full analysis
    pipeline.run_analysis()
    pipeline.generate_insights()

    if not args.no_plots:
        pipeline.generate_visualizations()
    if not args.no_reports:
        pipeline.generate_reports()

    return pipeline.results


def main() -> int:
    """Main CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.verbose, args.quiet)

    try:
        logger.info(f"Loading configuration from {args.config}")
        results = run(args)

        print_summary(results, args.quiet)

        logger.info("Analysis completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
