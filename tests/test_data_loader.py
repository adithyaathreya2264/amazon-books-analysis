"""
Unit tests for the amazon_books_analysis package.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from amazon_books_analysis.data.loader import DataLoader
from amazon_books_analysis.config.loader import load_config


@pytest.fixture
def sample_data():
    """Create sample book data for testing, using the same raw column
    names the real dataset (and config.yaml's column_mapping) expects."""
    return pd.DataFrame({
        'Name': ['Book 1', 'Book 2', 'Book 3', 'Book 4'],
        'Author': ['Author A', 'Author B', 'Author A', 'Author C'],
        'User Rating': [4.5, 4.2, 4.8, 3.9],
        'Reviews': [1000, 2000, 500, 3000],
        'Price': [19.99, 24.99, 14.99, 29.99],
        'Year': [2019, 2020, 2019, 2021],
        'Genre': ['Fiction', 'Non Fiction', 'Fiction', 'Non Fiction']
    })


@pytest.fixture
def temp_csv(sample_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def config():
    """Create a test configuration file matching the real Config schema."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
data:
  path: "data/amazon_books.csv"
  encoding: "utf-8"
  column_mapping:
    "Name": "name"
    "Author": "author"
    "User Rating": "user_rating"
    "Reviews": "reviews"
    "Price": "price"
    "Year": "year"
    "Genre": "genre"

output:
  output_dir: "output"
  figures_dir: "output/figures"
  reports_dir: "output/reports"
  dpi: 150
  formats: ["png"]
  font_size: 10
  figsize: [10, 6]
  style: "whitegrid"
  color_palette: "husl"

analysis:
  top_authors_count: 10
  price_bins: 20
  correlation_method: "pearson"
  significance_level: 0.05
  genre_order: ["Fiction", "Non Fiction"]
  year_range: [2000, 2030]

visualization:
  bar_chart: {}
  box_plot: {}
  scatter_plot: {}
  line_chart: {}

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "output/analysis.log"
  console: true

pipeline:
  questions: [1, 2, 3, 4, 5, 6, 7]
  generate_visualizations: true
  generate_insights: true
  save_cleaned_data: false
""")
    yield f.name
    os.unlink(f.name)


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_csv(self, temp_csv, config):
        """Test loading a CSV file."""
        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert list(df.columns) == ['Name', 'Author', 'User Rating', 'Reviews', 'Price', 'Year', 'Genre']

    def test_load_nonexistent_file(self, config):
        """Test loading a non-existent file raises FileNotFoundError."""
        loader = DataLoader(load_config(config))

        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent.csv")

    def test_get_shape_and_columns(self, temp_csv, config):
        """Test the loader's convenience accessors after loading."""
        loader = DataLoader(load_config(config))
        loader.load_data(temp_csv)

        assert loader.get_shape() == (4, 7)
        assert 'Genre' in loader.get_columns()


class TestDataCleaner:
    """Tests for DataCleaner class."""

    def test_clean_data(self, temp_csv, config):
        """Test data cleaning."""
        from amazon_books_analysis.data.cleaner import DataCleaner

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)

        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(df)

        assert isinstance(df_clean, pd.DataFrame)
        assert len(df_clean) == 4
        # Columns should now be renamed to snake_case
        assert list(df_clean.columns) == ['name', 'author', 'user_rating', 'reviews', 'price', 'year', 'genre']
        # Check that numeric columns are numeric
        assert pd.api.types.is_numeric_dtype(df_clean['user_rating'])
        assert pd.api.types.is_numeric_dtype(df_clean['reviews'])
        assert pd.api.types.is_numeric_dtype(df_clean['price'])
        assert pd.api.types.is_numeric_dtype(df_clean['year'])

    def test_clean_data_handles_missing_values(self, config):
        """Test cleaning fills missing values instead of dropping rows."""
        from amazon_books_analysis.data.cleaner import DataCleaner

        data_with_na = pd.DataFrame({
            'name': ['Book 1', 'Book 2', 'Book 3'],
            'author': ['Author A', 'Author B', 'Author C'],
            'user_rating': [4.5, np.nan, 4.8],
            'reviews': [1000, 2000, np.nan],
            'price': [19.99, 24.99, 14.99],
            'year': [2019, 2020, 2021],
            'genre': ['Fiction', 'Non Fiction', 'Fiction']
        })

        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(data_with_na)

        # Missing values should be filled with the median, not dropped
        assert len(df_clean) == 3
        assert df_clean['user_rating'].notna().all()
        assert df_clean['reviews'].notna().all()

    def test_clean_data_does_not_clip_outliers(self, config):
        """Outlier detection must see the original, uncensored values."""
        from amazon_books_analysis.data.cleaner import DataCleaner

        data = pd.DataFrame({
            'name': [f'Book {i}' for i in range(6)],
            'author': [f'Author {i}' for i in range(6)],
            'user_rating': [4.5] * 6,
            'reviews': [100, 110, 105, 95, 108, 100000],  # 100000 is an extreme outlier
            'price': [10, 11, 9, 10, 12, 10],
            'year': [2019] * 6,
            'genre': ['Fiction'] * 6
        })

        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(data)

        # The extreme value must survive cleaning unmodified
        assert df_clean['reviews'].max() == 100000


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_raw_data_valid(self, temp_csv, config):
        """Test validation of valid raw data."""
        from amazon_books_analysis.data.validator import DataValidator

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)

        validator = DataValidator(load_config(config))
        result = validator.validate_raw_data(df)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_raw_data_missing_columns(self, config):
        """Test validation fails for missing columns."""
        from amazon_books_analysis.data.validator import DataValidator

        bad_data = pd.DataFrame({
            'Name': ['Book 1'],
            'Author': ['Author A'],
            # Missing User Rating, Reviews, Price, Year, Genre
        })

        validator = DataValidator(load_config(config))
        result = validator.validate_raw_data(bad_data)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_cleaned_data(self, temp_csv, config):
        """Test validation of cleaned data."""
        from amazon_books_analysis.data.validator import DataValidator
        from amazon_books_analysis.data.cleaner import DataCleaner

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)

        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(df)

        validator = DataValidator(load_config(config))
        result = validator.validate_cleaned_data(df_clean)

        assert result.is_valid is True


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""

    def _get_clean_df(self, temp_csv, config):
        from amazon_books_analysis.data.cleaner import DataCleaner

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        return cleaner.clean_data(df)

    def test_get_descriptive_statistics(self, temp_csv, config):
        """Test descriptive statistics computation."""
        from amazon_books_analysis.analysis.statistical import StatisticalAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = StatisticalAnalyzer(load_config(config))
        stats = analyzer.get_descriptive_statistics(
            df_clean, columns=['user_rating', 'reviews', 'price', 'year']
        )

        assert 'user_rating' in stats
        assert 'reviews' in stats
        assert 'price' in stats
        assert 'year' in stats

        for col_stats in stats.values():
            assert 'mean' in col_stats
            assert 'median' in col_stats
            assert 'std' in col_stats
            assert 'min' in col_stats
            assert 'max' in col_stats

    def test_analyze_by_genre(self, temp_csv, config):
        """Test genre analysis."""
        from amazon_books_analysis.analysis.statistical import StatisticalAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = StatisticalAnalyzer(load_config(config))
        genre_stats = analyzer.analyze_by_genre(df_clean)

        assert 'Fiction' in genre_stats
        assert 'Non Fiction' in genre_stats

        for genre_data in genre_stats.values():
            assert 'count' in genre_data
            assert 'user_rating' in genre_data
            assert 'price' in genre_data

    def test_analyze_top_authors(self, temp_csv, config):
        """Test top authors analysis."""
        from amazon_books_analysis.analysis.statistical import StatisticalAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = StatisticalAnalyzer(load_config(config))
        author_stats = analyzer.analyze_top_authors(df_clean, top_n=2)

        assert len(author_stats) <= 2
        for author_stat in author_stats:
            assert 'author' in author_stat
            assert 'book_count' in author_stat
            assert 'avg_rating' in author_stat

    def test_detect_outliers_iqr(self, config):
        """Test IQR outlier detection returns the outlier rows."""
        from amazon_books_analysis.analysis.statistical import StatisticalAnalyzer

        data = pd.DataFrame({
            'name': [f'Book {i}' for i in range(6)],
            'author': [f'Author {i}' for i in range(6)],
            'user_rating': [4.5] * 6,
            'reviews': [100, 110, 105, 95, 108, 100000],
            'price': [10, 11, 9, 10, 12, 10],
            'year': [2019] * 6,
            'genre': ['Fiction'] * 6
        })

        analyzer = StatisticalAnalyzer(load_config(config))
        outliers = analyzer.detect_outliers_iqr(data, 'reviews')

        assert len(outliers) == 1
        assert outliers.iloc[0]['reviews'] == 100000

    def test_compare_groups_mannwhitney(self, temp_csv, config):
        """Test the Mann-Whitney U test wrapper."""
        from amazon_books_analysis.analysis.statistical import StatisticalAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)
        fiction = df_clean[df_clean['genre'] == 'Fiction']['user_rating']
        nonfiction = df_clean[df_clean['genre'] == 'Non Fiction']['user_rating']

        analyzer = StatisticalAnalyzer(load_config(config))
        result = analyzer.compare_groups_mannwhitney(fiction, nonfiction)

        assert 'p_value' in result
        assert 'significant' in result
        assert 'u_statistic' in result

    def test_test_trend_significance(self, temp_csv, config):
        """Test the linear trend significance test."""
        from amazon_books_analysis.analysis.statistical import StatisticalAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = StatisticalAnalyzer(load_config(config))
        result = analyzer.test_trend_significance(df_clean, 'year', 'user_rating')

        assert 'slope' in result
        assert 'p_value' in result
        assert 'significant' in result


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzerExtended class."""

    def _get_clean_df(self, temp_csv, config):
        from amazon_books_analysis.data.cleaner import DataCleaner

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        return cleaner.clean_data(df)

    def test_get_top_correlations(self, temp_csv, config):
        """Test top correlations computation."""
        from amazon_books_analysis.analysis.correlation import CorrelationAnalyzerExtended

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = CorrelationAnalyzerExtended(load_config(config))
        result = analyzer.get_top_correlations(df_clean, 'user_rating', top_n=3)

        top_correlations = result['top_correlations']
        assert len(top_correlations) <= 3
        for corr in top_correlations:
            assert 'variable' in corr
            assert 'correlation' in corr
            assert 'p_value' in corr
            assert 'significant' in corr

    def test_analyze_reviews_vs_rating(self, temp_csv, config):
        """Test reviews vs rating analysis."""
        from amazon_books_analysis.analysis.correlation import CorrelationAnalyzerExtended

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = CorrelationAnalyzerExtended(load_config(config))
        result = analyzer.analyze_reviews_vs_rating(df_clean)

        assert 'correlation' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'by_genre' in result
        assert 'Fiction' in result['by_genre']
        assert 'Non Fiction' in result['by_genre']


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer class."""

    def _get_clean_df(self, temp_csv, config):
        from amazon_books_analysis.data.cleaner import DataCleaner

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        return cleaner.clean_data(df)

    def test_analyze_yearly_trends(self, temp_csv, config):
        """Test yearly trends analysis."""
        from amazon_books_analysis.analysis.trends import TrendAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = TrendAnalyzer(load_config(config))
        trends = analyzer.analyze_yearly_trends(df_clean)

        assert 'yearly_metrics' in trends
        assert 'year_over_year_changes' in trends
        assert 'year_range' in trends

    def test_analyze_genre_trends(self, temp_csv, config):
        """Test genre trends analysis."""
        from amazon_books_analysis.analysis.trends import TrendAnalyzer

        df_clean = self._get_clean_df(temp_csv, config)

        analyzer = TrendAnalyzer(load_config(config))
        trends = analyzer.analyze_genre_trends(df_clean)

        assert 'Fiction' in trends['genre_trends']
        assert 'Non Fiction' in trends['genre_trends']


class TestInsightGenerator:
    """Tests for InsightGenerator class."""

    def test_generate_all_insights(self, temp_csv, config):
        """Test insight generation."""
        from amazon_books_analysis.data.cleaner import DataCleaner
        from amazon_books_analysis.insights.generator import InsightGenerator

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(df)

        generator = InsightGenerator(load_config(config))
        insights = generator.generate_all_insights(df_clean)

        assert 'data_summary' in insights
        assert 'key_findings' in insights
        assert 'correlation_insights' in insights
        assert 'trend_insights' in insights
        assert 'genre_insights' in insights
        assert 'author_insights' in insights
        assert 'outlier_insights' in insights
        assert 'statistical_test_insights' in insights


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def _get_insights(self, temp_csv, config):
        from amazon_books_analysis.data.cleaner import DataCleaner
        from amazon_books_analysis.insights.generator import InsightGenerator

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(df)

        generator = InsightGenerator(load_config(config))
        return generator.generate_all_insights(df_clean)

    def test_generate_text_report(self, temp_csv, config):
        """Test text report generation."""
        from amazon_books_analysis.insights.reporter import ReportGenerator

        insights = self._get_insights(temp_csv, config)

        reporter = ReportGenerator(load_config(config))
        reports = reporter.generate_all_reports(insights)

        assert 'text' in reports
        assert Path(reports['text']).exists()

        content = Path(reports['text']).read_text(encoding='utf-8')
        assert 'AMAZON BOOKS ANALYSIS - INSIGHTS REPORT' in content
        assert 'DATA SUMMARY' in content
        assert 'KEY FINDINGS' in content

    def test_generate_json_report(self, temp_csv, config):
        """Test JSON report generation."""
        from amazon_books_analysis.insights.reporter import ReportGenerator
        import json

        insights = self._get_insights(temp_csv, config)

        reporter = ReportGenerator(load_config(config))
        reports = reporter.generate_all_reports(insights)

        assert 'json' in reports
        assert Path(reports['json']).exists()

        with open(reports['json'], encoding='utf-8') as f:
            data = json.load(f)

        assert 'data_summary' in data
        assert 'key_findings' in data


class TestPlotGenerator:
    """Tests for PlotGenerator class."""

    def test_generate_all_plots(self, temp_csv, config):
        """Test plot generation."""
        from amazon_books_analysis.data.cleaner import DataCleaner
        from amazon_books_analysis.visualization.plots import PlotGenerator

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(df)

        generator = PlotGenerator(load_config(config))
        figures = generator.generate_all_plots(df_clean)

        assert len(figures) > 0

        expected_plots = [
            'genre_distribution',
            'price_distribution',
            'rating_distribution',
            'reviews_distribution',
            'price_vs_rating',
            'reviews_vs_rating',
            'yearly_trends',
            'rating_by_genre',
            'top_authors',
            'yearly_trends_by_genre'
        ]

        for plot_name in expected_plots:
            assert plot_name in figures, f"Missing plot: {plot_name}"


class TestFigureSaver:
    """Tests for FigureSaver class."""

    def test_save_figures(self, temp_csv, config):
        """Test figure saving."""
        from amazon_books_analysis.data.cleaner import DataCleaner
        from amazon_books_analysis.visualization.plots import PlotGenerator
        from amazon_books_analysis.visualization.saver import FigureSaver

        loader = DataLoader(load_config(config))
        df = loader.load_data(temp_csv)
        cleaner = DataCleaner(load_config(config))
        df_clean = cleaner.clean_data(df)

        generator = PlotGenerator(load_config(config))
        figures = generator.generate_all_plots(df_clean)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_config(config)
            cfg.output.figures_dir = tmpdir

            saver = FigureSaver(cfg)
            saved_files = saver.save_all_figures(figures)

            total_files = sum(len(v) for v in saved_files.values())
            assert total_files > 0

            for format_files in saved_files.values():
                for filepath in format_files:
                    assert Path(filepath).exists()


class TestAnalysisPipeline:
    """Tests for the high-level AnalysisPipeline."""

    def test_full_pipeline(self, temp_csv, config):
        """Test the full analysis pipeline."""
        from amazon_books_analysis.pipeline import AnalysisPipeline

        cfg = load_config(config)
        cfg.output.figures_dir = tempfile.mkdtemp()
        cfg.output.reports_dir = tempfile.mkdtemp()

        pipeline = AnalysisPipeline(cfg)
        results = pipeline.run_full_pipeline(temp_csv)

        assert 'descriptive_stats' in results
        assert 'genre_analysis' in results
        assert 'author_analysis' in results
        assert 'correlations' in results
        assert 'insights' in results
        assert 'figures' in results
        assert 'reports' in results

        summary = pipeline.get_summary()
        assert 'ANALYSIS PIPELINE SUMMARY' in summary
        assert 'Dataset:' in summary
        assert 'Key Findings:' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
