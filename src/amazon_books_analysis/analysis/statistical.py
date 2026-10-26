"""
Statistical analysis module with significance testing for Amazon Books Analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats  # pyrefly: ignore[untyped-import]
from scipy.stats import (
    ttest_ind, ttest_rel, f_oneway, chi2_contingency, pearsonr, spearmanr,
    kendalltau, mannwhitneyu, linregress, zscore as scipy_zscore,  # pyrefly: ignore[untyped-import]
)
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..config.loader import load_config, Config

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis with significance testing."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.analysis_config = self.config.analysis
        self.significance_level = self.analysis_config.significance_level
    
    def compare_ratings_by_genre(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare user ratings between Fiction and Non-Fiction using t-test.
        
        Args:
            df: Cleaned DataFrame with 'genre' and 'user_rating' columns.
            
        Returns:
            Dictionary with test results and statistics.
        """
        logger.info("Comparing ratings by genre using t-test")
        
        fiction_ratings = df[df['genre'] == 'Fiction']['user_rating']
        nonfiction_ratings = df[df['genre'] == 'Non Fiction']['user_rating']
        
        # Descriptive statistics
        fiction_stats = {
            'count': len(fiction_ratings),
            'mean': fiction_ratings.mean(),
            'std': fiction_ratings.std(),
            'median': fiction_ratings.median(),
            'min': fiction_ratings.min(),
            'max': fiction_ratings.max()
        }
        
        nonfiction_stats = {
            'count': len(nonfiction_ratings),
            'mean': nonfiction_ratings.mean(),
            'std': nonfiction_ratings.std(),
            'median': nonfiction_ratings.median(),
            'min': nonfiction_ratings.min(),
            'max': nonfiction_ratings.max()
        }
        
        # Independent t-test (assuming unequal variances - Welch's t-test)
        t_stat, p_value = ttest_ind(fiction_ratings, nonfiction_ratings, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((fiction_ratings.var() + nonfiction_ratings.var()) / 2)
        cohens_d = (fiction_ratings.mean() - nonfiction_ratings.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for mean difference
        mean_diff = fiction_ratings.mean() - nonfiction_ratings.mean()
        se_diff = np.sqrt(fiction_ratings.var()/len(fiction_ratings) + nonfiction_ratings.var()/len(nonfiction_ratings))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        result = {
            'test': 'Welch\'s t-test (independent samples)',
            'fiction_stats': fiction_stats,
            'nonfiction_stats': nonfiction_stats,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'significance_level': self.significance_level,
            'mean_difference': float(mean_diff),
            'cohens_d': float(cohens_d),
            'confidence_interval_95': [float(ci_lower), float(ci_upper)],
            'interpretation': self._interpret_ttest(p_value, cohens_d, mean_diff)
        }
        
        logger.info(f"T-test result: t={t_stat:.4f}, p={p_value:.6f}, significant={result['significant']}")
        return result
    
    def compare_prices_by_genre(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare prices between Fiction and Non-Fiction using t-test.
        
        Args:
            df: Cleaned DataFrame with 'genre' and 'price' columns.
            
        Returns:
            Dictionary with test results and statistics.
        """
        logger.info("Comparing prices by genre using t-test")
        
        fiction_prices = df[df['genre'] == 'Fiction']['price']
        nonfiction_prices = df[df['genre'] == 'Non Fiction']['price']
        
        # Descriptive statistics
        fiction_stats = {
            'count': len(fiction_prices),
            'mean': fiction_prices.mean(),
            'std': fiction_prices.std(),
            'median': fiction_prices.median(),
            'min': fiction_prices.min(),
            'max': fiction_prices.max()
        }
        
        nonfiction_stats = {
            'count': len(nonfiction_prices),
            'mean': nonfiction_prices.mean(),
            'std': nonfiction_prices.std(),
            'median': nonfiction_prices.median(),
            'min': nonfiction_prices.min(),
            'max': nonfiction_prices.max()
        }
        
        # Independent t-test (Welch's)
        t_stat, p_value = ttest_ind(fiction_prices, nonfiction_prices, equal_var=False)
        
        # Effect size
        pooled_std = np.sqrt((fiction_prices.var() + nonfiction_prices.var()) / 2)
        cohens_d = (fiction_prices.mean() - nonfiction_prices.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval
        mean_diff = fiction_prices.mean() - nonfiction_prices.mean()
        se_diff = np.sqrt(fiction_prices.var()/len(fiction_prices) + nonfiction_prices.var()/len(nonfiction_prices))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        result = {
            'test': 'Welch\'s t-test (independent samples)',
            'fiction_stats': fiction_stats,
            'nonfiction_stats': nonfiction_stats,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'significance_level': self.significance_level,
            'mean_difference': float(mean_diff),
            'cohens_d': float(cohens_d),
            'confidence_interval_95': [float(ci_lower), float(ci_upper)],
            'interpretation': self._interpret_ttest(p_value, cohens_d, mean_diff)
        }
        
        logger.info(f"Price t-test result: t={t_stat:.4f}, p={p_value:.6f}, significant={result['significant']}")
        return result
    
    def anova_rating_by_year(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform one-way ANOVA to test if ratings differ across years.
        
        Args:
            df: Cleaned DataFrame with 'year' and 'user_rating' columns.
            
        Returns:
            Dictionary with ANOVA results.
        """
        logger.info("Performing ANOVA for ratings across years")
        
        # Group ratings by year
        years = sorted(df['year'].unique())
        groups = [df[df['year'] == year]['user_rating'].values for year in years]
        
        # Descriptive statistics per year
        year_stats = {}
        for year in years:
            ratings = df[df['year'] == year]['user_rating']
            year_stats[int(year)] = {
                'count': len(ratings),
                'mean': float(ratings.mean()),
                'std': float(ratings.std()),
                'median': float(ratings.median())
            }
        
        # One-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Effect size (eta-squared)
        # Calculate sum of squares
        all_ratings = np.concatenate(groups)
        grand_mean = all_ratings.mean()
        
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_ratings)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = {
            'test': 'One-way ANOVA',
            'year_stats': year_stats,
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'significance_level': self.significance_level,
            'eta_squared': float(eta_squared),
            'interpretation': self._interpret_anova(p_value, eta_squared)
        }
        
        logger.info(f"ANOVA result: F={f_stat:.4f}, p={p_value:.6f}, significant={result['significant']}")
        return result
    
    def chi2_genre_year_association(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test association between genre and year using chi-square test.
        
        Args:
            df: Cleaned DataFrame with 'genre' and 'year' columns.
            
        Returns:
            Dictionary with chi-square test results.
        """
        logger.info("Performing chi-square test for genre-year association")
        
        # Create contingency table
        contingency = pd.crosstab(df['genre'], df['year'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Effect size (Cramer's V)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
        
        result = {
            'test': 'Chi-square test of independence',
            'contingency_table': contingency.to_dict(),
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < self.significance_level,
            'significance_level': self.significance_level,
            'cramers_v': float(cramers_v),
            'interpretation': self._interpret_chi2(p_value, cramers_v)
        }
        
        logger.info(f"Chi-square result: χ²={chi2:.4f}, p={p_value:.6f}, significant={result['significant']}")
        return result
    
    def get_descriptive_statistics(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive descriptive statistics.
        
        Args:
            df: DataFrame to analyze.
            columns: List of columns to analyze. If None, uses all numeric columns.
            
        Returns:
            Dictionary with descriptive statistics.
        """
        logger.info("Computing descriptive statistics")
        
        analysis_columns: List[str] = (
            columns
            if columns is not None
            else df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        stats = {}
        for col in analysis_columns:
            if col in df.columns:
                data = df[col].dropna()
                stats[col] = {
                    'count': int(data.count()),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'var': float(data.var()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'missing': int(df[col].isnull().sum())
                }
        
        return stats

    def analyze_by_genre(self, df: pd.DataFrame,
                          numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute per-genre descriptive statistics.

        Args:
            df: Cleaned DataFrame with a 'genre' column.
            numeric_columns: Numeric columns to summarize. Defaults to
                ['user_rating', 'price', 'reviews'].

        Returns:
            Dictionary keyed by genre, each containing 'count' and one
            descriptive-statistics dict per numeric column.
        """
        logger.info("Analyzing statistics by genre")

        if numeric_columns is None:
            numeric_columns = ['user_rating', 'price', 'reviews']
            numeric_columns = [c for c in numeric_columns if c in df.columns]

        results: Dict[str, Any] = {}
        for genre in df['genre'].unique():
            genre_df = df[df['genre'] == genre]
            genre_result: Dict[str, Any] = {'count': len(genre_df)}
            for col in numeric_columns:
                genre_result[col] = self.get_descriptive_statistics(genre_df, columns=[col])[col]
            results[genre] = genre_result

        return results

    def analyze_top_authors(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Compute per-author statistics for the most prolific authors.

        Args:
            df: Cleaned DataFrame with 'author', 'user_rating', 'price', 'reviews' columns.
            top_n: Number of top authors (by book count) to return.

        Returns:
            List of dicts sorted by book_count descending, one per author.
        """
        logger.info(f"Analyzing top {top_n} authors")

        author_stats = df.groupby('author').agg(
            book_count=('author', 'count'),
            avg_rating=('user_rating', 'mean'),
            avg_price=('price', 'mean'),
            avg_reviews=('reviews', 'mean'),
            total_reviews=('reviews', 'sum'),
        ).reset_index()

        author_stats = author_stats.sort_values('book_count', ascending=False).head(top_n)

        return [
            {
                'author': row['author'],
                'book_count': int(row['book_count']),
                'avg_rating': float(row['avg_rating']),
                'avg_price': float(row['avg_price']),
                'avg_reviews': float(row['avg_reviews']),
                'total_reviews': int(row['total_reviews']),
            }
            for _, row in author_stats.iterrows()
        ]

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outlier rows in a column using the IQR method.

        Args:
            df: DataFrame to analyze.
            column: Column to check for outliers.
            multiplier: IQR multiplier for the bounds (default 1.5).

        Returns:
            Subset of df containing only the outlier rows.
        """
        logger.info(f"Detecting IQR outliers in {column}")

        data = df[column].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    def detect_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outlier rows in a column using the Z-score method.

        Args:
            df: DataFrame to analyze.
            column: Column to check for outliers.
            threshold: Absolute Z-score threshold (default 3.0).

        Returns:
            Subset of df containing only the outlier rows.
        """
        logger.info(f"Detecting Z-score outliers in {column}")

        valid = df[column].notna()
        z_scores = pd.Series(np.nan, index=df.index)
        z_scores[valid] = scipy_zscore(df.loc[valid, column])

        return df[z_scores.abs() > threshold]

    def compare_groups_mannwhitney(self, group1: pd.Series, group2: pd.Series,
                                    group1_name: str = 'group1',
                                    group2_name: str = 'group2') -> Dict[str, Any]:
        """
        Compare two independent groups using the Mann-Whitney U test
        (a non-parametric alternative to the independent t-test).

        Args:
            group1: First sample.
            group2: Second sample.
            group1_name: Label for the first group in the result.
            group2_name: Label for the second group in the result.

        Returns:
            Dictionary with test statistic, p-value, and interpretation.
        """
        logger.info("Comparing groups using Mann-Whitney U test")

        g1 = group1.dropna()
        g2 = group2.dropna()

        u_stat, p_value = mannwhitneyu(g1, g2, alternative='two-sided')

        # Rank-biserial correlation as effect size
        n1, n2 = len(g1), len(g2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

        result = {
            'test': 'Mann-Whitney U test',
            f'{group1_name}_median': float(g1.median()) if len(g1) else None,
            f'{group2_name}_median': float(g2.median()) if len(g2) else None,
            f'{group1_name}_n': n1,
            f'{group2_name}_n': n2,
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'significance_level': self.significance_level,
            'effect_size_rank_biserial': float(effect_size),
        }

        sig = "significant" if result['significant'] else "not significant"
        result['interpretation'] = (
            f"The difference between {group1_name} and {group2_name} is {sig} "
            f"(U={u_stat:.2f}, p={p_value:.6f})."
        )

        logger.info(f"Mann-Whitney result: U={u_stat:.4f}, p={p_value:.6f}, significant={result['significant']}")
        return result

    def test_trend_significance(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """
        Test whether y_col shows a statistically significant linear trend
        against x_col (e.g. rating trend over years) using linear regression.

        Args:
            df: Cleaned DataFrame.
            x_col: Independent variable (typically 'year').
            y_col: Dependent variable to test for a trend.

        Returns:
            Dictionary with slope, p-value, and interpretation.
        """
        logger.info(f"Testing trend significance of {y_col} over {x_col}")

        data = df[[x_col, y_col]].dropna()
        x = data[x_col].astype(float)
        y = data[y_col].astype(float)

        regression = linregress(x, y)

        result = {
            'test': 'Linear regression trend test',
            'x': x_col,
            'y': y_col,
            'slope': float(regression.slope),
            'intercept': float(regression.intercept),
            'r_value': float(regression.rvalue),
            'r_squared': float(regression.rvalue ** 2),
            'p_value': float(regression.pvalue),
            'std_err': float(regression.stderr),
            'significant': bool(regression.pvalue < self.significance_level),
            'significance_level': self.significance_level,
        }

        sig = "significant" if result['significant'] else "not significant"
        direction = "increasing" if regression.slope > 0 else "decreasing"
        result['interpretation'] = (
            f"The trend in {y_col} over {x_col} is {sig} (p={regression.pvalue:.6f}), "
            f"{direction} at a rate of {regression.slope:.4f} per unit of {x_col}."
        )

        logger.info(f"Trend test result: slope={regression.slope:.4f}, p={regression.pvalue:.6f}, "
                    f"significant={result['significant']}")
        return result

    def _interpret_ttest(self, p_value: float, cohens_d: float, mean_diff: float) -> str:
        """Interpret t-test results."""
        sig = "significant" if p_value < self.significance_level else "not significant"
        
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        direction = "higher" if mean_diff > 0 else "lower"
        
        return (f"The difference is {sig} (p={p_value:.6f}). "
                f"Effect size (Cohen's d) is {effect} ({cohens_d:.3f}). "
                f"Fiction has {direction} mean values than Non-Fiction.")
    
    def _interpret_anova(self, p_value: float, eta_squared: float) -> str:
        """Interpret ANOVA results."""
        sig = "significant" if p_value < self.significance_level else "not significant"
        
        if eta_squared < 0.01:
            effect = "negligible"
        elif eta_squared < 0.06:
            effect = "small"
        elif eta_squared < 0.14:
            effect = "medium"
        else:
            effect = "large"
        
        return (f"The effect of year on ratings is {sig} (p={p_value:.6f}). "
                f"Effect size (η²) is {effect} ({eta_squared:.3f}).")
    
    def _interpret_chi2(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square results."""
        sig = "significant" if p_value < self.significance_level else "not significant"
        
        if cramers_v < 0.1:
            effect = "negligible"
        elif cramers_v < 0.3:
            effect = "small"
        elif cramers_v < 0.5:
            effect = "medium"
        else:
            effect = "large"
        
        return (f"The association between genre and year is {sig} (p={p_value:.6f}). "
                f"Effect size (Cramér's V) is {effect} ({cramers_v:.3f}).")


class CorrelationAnalyzer:
    """Analyzes correlations with significance testing."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the correlation analyzer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.analysis_config = self.config.analysis
        self.significance_level = self.analysis_config.significance_level
        self.method = self.analysis_config.correlation_method
    
    def compute_correlations(self, df: pd.DataFrame, 
                           columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute correlation matrix with significance tests.
        
        Args:
            df: DataFrame to analyze.
            columns: List of columns to include. If None, uses all numeric columns.
            
        Returns:
            Dictionary with correlation matrix and p-values.
        """
        logger.info(f"Computing correlations using {self.method} method")
        
        correlation_columns: List[str] = (
            columns
            if columns is not None
            else df.select_dtypes(include=[np.number]).columns.tolist()
        )

        data = df[correlation_columns].dropna()
        
        # Compute correlation matrix
        if self.method == 'pearson':
            corr_matrix = data.corr(method='pearson')
        elif self.method == 'spearman':
            corr_matrix = data.corr(method='spearman')
        elif self.method == 'kendall':
            corr_matrix = data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {self.method}")
        
        # Compute p-values for each pair
        n = len(data)
        p_values = pd.DataFrame(
            index=correlation_columns, columns=correlation_columns, dtype=float
        )

        for i, col1 in enumerate(correlation_columns):
            for j, col2 in enumerate(correlation_columns):
                if i <= j:
                    if self.method == 'pearson':
                        _, p = pearsonr(data[col1], data[col2])
                    elif self.method == 'spearman':
                        _, p = spearmanr(data[col1], data[col2])
                    elif self.method == 'kendall':
                        _, p = kendalltau(data[col1], data[col2])
                    p_values.loc[col1, col2] = p
                    p_values.loc[col2, col1] = p
        
        # Significance matrix
        sig_matrix = p_values < self.significance_level
        
        return {
            'method': self.method,
            'correlation_matrix': corr_matrix.to_dict(),
            'p_values': p_values.to_dict(),
            'significance_matrix': sig_matrix.to_dict(),
            'significance_level': self.significance_level,
            'sample_size': n
        }
    
    def correlation_with_significance(self, df: pd.DataFrame, 
                                      x: str, y: str) -> Dict[str, Any]:
        """
        Compute correlation between two variables with significance test.
        
        Args:
            df: DataFrame containing the data.
            x: First variable name.
            y: Second variable name.
            
        Returns:
            Dictionary with correlation coefficient, p-value, and interpretation.
        """
        logger.info(f"Computing {self.method} correlation between {x} and {y}")
        
        data_x = df[x].dropna()
        data_y = df[y].dropna()
        
        # Align indices
        common_idx = data_x.index.intersection(data_y.index)
        data_x = data_x.loc[common_idx]
        data_y = data_y.loc[common_idx]
        
        if self.method == 'pearson':
            corr, p_value = pearsonr(data_x, data_y)
        elif self.method == 'spearman':
            corr, p_value = spearmanr(data_x, data_y)
        elif self.method == 'kendall':
            corr, p_value = kendalltau(data_x, data_y)
        else:
            raise ValueError(f"Unknown correlation method: {self.method}")
        
        # Confidence interval for Pearson correlation
        if self.method == 'pearson' and len(data_x) > 3 and abs(corr) < 1.0:
            z = np.arctanh(corr)
            se = 1 / np.sqrt(len(data_x) - 3)
            z_lower = z - 1.96 * se
            z_upper = z + 1.96 * se
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
        else:
            ci_lower = ci_upper = None
        
        result = {
            'method': self.method,
            'variable_x': x,
            'variable_y': y,
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'significance_level': self.significance_level,
            'sample_size': len(data_x),
            'confidence_interval_95': (
                [float(ci_lower), float(ci_upper)]
                if ci_lower is not None and ci_upper is not None
                else None
            ),
            'interpretation': self._interpret_correlation(corr, p_value)
        }
        
        logger.info(f"Correlation result: r={corr:.4f}, p={p_value:.6f}, significant={result['significant']}")
        return result
    
    def _interpret_correlation(self, corr: float, p_value: float) -> str:
        """Interpret correlation results."""
        sig = "significant" if p_value < self.significance_level else "not significant"
        
        if abs(corr) < 0.1:
            strength = "negligible"
        elif abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.5:
            strength = "moderate"
        elif abs(corr) < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if corr > 0 else "negative"
        
        return (f"The correlation is {sig} (p={p_value:.6f}). "
                f"It is a {strength} {direction} correlation (r={corr:.3f}).")
