"""
Plot generation module for Amazon Books Analysis.
"""

import matplotlib
# This package only ever renders figures to files (see FigureSaver), never
# to an interactive window, so force the non-interactive Agg backend. This
# also avoids a hard dependency on a working Tk/Tcl installation.
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, cast
import logging
from pathlib import Path

from ..config.loader import Config, load_config

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generates various plots for book data analysis."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the plot generator.

        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.viz_config = self.config.output

        # Set style (config uses seaborn style names, e.g. "whitegrid")
        sns.set_style(self.viz_config.style)
        sns.set_palette(self.viz_config.color_palette)

        # Default figure size
        self.figsize = self.viz_config.figsize
        self.dpi = self.viz_config.dpi
    
    def plot_genre_distribution(self, df: pd.DataFrame, 
                                ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot genre distribution as a bar chart.
        
        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        logger.info("Generating genre distribution plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        genre_counts = df['genre'].value_counts()
        colors = sns.color_palette(self.viz_config.color_palette, len(genre_counts))
        
        bars = ax.bar(genre_counts.index, genre_counts.values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        ax.set_title('Book Distribution by Genre', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Number of Books', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_price_distribution(self, df: pd.DataFrame,
                                ax: Optional[plt.Axes] = None,
                                by_genre: bool = False) -> plt.Figure:
        """
        Plot price distribution as histogram.
        
        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            by_genre: Whether to show separate distributions by genre.
            
        Returns:
            Matplotlib figure.
        """
        logger.info("Generating price distribution plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if by_genre:
            for genre in df['genre'].unique():
                genre_data = df[df['genre'] == genre]['price']
                ax.hist(genre_data, bins=30, alpha=0.6, label=genre, density=True, edgecolor='black', linewidth=0.3)
            ax.legend(title='Genre', fontsize=10)
        else:
            ax.hist(df['price'], bins=30, color=sns.color_palette(self.viz_config.color_palette)[0], 
                   edgecolor='black', linewidth=0.3, alpha=0.7, density=True)
            # Add KDE
            sns.kdeplot(df['price'], ax=ax, color='red', linewidth=2)
        
        ax.set_title('Price Distribution of Books', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Price (USD)', fontsize=12)
        ax.set_ylabel('Density' if by_genre else 'Frequency', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_rating_distribution(self, df: pd.DataFrame,
                                 ax: Optional[plt.Axes] = None,
                                 by_genre: bool = False) -> plt.Figure:
        """
        Plot rating distribution as histogram.
        
        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            by_genre: Whether to show separate distributions by genre.
            
        Returns:
            Matplotlib figure.
        """
        logger.info("Generating rating distribution plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if by_genre:
            for genre in df['genre'].unique():
                genre_data = df[df['genre'] == genre]['user_rating']
                ax.hist(genre_data, bins=20, alpha=0.6, label=genre, density=True, edgecolor='black', linewidth=0.3)
            ax.legend(title='Genre', fontsize=10)
        else:
            ax.hist(df['user_rating'], bins=20, color=sns.color_palette(self.viz_config.color_palette)[1], 
                   edgecolor='black', linewidth=0.3, alpha=0.7, density=True)
            sns.kdeplot(df['user_rating'], ax=ax, color='red', linewidth=2)
        
        ax.set_title('User Rating Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('User Rating (0-5)', fontsize=12)
        ax.set_ylabel('Density' if by_genre else 'Frequency', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig

    def plot_reviews_distribution(self, df: pd.DataFrame,
                                  ax: Optional[plt.Axes] = None,
                                  by_genre: bool = False,
                                  log_scale: bool = True) -> plt.Figure:
        """
        Plot reviews distribution as histogram.

        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            by_genre: Whether to show separate distributions by genre.
            log_scale: Whether to use a log scale for the x-axis.

        Returns:
            Matplotlib figure.
        """
        logger.info("Generating reviews distribution plot")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)

        if by_genre:
            for genre in df['genre'].unique():
                genre_data = df[df['genre'] == genre]['reviews']
                ax.hist(genre_data, bins=30, alpha=0.6, label=genre, density=True, edgecolor='black', linewidth=0.3)
            ax.legend(title='Genre', fontsize=10)
        else:
            ax.hist(df['reviews'], bins=30, color=sns.color_palette(self.viz_config.color_palette, 3)[2],
                   edgecolor='black', linewidth=0.3, alpha=0.7, density=True)

        if log_scale:
            ax.set_xscale('log')

        ax.set_title('Reviews Distribution of Books', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Number of Reviews (log scale)' if log_scale else 'Number of Reviews', fontsize=12)
        ax.set_ylabel('Density' if by_genre else 'Frequency', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_price_vs_rating(self, df: pd.DataFrame,
                             ax: Optional[plt.Axes] = None,
                             by_genre: bool = True) -> plt.Figure:
        """
        Plot price vs rating scatter plot.
        
        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            by_genre: Whether to color by genre.
            
        Returns:
            Matplotlib figure.
        """
        logger.info("Generating price vs rating plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if by_genre:
            for genre in df['genre'].unique():
                genre_data = df[df['genre'] == genre]
                ax.scatter(genre_data['price'], genre_data['user_rating'], 
                          alpha=0.5, label=genre, s=30, edgecolor='white', linewidth=0.3)
            ax.legend(title='Genre', fontsize=10)
        else:
            ax.scatter(df['price'], df['user_rating'], alpha=0.5, s=30, 
                      color=sns.color_palette(self.viz_config.color_palette)[0], edgecolor='white', linewidth=0.3)
        
        # Add trend line
        z = np.polyfit(df['price'], df['user_rating'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['price'].min(), df['price'].max(), 100)
        ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        
        ax.set_title('Price vs User Rating', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Price (USD)', fontsize=12)
        ax.set_ylabel('User Rating', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_reviews_vs_rating(self, df: pd.DataFrame,
                               ax: Optional[plt.Axes] = None,
                               by_genre: bool = True,
                               log_scale: bool = True) -> plt.Figure:
        """
        Plot reviews vs rating scatter plot.
        
        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            by_genre: Whether to color by genre.
            log_scale: Whether to use log scale for reviews.
            
        Returns:
            Matplotlib figure.
        """
        logger.info("Generating reviews vs rating plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if by_genre:
            for genre in df['genre'].unique():
                genre_data = df[df['genre'] == genre]
                ax.scatter(genre_data['reviews'], genre_data['user_rating'], 
                          alpha=0.5, label=genre, s=30, edgecolor='white', linewidth=0.3)
            ax.legend(title='Genre', fontsize=10)
        else:
            ax.scatter(df['reviews'], df['user_rating'], alpha=0.5, s=30,
                      color=sns.color_palette(self.viz_config.color_palette)[1], edgecolor='white', linewidth=0.3)
        
        if log_scale:
            ax.set_xscale('log')
        
        # Add trend line
        if log_scale:
            z = np.polyfit(np.log(df['reviews']), df['user_rating'], 1)
            x_range = np.logspace(np.log10(df['reviews'].min()), np.log10(df['reviews'].max()), 100)
            p = np.poly1d(z)
            ax.plot(x_range, p(np.log(x_range)), "r--", alpha=0.8, linewidth=2)
        else:
            z = np.polyfit(df['reviews'], df['user_rating'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['reviews'].min(), df['reviews'].max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
        
        ax.set_title('Number of Reviews vs User Rating', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Number of Reviews (log scale)' if log_scale else 'Number of Reviews', fontsize=12)
        ax.set_ylabel('User Rating', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_rating_by_genre(self, df: pd.DataFrame,
                             ax: Optional[plt.Axes] = None,
                             plot_type: str = 'box') -> plt.Figure:
        """
        Plot rating distribution by genre (box plot or violin plot).
        
        Args:
            df: Cleaned DataFrame.
            ax: Optional matplotlib axes to plot on.
            plot_type: 'box' or 'violin'.
            
        Returns:
            Matplotlib figure.
        """
        logger.info(f"Generating rating by genre plot ({plot_type})")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        genres = df['genre'].unique()
        data = [df[df['genre'] == g]['user_rating'].values for g in genres]
        
        if plot_type == 'box':
            bp = ax.boxplot(data, patch_artist=True, widths=0.6)
            colors = sns.color_palette(self.viz_config.color_palette, len(genres))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels(genres)
        elif plot_type == 'violin':
            vp = ax.violinplot(data, showmeans=True, showmedians=True)
            colors = sns.color_palette(self.viz_config.color_palette, len(genres))
            bodies = cast(List[Any], vp['bodies'])
            for i, body in enumerate(bodies):
                body.set_facecolor(colors[i])
                body.set_alpha(0.7)
            ax.set_xticks(range(1, len(genres) + 1))
            ax.set_xticklabels(genres)
        
        ax.set_title('User Rating Distribution by Genre', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('User Rating', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_yearly_trends(self, df: pd.DataFrame,
                           metrics: Optional[List[str]] = None,
                           ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot yearly trends for specified metrics.
        
        Args:
            df: Cleaned DataFrame.
            metrics: List of metrics to plot. Default: ['user_rating', 'price', 'reviews'].
            ax: Optional matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        logger.info("Generating yearly trends plot")
        
        if metrics is None:
            metrics = ['user_rating', 'price', 'reviews']
        
        if ax is None:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(self.figsize[0], self.figsize[1] * len(metrics)), dpi=self.dpi)
            if len(metrics) == 1:
                axes = [axes]
        else:
            fig = cast(plt.Figure, ax.figure)
            axes = [ax]
            if len(metrics) > 1:
                logger.warning(
                    "Multiple metrics requested but a single ax was provided; "
                    "only plotting the first metric (%s)", metrics[0]
                )
                metrics = metrics[:1]
        
        years = sorted(df['year'].unique())
        colors = sns.color_palette(self.viz_config.color_palette, len(metrics))
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if len(metrics) > 1 else axes[0]
            yearly_avg = df.groupby('year')[metric].mean()
            
            ax.plot(yearly_avg.index, yearly_avg.values, 'o-', color=colors[i], linewidth=2, markersize=8)
            ax.fill_between(yearly_avg.index, yearly_avg.values, alpha=0.1, color=colors[i])
            
            metric_labels = {
                'user_rating': 'Average User Rating',
                'price': 'Average Price (USD)',
                'reviews': 'Average Number of Reviews'
            }
            
            ax.set_title(f'{metric_labels.get(metric, metric)} Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_yearly_trends_by_genre(self, df: pd.DataFrame,
                                    metric: str = 'user_rating',
                                    ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot the yearly trend of a metric, split out by genre.

        Args:
            df: Cleaned DataFrame.
            metric: Metric to plot ('user_rating', 'price', or 'reviews').
            ax: Optional matplotlib axes to plot on.

        Returns:
            Matplotlib figure.
        """
        logger.info(f"Generating yearly trends by genre plot (metric: {metric})")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)

        genres = df['genre'].unique()
        colors = sns.color_palette(self.viz_config.color_palette, len(genres))

        for i, genre in enumerate(genres):
            genre_data = df[df['genre'] == genre]
            yearly_avg = genre_data.groupby('year')[metric].mean()
            ax.plot(yearly_avg.index, yearly_avg.values, 'o-', color=colors[i],
                    linewidth=2, markersize=6, label=genre)

        metric_labels = {
            'user_rating': 'Average User Rating',
            'price': 'Average Price (USD)',
            'reviews': 'Average Number of Reviews'
        }

        ax.set_title(f'{metric_labels.get(metric, metric)} Over Time by Genre', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.legend(title='Genre', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_top_authors(self, df: pd.DataFrame,
                         top_n: int = 10,
                         metric: str = 'count',
                         ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot top authors by count or average rating.
        
        Args:
            df: Cleaned DataFrame.
            top_n: Number of top authors to show.
            metric: 'count' for number of books, 'rating' for average rating.
            ax: Optional matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        logger.info(f"Generating top authors plot (metric: {metric})")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if metric == 'count':
            author_stats = df['author'].value_counts().head(top_n)
            values = author_stats.values
            labels = author_stats.index
            xlabel = 'Number of Books'
            title = f'Top {top_n} Authors by Number of Books'
        elif metric == 'rating':
            author_stats = df.groupby('author')['user_rating'].mean().sort_values(ascending=False).head(top_n)
            values = author_stats.values
            labels = author_stats.index
            xlabel = 'Average User Rating'
            title = f'Top {top_n} Authors by Average Rating'
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        colors = sns.color_palette(self.viz_config.color_palette, len(values))
        bars = ax.barh(range(len(values)), values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.annotate(f'{val:.2f}' if metric == 'rating' else f'{int(val):,}',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center',
                       fontsize=10, fontweight='bold')
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 ax: Optional[plt.Axes] = None,
                                 method: str = 'pearson') -> plt.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            df: Cleaned DataFrame.
            columns: List of columns to include. If None, uses all numeric.
            ax: Optional matplotlib axes to plot on.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            
        Returns:
            Matplotlib figure.
        """
        logger.info(f"Generating correlation heatmap ({method})")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr(method=method)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return fig
    
    def plot_genre_year_heatmap(self, df: pd.DataFrame,
                                metric: str = 'count',
                                ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot heatmap of genre vs year.
        
        Args:
            df: Cleaned DataFrame.
            metric: 'count', 'avg_rating', 'avg_price', 'avg_reviews'.
            ax: Optional matplotlib axes to plot on.
            
        Returns:
            Matplotlib figure.
        """
        logger.info(f"Generating genre-year heatmap (metric: {metric})")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        else:
            fig = cast(plt.Figure, ax.figure)
        
        if metric == 'count':
            pivot = pd.crosstab(df['genre'], df['year'])
            fmt = 'd'
            cmap = 'YlOrRd'
        elif metric == 'avg_rating':
            pivot = df.pivot_table(values='user_rating', index='genre', columns='year', aggfunc='mean')
            fmt = '.2f'
            cmap = 'RdYlGn'
        elif metric == 'avg_price':
            pivot = df.pivot_table(values='price', index='genre', columns='year', aggfunc='mean')
            fmt = '.0f'
            cmap = 'YlOrBr'
        elif metric == 'avg_reviews':
            pivot = df.pivot_table(values='reviews', index='genre', columns='year', aggfunc='mean')
            fmt = '.0f'
            cmap = 'Blues'
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        metric_labels = {
            'count': 'Number of Books',
            'avg_rating': 'Average Rating',
            'avg_price': 'Average Price (USD)',
            'avg_reviews': 'Average Reviews'
        }
        
        ax.set_title(f'{metric_labels.get(metric, metric)} by Genre and Year', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        
        plt.tight_layout()
        return fig

    def generate_all_plots(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate the standard set of individual analysis plots.

        Args:
            df: Cleaned DataFrame.

        Returns:
            Dictionary mapping plot name to the generated matplotlib figure.
        """
        logger.info("Generating all standard plots")

        return {
            'genre_distribution': self.plot_genre_distribution(df),
            'price_distribution': self.plot_price_distribution(df),
            'rating_distribution': self.plot_rating_distribution(df),
            'reviews_distribution': self.plot_reviews_distribution(df),
            'price_vs_rating': self.plot_price_vs_rating(df),
            'reviews_vs_rating': self.plot_reviews_vs_rating(df),
            'yearly_trends': self.plot_yearly_trends(df),
            'rating_by_genre': self.plot_rating_by_genre(df),
            'top_authors': self.plot_top_authors(df),
            'yearly_trends_by_genre': self.plot_yearly_trends_by_genre(df),
        }

    def create_dashboard(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            Matplotlib figure with dashboard.
        """
        logger.info("Creating comprehensive dashboard")
        
        fig = plt.figure(figsize=(20, 16), dpi=self.dpi)
        
        # Grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Genre distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_genre_distribution(df, ax1)
        
        # 2. Price distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_price_distribution(df, ax2)
        
        # 3. Rating distribution (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_rating_distribution(df, ax3)
        
        # 4. Price vs Rating (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_price_vs_rating(df, ax4)
        
        # 5. Reviews vs Rating (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_reviews_vs_rating(df, ax5)
        
        # 6. Rating by Genre (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_rating_by_genre(df, ax6, plot_type='box')
        
        # 7. Yearly trends (bottom left, spanning 2 columns)
        ax7 = fig.add_subplot(gs[2, :2])
        self.plot_yearly_trends(df, ax=ax7)
        
        # 8. Top authors by count (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        self.plot_top_authors(df, top_n=10, metric='count', ax=ax8)
        
        # 9. Correlation heatmap (bottom row, spanning all)
        ax9 = fig.add_subplot(gs[3, :])
        self.plot_correlation_heatmap(df, ax=ax9)
        
        fig.suptitle('Amazon Books Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        return fig
