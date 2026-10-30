"""
Figure saving module for Amazon Books Analysis.
"""

import matplotlib
matplotlib.use('Agg')  # this module only ever saves figures to disk

import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from ..config.loader import Config, load_config

logger = logging.getLogger(__name__)


class FigureSaver:
    """Handles saving matplotlib figures to various formats."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the figure saver.

        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or load_config()
        self.output_config = self.config.output
        self.output_dir = Path(self.output_config.figures_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.default_formats = self.output_config.formats
        self.dpi = self.output_config.dpi
    
    def save_figure(self, fig: plt.Figure, filename: str,
                    formats: Optional[List[str]] = None,
                    dpi: Optional[int] = None,
                    subdir: Optional[str] = None) -> List[Path]:
        """
        Save a figure to multiple formats.
        
        Args:
            fig: Matplotlib figure to save.
            filename: Base filename (without extension).
            formats: List of formats to save. Defaults to config formats.
            dpi: DPI for raster formats. Defaults to config DPI.
            subdir: Optional subdirectory within output directory.
            
        Returns:
            List of saved file paths.
        """
        formats = formats or self.default_formats
        dpi = dpi or self.dpi
        
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for fmt in formats:
            filepath = save_dir / f"{filename}.{fmt}"
            try:
                if fmt in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                elif fmt in ['pdf', 'svg', 'eps']:
                    fig.savefig(filepath, format=fmt, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                else:
                    logger.warning(f"Unsupported format: {fmt}, skipping")
                    continue
                
                saved_paths.append(filepath)
                logger.info(f"Saved figure to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save figure to {filepath}: {e}")
        
        return saved_paths
    
    def save_dashboard(self, fig: plt.Figure, name: str = "dashboard",
                       formats: Optional[List[str]] = None) -> List[Path]:
        """
        Save a dashboard figure.
        
        Args:
            fig: Dashboard figure to save.
            name: Base name for the file.
            formats: List of formats to save.
            
        Returns:
            List of saved file paths.
        """
        return self.save_figure(fig, name, formats, subdir="dashboards")
    
    def save_analysis_plots(self, fig: plt.Figure, analysis_name: str,
                            formats: Optional[List[str]] = None) -> List[Path]:
        """
        Save analysis-specific plots.
        
        Args:
            fig: Figure to save.
            analysis_name: Name of the analysis (e.g., 'genre_distribution').
            formats: List of formats to save.
            
        Returns:
            List of saved file paths.
        """
        return self.save_figure(fig, analysis_name, formats, subdir="analysis")
    
    def save_all_figures(self, figures: Dict[str, plt.Figure],
                         formats: Optional[List[str]] = None,
                         subdir: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        Save multiple figures at once.
        
        Args:
            figures: Dictionary mapping names to figures.
            formats: List of formats to save.
            subdir: Optional subdirectory.
            
        Returns:
            Dictionary mapping figure names to lists of saved paths.
        """
        results = {}
        for name, fig in figures.items():
            results[name] = self.save_figure(fig, name, formats, subdir=subdir)
            plt.close(fig)
        return results
    
    def create_output_summary(self, saved_files: Dict[str, List[Path]]) -> str:
        """
        Create a summary of saved files.
        
        Args:
            saved_files: Dictionary from save_all_figures.
            
        Returns:
            Formatted summary string.
        """
        lines = ["Saved Figures Summary:", "=" * 40]
        total_files = 0
        
        for name, paths in saved_files.items():
            lines.append(f"\n{name}:")
            for path in paths:
                size_kb = path.stat().st_size / 1024
                lines.append(f"  {path.name} ({size_kb:.1f} KB)")
                total_files += 1
        
        lines.append(f"\n{'=' * 40}")
        lines.append(f"Total files saved: {total_files}")
        lines.append(f"Output directory: {self.output_dir.absolute()}")
        
        return "\n".join(lines)
    
    def save_analysis_report(self, figures: Dict[str, plt.Figure],
                             analysis_name: str,
                             formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Save all figures for an analysis and create a summary report.
        
        Args:
            figures: Dictionary of figure names to figures.
            analysis_name: Name of the analysis.
            formats: List of formats to save.
            
        Returns:
            Dictionary with saved paths and summary.
        """
        saved = self.save_all_figures(figures, formats, subdir=f"analysis/{analysis_name}")
        summary = self.create_output_summary(saved)
        
        # Save summary as text file
        summary_path = self.output_dir / f"analysis/{analysis_name}/summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding='utf-8')
        
        logger.info(f"Analysis report saved to {summary_path.parent}")
        
        return {
            'saved_files': saved,
            'summary': summary,
            'summary_path': summary_path
        }