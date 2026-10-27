"""
Configuration loader module.
Loads and validates configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration."""
    path: str = "data/amazon_books.csv"
    encoding: str = "utf-8"
    column_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "output"
    figures_dir: str = "output/figures"
    reports_dir: str = "output/reports"
    dpi: int = 300
    formats: list = field(default_factory=lambda: ["png"])
    font_size: int = 10
    figsize: list = field(default_factory=lambda: [12, 6])
    style: str = "whitegrid"
    color_palette: str = "viridis"


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    top_authors_count: int = 10
    price_bins: int = 20
    correlation_method: str = "pearson"
    significance_level: float = 0.05
    genre_order: list = field(default_factory=lambda: ["Fiction", "Non Fiction"])
    year_range: list = field(default_factory=lambda: [2009, 2019])


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    bar_chart: Dict[str, Any] = field(default_factory=dict)
    box_plot: Dict[str, Any] = field(default_factory=dict)
    scatter_plot: Dict[str, Any] = field(default_factory=dict)
    line_chart: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "output/analysis.log"
    console: bool = True


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    questions: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    generate_visualizations: bool = True
    generate_insights: bool = True
    save_cleaned_data: bool = False


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to the YAML configuration file. If None, uses default config.yaml.

    Returns:
    --------
    Config
        Configuration object with all settings.
    """
    resolved_path = Path(config_path) if config_path is not None else (
        Path(__file__).parent.parent.parent.parent / "config.yaml"
    )

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with open(resolved_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects from dict
    config = Config()
    
    if config_dict:
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'output' in config_dict:
            config.output = OutputConfig(**config_dict['output'])
        if 'analysis' in config_dict:
            config.analysis = AnalysisConfig(**config_dict['analysis'])
        if 'visualization' in config_dict:
            config.visualization = VisualizationConfig(**config_dict['visualization'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        if 'pipeline' in config_dict:
            config.pipeline = PipelineConfig(**config_dict['pipeline'])
    
    return config