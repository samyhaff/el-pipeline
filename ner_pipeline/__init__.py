"""Modular NER pipeline package."""

__all__ = [
    "PipelineConfig",
    "NERPipeline",
]

__version__ = "0.1.0"

from .config import PipelineConfig  # noqa: E402
from .pipeline import NERPipeline  # noqa: E402

