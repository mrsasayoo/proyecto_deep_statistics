"""
Módulo de Utilidades para el Proyecto Situación 2
==================================================
Funciones auxiliares para carga de imágenes, gestión de memoria y visualización.
"""

from .image_loader import get_all_image_paths, load_single_image, load_and_flatten_batch, extract_metadata
from .memory_utils import clear_memory, get_memory_usage
from .plotting_utils import configure_plot_style, save_figure

__all__ = [
    'get_all_image_paths',
    'load_single_image',
    'load_and_flatten_batch',
    'extract_metadata',
    'clear_memory',
    'get_memory_usage',
    'configure_plot_style',
    'save_figure'
]

__version__ = '1.0.0'
