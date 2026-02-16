"""
plotting_utils.py - Funciones de visualización profesional
==========================================================
Genera gráficos de calidad académica (300 DPI) para el informe.
Estilo consistente en todos los scripts del pipeline.
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidores/scripts
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def configure_plot_style() -> None:
    """
    Configura estilo global de matplotlib para gráficos profesionales.
    Aplica estilo académico con fuentes legibles y colores consistentes.
    """
    plt.rcParams.update({
        'figure.figsize': (10, 7),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Intentar usar estilo seaborn si está disponible
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # Usar defaults


def save_figure(fig: plt.Figure, filepath: Path, close: bool = True) -> None:
    """
    Guarda una figura en disco con alta resolución.
    
    Args:
        fig: Objeto Figure de matplotlib
        filepath: Ruta donde guardar (incluyendo extensión .png)
        close: Si True, cierra la figura después de guardar (libera memoria)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"📊 Figura guardada: {filepath.name}")
    
    if close:
        plt.close(fig)
