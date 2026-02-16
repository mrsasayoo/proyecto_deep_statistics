"""
plotting_utils.py — Funciones de Visualización Profesional
==========================================================
Situación 3: Portafolio Hipotecario

Genera gráficos de calidad académica (300 DPI) para el informe.
Estilo consistente en todos los scripts del pipeline.
Todas las figuras se guardan automáticamente en outputs/figures/
"""

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para servidores/scripts sin pantalla
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


# ============================================================================
# CONFIGURACIÓN GLOBAL DE ESTILO
# ============================================================================

def configure_plot_style() -> None:
    """
    Configura estilo global de matplotlib para gráficos profesionales
    de calidad académica. Aplica fuentes legibles, colores consistentes
    y resolución de 300 DPI para impresión.
    """
    plt.rcParams.update({
        # Tamaño y resolución
        "figure.figsize": (12, 7),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",

        # Fuentes
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,

        # Ejes
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,

        # Líneas y marcadores
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    # Intentar usar estilo seaborn si está disponible
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Usar defaults de matplotlib


# Aplicar estilo al importar el módulo
configure_plot_style()


# ============================================================================
# PALETA DE COLORES DEL PROYECTO
# ============================================================================

# Paleta principal para clusters y categorías
PALETTE_CLUSTERS = [
    "#1f77b4",  # Azul
    "#ff7f0e",  # Naranja
    "#2ca02c",  # Verde
    "#d62728",  # Rojo
    "#9467bd",  # Morado
    "#8c564b",  # Marrón
    "#e377c2",  # Rosa
    "#7f7f7f",  # Gris
    "#bcbd22",  # Lima
    "#17becf",  # Cian
    "#aec7e8",  # Azul claro
    "#ffbb78",  # Naranja claro
    "#98df8a",  # Verde claro
    "#ff9896",  # Rojo claro
    "#c5b0d5",  # Morado claro
]

# Paleta para métricas financieras
PALETTE_RISK = {
    "bajo": "#2ca02c",      # Verde = riesgo bajo
    "medio": "#ff7f0e",     # Naranja = riesgo medio
    "alto": "#d62728",      # Rojo = riesgo alto
    "neutral": "#7f7f7f",   # Gris = neutro
}

# Paleta secuencial para heatmaps
PALETTE_SEQUENTIAL = "YlOrRd"


# ============================================================================
# GUARDAR FIGURAS
# ============================================================================

def save_figure(
    fig: plt.Figure,
    filepath: Path,
    close: bool = True,
    dpi: int = 300,
) -> None:
    """
    Guarda una figura en disco con alta resolución.

    Args:
        fig: Objeto Figure de matplotlib
        filepath: Ruta donde guardar (incluyendo extensión .png)
        close: Si True, cierra la figura después de guardar (libera RAM)
        dpi: Resolución en puntos por pulgada
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(f"  📊 Figura guardada: {filepath.name}")

    if close:
        plt.close(fig)


# ============================================================================
# GRÁFICOS GENÉRICOS REUTILIZABLES
# ============================================================================

def plot_bar_chart(
    categories: list,
    values: list,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: Optional[Path] = None,
    horizontal: bool = False,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
) -> plt.Figure:
    """
    Crea un gráfico de barras profesional.

    Args:
        categories: Lista de etiquetas del eje X (o Y si horizontal)
        values: Lista de valores correspondientes
        title: Título del gráfico
        xlabel: Etiqueta del eje X
        ylabel: Etiqueta del eje Y
        filepath: Ruta para guardar (None = solo mostrar)
        horizontal: Si True, barras horizontales
        color: Color de las barras (None = automático)
        figsize: Tamaño de la figura

    Returns:
        Objeto Figure de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    bar_color = color or PALETTE_CLUSTERS[0]

    if horizontal:
        ax.barh(categories, values, color=bar_color, edgecolor="none", alpha=0.85)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(categories, values, color=bar_color, edgecolor="none", alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_line_chart(
    x: list,
    y: list,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: Optional[Path] = None,
    marker: str = "o",
    color: Optional[str] = None,
    fill_below: bool = False,
    figsize: Tuple[int, int] = (12, 7),
) -> plt.Figure:
    """
    Crea un gráfico de líneas profesional.
    """
    fig, ax = plt.subplots(figsize=figsize)

    line_color = color or PALETTE_CLUSTERS[0]
    ax.plot(x, y, marker=marker, color=line_color, linewidth=2, markersize=5)

    if fill_below:
        ax.fill_between(x, y, alpha=0.15, color=line_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_heatmap(
    data: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    filepath: Optional[Path] = None,
    cmap: str = PALETTE_SEQUENTIAL,
    annotate: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    fmt: str = ".1f",
) -> plt.Figure:
    """
    Crea un heatmap (mapa de calor) profesional.

    Args:
        data: Matriz numpy 2D con los valores
        row_labels: Etiquetas de las filas
        col_labels: Etiquetas de las columnas
        title: Título del gráfico
        filepath: Ruta para guardar
        cmap: Mapa de colores
        annotate: Si True, muestra valores numéricos en cada celda
        figsize: Tamaño automático si es None
        fmt: Formato numérico para anotaciones
    """
    if figsize is None:
        figsize = (max(8, len(col_labels) * 0.8), max(6, len(row_labels) * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    if annotate and data.size < 500:  # Solo anotar si no es demasiado grande
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = data[i, j]
                text_color = "white" if val > (data.max() - data.min()) / 2 + data.min() else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        color=text_color, fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_radar_chart(
    categories: list,
    values_dict: dict,
    title: str,
    filepath: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Crea un gráfico de radar/araña para perfilado de clusters.

    Args:
        categories: Lista de dimensiones/ejes del radar
        values_dict: Dict {nombre_cluster: [valores]} para cada cluster
        title: Título del gráfico
        filepath: Ruta para guardar
        figsize: Tamaño de la figura
    """
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el polígono

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for idx, (name, values) in enumerate(values_dict.items()):
        values_closed = list(values) + [values[0]]  # Cerrar
        color = PALETTE_CLUSTERS[idx % len(PALETTE_CLUSTERS)]
        ax.plot(angles, values_closed, linewidth=2, color=color, label=name)
        ax.fill(angles, values_closed, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_file_inventory(
    filenames: list,
    sizes_gb: list,
    title: str = "Inventario de Archivos — Tamaño Descomprimido",
    filepath: Optional[Path] = None,
) -> plt.Figure:
    """
    Gráfico de barras horizontales del inventario de archivos del ZIP.
    Útil para visualizar la distribución de tamaños.

    Args:
        filenames: Lista de nombres de archivo (ej: ["2000Q1.csv", ...])
        sizes_gb: Lista de tamaños en GB
        title: Título del gráfico
        filepath: Ruta para guardar
    """
    fig, ax = plt.subplots(figsize=(14, max(8, len(filenames) * 0.22)))

    colors = []
    for s in sizes_gb:
        if s < 2:
            colors.append("#2ca02c")    # Verde = pequeño
        elif s < 10:
            colors.append("#ff7f0e")    # Naranja = mediano
        else:
            colors.append("#d62728")    # Rojo = grande

    y_pos = range(len(filenames))
    ax.barh(y_pos, sizes_gb, color=colors, edgecolor="none", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(filenames, fontsize=7)
    ax.set_xlabel("Tamaño (GB)")
    ax.set_title(title)
    ax.invert_yaxis()  # Mayor arriba

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="< 2 GB (carga directa)"),
        Patch(facecolor="#ff7f0e", label="2-10 GB (chunks 500K)"),
        Patch(facecolor="#d62728", label="> 10 GB (chunks 250K)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    if filepath:
        save_figure(fig, filepath)

    return fig


# ============================================================================
# MAIN (TEST)
# ============================================================================

if __name__ == "__main__":
    print("Testing plotting_utils...")

    # Test: gráfico de barras
    fig = plot_bar_chart(
        categories=["A", "B", "C", "D"],
        values=[10, 25, 15, 30],
        title="Test: Gráfico de Barras",
        xlabel="Categoría",
        ylabel="Valor",
    )
    plt.close(fig)
    print("  ✅ plot_bar_chart funciona correctamente")

    # Test: gráfico de líneas
    fig = plot_line_chart(
        x=list(range(10)),
        y=[x**2 for x in range(10)],
        title="Test: Gráfico de Líneas",
        xlabel="X",
        ylabel="Y",
    )
    plt.close(fig)
    print("  ✅ plot_line_chart funciona correctamente")

    print("\n  Todas las funciones de visualización OK.")
