"""
utils — Paquete de utilidades para Situación 3
================================================
Módulos:
  - data_loader:    Carga distribuida desde ZIP + chunks
  - memory_utils:   Monitoreo y liberación de RAM
  - plotting_utils: Gráficos de calidad académica (300 DPI)
"""

from utils.memory_utils import (
    get_memory_usage,
    print_memory_status,
    clear_memory,
    check_memory_threshold,
    wait_for_memory,
    ProcessingTimer,
    print_system_summary,
)

from utils.data_loader import ZipDataLoader, quick_peek

from utils.plotting_utils import (
    configure_plot_style,
    save_figure,
    PALETTE_CLUSTERS,
    PALETTE_RISK,
)
