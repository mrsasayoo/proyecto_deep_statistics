#!/usr/bin/env python3
"""
010_eda_masivo.py — EDA Masivo del Portafolio Hipotecario
=====================================================================
Situación 3: Análisis de Portafolio Hipotecario a Gran Escala

Lee los perfiles JSON individuales (generados por 001_csv_to_parquet_parts.py)
y los archivos Parquet para producir un análisis exploratorio exhaustivo
SIN releer los 820 GB del ZIP original.

Genera:
  1. Resumen de procesamiento (tiempos, tamaños, compresión)
  2. Mapa de nulidad temporal (101 trimestres × 110 columnas)
  3. Evolución temporal de variables financieras clave (FICO, LTV, DTI, tasas, UPB)
  4. Distribuciones de variables clave (histogramas acumulados)
  5. Ranking de informatividad (CV numérico, entropía categórica)
  6. Distribución geográfica y categórica
  7. Matriz de correlación (muestreo eficiente de Parquets)
  8. Tabla descriptiva global consolidada

Outputs:
  - outputs/figures/00_exploratorio/*.png (300 DPI)
  - outputs/tables/eda_*.csv
  - data_processed/perfil_global.json (consolidado)

Autor: Nicolás Zapata Obando
Fecha: Febrero 2026
"""

import sys
import gc
import json
import time
import warnings
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_PROCESSED_PATH, PANEL_PATH, PERFILES_PATH,
    FIGURES_PATH, TABLES_PATH, FIGURES_SUBDIRS,
    PERFORMANCE_COLUMNS, KEY_NUMERIC_COLUMNS, KEY_CATEGORICAL_COLUMNS,
    LOAN_ID_COLUMN, DATE_COLUMN, TOTAL_CSV_FILES,
    PARQUET_COMPRESSION, FIGURE_DPI,
    CHECKPOINTS_PATH, PROCESSING_STATE_FILE,
)
from utils.memory_utils import (
    print_system_summary, print_memory_status,
    ProcessingTimer, clear_memory, check_memory_threshold,
)
from utils.plotting_utils import (
    save_figure, configure_plot_style, PALETTE_CLUSTERS,
    PALETTE_RISK, PALETTE_SEQUENTIAL,
    plot_bar_chart, plot_line_chart, plot_heatmap,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Directorio de salida para figuras del EDA
EDA_FIGURES_DIR = FIGURES_SUBDIRS["exploratorio"]
EDA_TABLES_DIR = TABLES_PATH


# ============================================================================
# UTILIDADES
# ============================================================================

def log_message(msg: str, level: str = "info") -> None:
    """Imprime mensaje con timestamp."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {"info": "INFO", "warn": "WARN", "error": "ERROR"}.get(level, "INFO")
    print(f"  {ts} | {prefix:8s} | {msg}", flush=True)


def load_all_profiles(perfiles_dir: Path = PERFILES_PATH) -> Dict[str, Dict]:
    """
    Carga todos los perfiles JSON individuales.

    Returns:
        Dict {trimestre: perfil_dict}, ordenado cronológicamente
    """
    profiles = {}
    for pf in sorted(perfiles_dir.glob("perfil_*.json")):
        trimestre = pf.stem.replace("perfil_", "")
        if trimestre == "global":
            continue
        try:
            data = json.load(open(pf, "r"))
            profiles[trimestre] = data
        except Exception as e:
            log_message(f"Error leyendo {pf.name}: {e}", level="warn")
    return dict(sorted(profiles.items()))


def load_checkpoint() -> Dict:
    """Carga el estado del checkpoint."""
    if PROCESSING_STATE_FILE.exists():
        return json.load(open(PROCESSING_STATE_FILE, "r"))
    return {}


def get_completed_trimestres() -> List[str]:
    """Retorna la lista de trimestres completados según el checkpoint."""
    cp = load_checkpoint()
    files = cp.get("files", {})
    return sorted([
        f.replace(".csv", "")
        for f, info in files.items()
        if info.get("status") == "COMPLETADO"
    ])


def trimestre_to_date(trimestre: str) -> str:
    """Convierte '2003Q1' → '2003-Q1' para display."""
    return trimestre.replace("Q", "-Q")


def trimestre_sort_key(trimestre: str) -> Tuple[int, int]:
    """Retorna (año, trimestre) para ordenar cronológicamente."""
    year = int(trimestre[:4])
    quarter = int(trimestre[5]) if len(trimestre) > 5 else int(trimestre[4:].replace("Q", ""))
    return (year, quarter)


# ============================================================================
# 1. RESUMEN DE PROCESAMIENTO
# ============================================================================

def generate_processing_summary(
    profiles: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Genera tabla resumen del procesamiento: tiempos, tamaños, compresión.

    Args:
        profiles: Dict de perfiles cargados

    Returns:
        DataFrame con resumen por trimestre
    """
    log_message("Generando resumen de procesamiento...")

    rows = []
    for trimestre, p in profiles.items():
        rows.append({
            "trimestre": trimestre,
            "filas": p.get("total_rows", 0),
            "columnas": p.get("total_columns", 0),
            "tamaño_csv_gb": p.get("file_size_gb", 0),
            "tamaño_parquet_gb": p.get("parquet_size_gb", 0),
            "ratio_compresion": p.get("compression_ratio", 0),
            "tiempo_segundos": p.get("processing_time_seconds", 0),
            "filas_por_segundo": p.get("rows_per_second", 0),
            "chunks": p.get("total_chunks", 0),
            "columnas_alto_nulo": len(p.get("high_null_columns", [])),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        log_message("Sin datos de procesamiento", level="warn")
        return df

    # Guardar tabla
    df.to_csv(EDA_TABLES_DIR / "eda_resumen_procesamiento.csv", index=False)
    log_message(f"  Tabla guardada: eda_resumen_procesamiento.csv ({len(df)} trimestres)")

    return df


def plot_processing_summary(df: pd.DataFrame) -> None:
    """
    Genera gráficos del resumen de procesamiento:
    - Tamaño CSV vs Parquet por trimestre
    - Velocidad de procesamiento
    - Ratio de compresión
    """
    if df.empty:
        return

    trimestres = df["trimestre"].tolist()
    x_positions = range(len(trimestres))

    # --- Gráfico 1: Tamaño CSV vs Parquet ---
    fig, ax = plt.subplots(figsize=(16, 6))
    width = 0.35
    ax.bar([x - width/2 for x in x_positions], df["tamaño_csv_gb"],
           width, label="CSV (descomprimido)", color="#d62728", alpha=0.8)
    ax.bar([x + width/2 for x in x_positions], df["tamaño_parquet_gb"],
           width, label="Parquet (Zstd)", color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Tamaño (GB)")
    ax.set_title("Tamaño CSV Descomprimido vs Parquet Comprimido por Trimestre")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(trimestres, rotation=90, fontsize=6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "01_tamano_csv_vs_parquet.png")

    # --- Gráfico 2: Filas por trimestre ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x_positions, df["filas"] / 1e6, color=PALETTE_CLUSTERS[0], alpha=0.8)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Millones de Filas")
    ax.set_title("Volumen de Registros por Trimestre (Millones)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(trimestres, rotation=90, fontsize=6)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "02_filas_por_trimestre.png")

    # --- Gráfico 3: Ratio de compresión ---
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#2ca02c" if r > 20 else "#ff7f0e" if r > 10 else "#d62728"
              for r in df["ratio_compresion"]]
    ax.bar(x_positions, df["ratio_compresion"], color=colors, alpha=0.8)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Ratio de Compresión (CSV/Parquet)")
    ax.set_title("Ratio de Compresión CSV → Parquet (Zstd nivel 3)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(trimestres, rotation=90, fontsize=6)
    ax.axhline(y=df["ratio_compresion"].mean(), color="red", linestyle="--",
               label=f"Media: {df['ratio_compresion'].mean():.1f}x")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "03_ratio_compresion.png")

    # --- Gráfico 4: Velocidad de procesamiento ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(trimestres, df["filas_por_segundo"], marker="o", markersize=4,
            color=PALETTE_CLUSTERS[4], linewidth=1.5)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Filas/Segundo")
    ax.set_title("Velocidad de Procesamiento por Trimestre")
    ax.set_xticks(trimestres)
    ax.set_xticklabels(trimestres, rotation=90, fontsize=6)
    ax.axhline(y=df["filas_por_segundo"].mean(), color="red", linestyle="--",
               label=f"Media: {df['filas_por_segundo'].mean():,.0f} filas/s")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "04_velocidad_procesamiento.png")

    log_message(f"  4 gráficos de procesamiento generados")


# ============================================================================
# 2. MAPA DE NULIDAD TEMPORAL
# ============================================================================

def generate_nullity_heatmap(profiles: Dict[str, Dict]) -> pd.DataFrame:
    """
    Genera un mapa de calor de % de nulos por trimestre × columna.

    Solo muestra columnas que tienen >0% nulos en algún trimestre.
    """
    log_message("Generando mapa de nulidad temporal...")

    trimestres = sorted(profiles.keys())
    # Recopilar todas las columnas con datos
    all_columns = set()
    for p in profiles.values():
        all_columns.update(p.get("columns", {}).keys())
    all_columns = sorted(all_columns)

    # Construir matriz de nulidad
    null_matrix = np.zeros((len(trimestres), len(all_columns)))
    for i, t in enumerate(trimestres):
        cols = profiles[t].get("columns", {})
        for j, col in enumerate(all_columns):
            if col in cols:
                null_matrix[i, j] = cols[col].get("null_pct", 0)

    # Filtrar columnas que nunca tienen nulos
    cols_with_nulls = np.any(null_matrix > 0, axis=0)
    null_matrix_filtered = null_matrix[:, cols_with_nulls]
    cols_filtered = [c for c, has in zip(all_columns, cols_with_nulls) if has]

    # Guardar como DataFrame
    df_null = pd.DataFrame(null_matrix_filtered,
                           index=trimestres, columns=cols_filtered)
    df_null.to_csv(EDA_TABLES_DIR / "eda_mapa_nulidad.csv")
    log_message(f"  Mapa de nulidad: {len(trimestres)} trimestres × {len(cols_filtered)} columnas con nulos")

    # Generar heatmap
    if len(cols_filtered) > 0 and len(trimestres) > 0:
        fig, ax = plt.subplots(figsize=(max(20, len(cols_filtered) * 0.35),
                                         max(8, len(trimestres) * 0.18)))
        im = ax.imshow(null_matrix_filtered, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=100)
        ax.set_xticks(range(len(cols_filtered)))
        ax.set_xticklabels(cols_filtered, rotation=90, fontsize=5)
        ax.set_yticks(range(len(trimestres)))
        ax.set_yticklabels(trimestres, fontsize=5)
        ax.set_title("Mapa de Nulidad Temporal (% Nulos por Trimestre × Columna)",
                      fontsize=14, fontweight="bold")
        ax.set_xlabel("Columna")
        ax.set_ylabel("Trimestre")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="% Nulos")
        save_figure(fig, EDA_FIGURES_DIR / "05_mapa_nulidad_temporal.png")

    # Heatmap condensado: solo top-30 columnas con más nulos
    if len(cols_filtered) > 30:
        mean_null = null_matrix_filtered.mean(axis=0)
        top_30_idx = np.argsort(mean_null)[-30:][::-1]
        top_30_cols = [cols_filtered[i] for i in top_30_idx]
        top_30_data = null_matrix_filtered[:, top_30_idx]

        fig, ax = plt.subplots(figsize=(16, max(8, len(trimestres) * 0.15)))
        im = ax.imshow(top_30_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(top_30_cols)))
        ax.set_xticklabels(top_30_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(trimestres)))
        ax.set_yticklabels(trimestres, fontsize=5)
        ax.set_title("Top 30 Columnas con Mayor % de Nulidad",
                      fontsize=14, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="% Nulos")
        save_figure(fig, EDA_FIGURES_DIR / "06_mapa_nulidad_top30.png")

    return df_null


# ============================================================================
# 3. EVOLUCIÓN TEMPORAL DE VARIABLES FINANCIERAS
# ============================================================================

def generate_temporal_evolution(profiles: Dict[str, Dict]) -> pd.DataFrame:
    """
    Genera gráficos de evolución temporal de las variables financieras clave:
    FICO, LTV, DTI, tasas de interés, UPB.

    Usa las estadísticas (mean, std, min, max) acumuladas del perfil JSON.
    """
    log_message("Generando evolución temporal de variables financieras...")

    # Variables a rastrear
    vars_to_track = {
        "borrower_credit_score_at_origination": "Puntaje FICO del Prestatario",
        "original_loan_to_value_ratio_ltv": "LTV Original (%)",
        "debt_to_income_dti": "DTI Original (%)",
        "original_interest_rate": "Tasa de Interés Original (%)",
        "current_interest_rate": "Tasa de Interés Actual (%)",
        "original_upb": "Saldo Original (UPB, $)",
        "current_actual_upb": "Saldo Actual (UPB, $)",
        "loan_age": "Edad del Préstamo (meses)",
        "original_loan_term": "Plazo Original (meses)",
        "mortgage_insurance_percentage": "% Seguro Hipotecario",
    }

    trimestres = sorted(profiles.keys())
    all_rows = []

    for t in trimestres:
        cols = profiles[t].get("columns", {})
        row = {"trimestre": t}
        for var_name, label in vars_to_track.items():
            if var_name in cols:
                c = cols[var_name]
                row[f"{var_name}_mean"] = c.get("mean")
                row[f"{var_name}_std"] = c.get("std")
                row[f"{var_name}_min"] = c.get("min")
                row[f"{var_name}_max"] = c.get("max")
                row[f"{var_name}_null_pct"] = c.get("null_pct")
                # Medianas si hay percentiles
                pcts = c.get("percentiles", {})
                row[f"{var_name}_p25"] = pcts.get("p25")
                row[f"{var_name}_p50"] = pcts.get("p50")
                row[f"{var_name}_p75"] = pcts.get("p75")
        all_rows.append(row)

    df_temporal = pd.DataFrame(all_rows)
    df_temporal.to_csv(EDA_TABLES_DIR / "eda_evolucion_temporal.csv", index=False)

    # --- Generar gráficos de líneas por variable ---
    n_vars = len(vars_to_track)
    fig_idx = 7  # Continuar numeración de figuras

    # Gráfico multi-panel: variables de riesgo principales
    main_vars = [
        ("borrower_credit_score_at_origination", "Puntaje FICO"),
        ("original_loan_to_value_ratio_ltv", "LTV Original (%)"),
        ("debt_to_income_dti", "DTI Original (%)"),
        ("original_interest_rate", "Tasa Interés Original (%)"),
        ("current_interest_rate", "Tasa Interés Actual (%)"),
        ("original_upb", "Saldo Original UPB ($)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()

    for idx, (var_name, title) in enumerate(main_vars):
        ax = axes[idx]
        col_mean = f"{var_name}_mean"
        col_p25 = f"{var_name}_p25"
        col_p75 = f"{var_name}_p75"

        if col_mean in df_temporal.columns:
            values = df_temporal[col_mean].values
            valid_mask = pd.notna(values)
            x_vals = np.arange(len(trimestres))

            ax.plot(x_vals[valid_mask], values[valid_mask],
                    color=PALETTE_CLUSTERS[idx], linewidth=2, marker=".", markersize=3)

            # Banda IQR si disponible
            if col_p25 in df_temporal.columns and col_p75 in df_temporal.columns:
                p25 = df_temporal[col_p25].values
                p75 = df_temporal[col_p75].values
                valid_band = valid_mask & pd.notna(p25) & pd.notna(p75)
                if valid_band.any():
                    ax.fill_between(x_vals[valid_band], p25[valid_band], p75[valid_band],
                                    alpha=0.2, color=PALETTE_CLUSTERS[idx], label="IQR (P25-P75)")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Trimestre")

            # Mostrar cada N etiquetas para legibilidad
            step = max(1, len(trimestres) // 15)
            tick_positions = x_vals[::step]
            tick_labels = [trimestres[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
            ax.grid(True, alpha=0.3)
            if col_p25 in df_temporal.columns and col_p75 in df_temporal.columns:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(fontsize=8)

    plt.suptitle("Evolución Temporal de Variables Financieras Clave\nFreddie Mac Performance Data (Media ± IQR)",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "07_evolucion_temporal_multipanel.png")

    # Gráficos individuales más detallados para cada variable
    for var_name, label in vars_to_track.items():
        col_mean = f"{var_name}_mean"
        col_std = f"{var_name}_std"
        col_p25 = f"{var_name}_p25"
        col_p50 = f"{var_name}_p50"
        col_p75 = f"{var_name}_p75"

        if col_mean not in df_temporal.columns:
            continue

        values = df_temporal[col_mean].values
        valid_mask = pd.notna(values)
        if not valid_mask.any():
            continue

        fig, ax = plt.subplots(figsize=(16, 6))
        x_vals = np.arange(len(trimestres))

        # Media
        ax.plot(x_vals[valid_mask], values[valid_mask],
                color=PALETTE_CLUSTERS[0], linewidth=2, label="Media", marker=".", markersize=3)

        # Mediana si existe
        if col_p50 in df_temporal.columns:
            p50 = df_temporal[col_p50].values
            valid_p50 = valid_mask & pd.notna(p50)
            if valid_p50.any():
                ax.plot(x_vals[valid_p50], p50[valid_p50],
                        color=PALETTE_CLUSTERS[1], linewidth=1.5, linestyle="--",
                        label="Mediana (P50)", marker=".", markersize=2)

        # Banda IQR
        if col_p25 in df_temporal.columns and col_p75 in df_temporal.columns:
            p25 = df_temporal[col_p25].values
            p75 = df_temporal[col_p75].values
            valid_band = valid_mask & pd.notna(p25) & pd.notna(p75)
            if valid_band.any():
                ax.fill_between(x_vals[valid_band], p25[valid_band], p75[valid_band],
                                alpha=0.15, color=PALETTE_CLUSTERS[0], label="IQR (P25-P75)")

        ax.set_title(f"Evolución Temporal: {label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trimestre")
        ax.set_ylabel(label)

        step = max(1, len(trimestres) // 20)
        tick_positions = x_vals[::step]
        tick_labels = [trimestres[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_idx += 1
        safe_name = var_name.replace(" ", "_")
        save_figure(fig, EDA_FIGURES_DIR / f"08_temporal_{safe_name}.png")

    log_message(f"  Gráficos temporales generados para {len(vars_to_track)} variables")

    return df_temporal


# ============================================================================
# 4. DISTRIBUCIONES DE VARIABLES CLAVE (Histogramas acumulados)
# ============================================================================

def generate_distribution_plots(profiles: Dict[str, Dict]) -> None:
    """
    Genera histogramas de distribución para las variables numéricas clave
    usando los histogramas acumulados del perfil (no requiere leer Parquets).
    """
    log_message("Generando distribuciones de variables clave...")

    target_vars = [
        "borrower_credit_score_at_origination",
        "original_loan_to_value_ratio_ltv",
        "debt_to_income_dti",
        "original_interest_rate",
        "current_interest_rate",
        "original_upb",
        "current_actual_upb",
        "loan_age",
        "original_loan_term",
    ]

    # Acumular histogramas de todos los trimestres disponibles
    for var_name in target_vars:
        total_counts = None

        for t, p in profiles.items():
            cols = p.get("columns", {})
            if var_name not in cols:
                continue
            hist = cols[var_name].get("histogram", {})
            counts = hist.get("counts", [])
            bins = hist.get("bins", [])

            if counts and len(counts) > 0:
                counts_arr = np.array(counts, dtype=np.float64)
                if total_counts is None:
                    total_counts = counts_arr
                elif len(counts_arr) == len(total_counts):
                    total_counts += counts_arr

        if total_counts is None or total_counts.sum() == 0:
            continue

        # Generar histograma
        fig, ax = plt.subplots(figsize=(12, 6))

        # Obtener bins del último perfil que los tenga
        last_bins = None
        for t, p in profiles.items():
            hist = p.get("columns", {}).get(var_name, {}).get("histogram", {})
            if hist.get("bins"):
                last_bins = hist["bins"]

        if last_bins and len(last_bins) > 0:
            bin_centers = np.array(last_bins)
            if len(bin_centers) == len(total_counts):
                ax.bar(bin_centers, total_counts, width=(bin_centers[1]-bin_centers[0])*0.9
                       if len(bin_centers) > 1 else 1,
                       color=PALETTE_CLUSTERS[0], alpha=0.8, edgecolor="none")
            else:
                ax.bar(range(len(total_counts)), total_counts,
                       color=PALETTE_CLUSTERS[0], alpha=0.8, edgecolor="none")
        else:
            ax.bar(range(len(total_counts)), total_counts,
                   color=PALETTE_CLUSTERS[0], alpha=0.8, edgecolor="none")

        # Estadísticas globales
        global_stats = _compute_global_stats(profiles, var_name)
        stats_text = ""
        if global_stats:
            stats_text = (f"Media: {global_stats['mean']:.2f}  |  "
                         f"Std: {global_stats['std']:.2f}  |  "
                         f"N válidos: {global_stats['n_valid']:,.0f}")
            ax.set_xlabel(f"{var_name}\n({stats_text})", fontsize=10)
        else:
            ax.set_xlabel(var_name)

        ax.set_ylabel("Frecuencia")
        title_map = {
            "borrower_credit_score_at_origination": "Distribución del Puntaje FICO",
            "original_loan_to_value_ratio_ltv": "Distribución del LTV Original",
            "debt_to_income_dti": "Distribución del DTI Original",
            "original_interest_rate": "Distribución de la Tasa de Interés Original",
            "current_interest_rate": "Distribución de la Tasa de Interés Actual",
            "original_upb": "Distribución del Saldo Original (UPB)",
            "current_actual_upb": "Distribución del Saldo Actual (UPB)",
            "loan_age": "Distribución de la Edad del Préstamo",
            "original_loan_term": "Distribución del Plazo Original",
        }
        ax.set_title(title_map.get(var_name, var_name), fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"
                                                            if x >= 1e6 else f"{x/1e3:.0f}K"
                                                            if x >= 1e3 else f"{x:.0f}"))
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, EDA_FIGURES_DIR / f"09_dist_{var_name}.png")

    log_message(f"  Histogramas generados para {len(target_vars)} variables")


def _compute_global_stats(
    profiles: Dict[str, Dict], var_name: str
) -> Optional[Dict]:
    """
    Calcula estadísticas globales ponderadas por n_valid
    combinando perfiles individuales.
    """
    total_n = 0
    weighted_sum = 0.0
    weighted_sum_sq = 0.0
    global_min = float("inf")
    global_max = float("-inf")

    for p in profiles.values():
        col = p.get("columns", {}).get(var_name, {})
        mean = col.get("mean")
        std = col.get("std")
        n = col.get("n_valid", 0)
        mn = col.get("min")
        mx = col.get("max")

        if mean is None or n == 0:
            continue

        total_n += n
        weighted_sum += mean * n
        if std is not None:
            weighted_sum_sq += (std**2 + mean**2) * n

        if mn is not None:
            global_min = min(global_min, mn)
        if mx is not None:
            global_max = max(global_max, mx)

    if total_n == 0:
        return None

    global_mean = weighted_sum / total_n
    global_var = weighted_sum_sq / total_n - global_mean**2
    global_std = np.sqrt(max(0, global_var))

    return {
        "mean": global_mean,
        "std": global_std,
        "min": global_min if global_min != float("inf") else None,
        "max": global_max if global_max != float("-inf") else None,
        "n_valid": total_n,
    }


# ============================================================================
# 5. RANKING DE INFORMATIVIDAD
# ============================================================================

def generate_informativity_ranking(profiles: Dict[str, Dict]) -> pd.DataFrame:
    """
    Ranking de columnas por informatividad:
    - Numéricas: Coeficiente de Variación (CV) promedio
    - Categóricas: Entropía normalizada promedio

    Columnas con CV~0 o entropía~0 son candidatas a eliminación.
    """
    log_message("Generando ranking de informatividad...")

    all_columns = set()
    for p in profiles.values():
        all_columns.update(p.get("columns", {}).keys())

    numeric_stats = {}
    categorical_stats = {}

    for col_name in all_columns:
        cvs = []
        entropies = []
        null_pcts = []
        col_type = None

        for p in profiles.values():
            col = p.get("columns", {}).get(col_name, {})
            if not col:
                continue
            col_type = col.get("column_type", "unknown")
            null_pcts.append(col.get("null_pct", 0))

            if col_type == "numeric":
                cv = col.get("cv")
                if cv is not None:
                    cvs.append(cv)
            elif col_type == "categorical":
                ent = col.get("normalized_entropy")
                if ent is not None:
                    entropies.append(ent)

        if col_type == "numeric" and cvs:
            numeric_stats[col_name] = {
                "tipo": "numérica",
                "cv_medio": np.mean(cvs),
                "cv_std": np.std(cvs),
                "null_pct_medio": np.mean(null_pcts),
                "n_trimestres": len(cvs),
            }
        elif col_type == "categorical" and entropies:
            categorical_stats[col_name] = {
                "tipo": "categórica",
                "entropia_media": np.mean(entropies),
                "entropia_std": np.std(entropies),
                "null_pct_medio": np.mean(null_pcts),
                "n_trimestres": len(entropies),
            }

    # Combinar en DataFrame
    rows = []
    for col_name, stats in numeric_stats.items():
        rows.append({"columna": col_name, "informatividad": stats["cv_medio"], **stats})
    for col_name, stats in categorical_stats.items():
        rows.append({"columna": col_name, "informatividad": stats["entropia_media"], **stats})

    df_info = pd.DataFrame(rows)
    if df_info.empty:
        return df_info

    df_info = df_info.sort_values("informatividad", ascending=False).reset_index(drop=True)
    df_info.to_csv(EDA_TABLES_DIR / "eda_ranking_informatividad.csv", index=False)

    # --- Gráfico: Top-30 variables más informativas ---
    # Separar por tipo
    df_num = df_info[df_info["tipo"] == "numérica"].head(20)
    df_cat = df_info[df_info["tipo"] == "categórica"].head(15)

    if not df_num.empty:
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_num) * 0.35)))
        colors = [PALETTE_CLUSTERS[0] if cv > 10 else PALETTE_CLUSTERS[1] if cv > 1 else "#d62728"
                  for cv in df_num["informatividad"]]
        ax.barh(range(len(df_num)), df_num["informatividad"], color=colors, alpha=0.85)
        ax.set_yticks(range(len(df_num)))
        ax.set_yticklabels(df_num["columna"], fontsize=9)
        ax.set_xlabel("Coeficiente de Variación (CV %)")
        ax.set_title("Top Variables Numéricas por Informatividad (CV)",
                      fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        save_figure(fig, EDA_FIGURES_DIR / "10_ranking_cv_numericas.png")

    if not df_cat.empty:
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_cat) * 0.35)))
        colors = [PALETTE_CLUSTERS[4] if e > 0.5 else "#d62728"
                  for e in df_cat["informatividad"]]
        ax.barh(range(len(df_cat)), df_cat["informatividad"], color=colors, alpha=0.85)
        ax.set_yticks(range(len(df_cat)))
        ax.set_yticklabels(df_cat["columna"], fontsize=9)
        ax.set_xlabel("Entropía Normalizada")
        ax.set_title("Variables Categóricas por Informatividad (Entropía Normalizada)",
                      fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Umbral mínimo (0.5)")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        save_figure(fig, EDA_FIGURES_DIR / "11_ranking_entropia_categoricas.png")

    # --- Columnas candidatas a eliminación ---
    low_info = df_info[
        ((df_info["tipo"] == "numérica") & (df_info["informatividad"] < 1)) |
        ((df_info["tipo"] == "categórica") & (df_info["informatividad"] < 0.1))
    ]
    if not low_info.empty:
        low_info.to_csv(EDA_TABLES_DIR / "eda_columnas_baja_informatividad.csv", index=False)
        log_message(f"  ⚠️ {len(low_info)} columnas con baja informatividad identificadas")

    log_message(f"  Ranking: {len(numeric_stats)} numéricas, {len(categorical_stats)} categóricas")
    return df_info


# ============================================================================
# 6. DISTRIBUCIÓN GEOGRÁFICA Y CATEGÓRICA
# ============================================================================

def generate_categorical_analysis(profiles: Dict[str, Dict]) -> None:
    """
    Análisis de variables categóricas clave:
    - Distribución geográfica (states)
    - Propósito del préstamo
    - Tipo de propiedad
    - Ocupación
    """
    log_message("Generando análisis de variables categóricas...")

    cat_vars = {
        "property_state": "Distribución Geográfica (Top 20 Estados)",
        "loan_purpose": "Propósito del Préstamo",
        "property_type": "Tipo de Propiedad",
        "occupancy_status": "Estado de Ocupación",
        "first_time_home_buyer_indicator": "Primer Comprador de Vivienda",
        "zero_balance_code": "Código de Saldo Cero (Resolución)",
    }

    for var_name, title in cat_vars.items():
        # Acumular frecuencias de todos los trimestres
        global_counts = {}
        for p in profiles.values():
            col = p.get("columns", {}).get(var_name, {})
            top_vals = col.get("top_values", [])
            for entry in top_vals:
                val = str(entry.get("value", ""))
                count = entry.get("count", 0)
                global_counts[val] = global_counts.get(val, 0) + count

        if not global_counts:
            continue

        # Ordenar y tomar top-N
        sorted_items = sorted(global_counts.items(), key=lambda x: -x[1])
        n_show = 20 if var_name == "property_state" else min(15, len(sorted_items))
        top_items = sorted_items[:n_show]

        categories = [item[0] for item in top_items]
        values = [item[1] for item in top_items]
        total = sum(global_counts.values())

        # Gráfico de barras horizontal
        fig, ax = plt.subplots(figsize=(12, max(5, n_show * 0.4)))
        colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, n_show))
        ax.barh(range(n_show), values, color=colors_grad, alpha=0.85, edgecolor="none")
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlabel("Frecuencia Total")
        ax.set_title(f"{title}\n(Total: {total:,.0f} registros)")
        ax.invert_yaxis()

        # Añadir porcentajes
        for i, (cat, val) in enumerate(zip(categories, values)):
            pct = val / total * 100
            ax.text(val + total * 0.005, i, f"{pct:.1f}%", va="center", fontsize=8)

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
        ax.grid(axis="x", alpha=0.3)

        save_figure(fig, EDA_FIGURES_DIR / f"12_cat_{var_name}.png")

    log_message(f"  Análisis categórico completado para {len(cat_vars)} variables")


# ============================================================================
# 7. MATRIZ DE CORRELACIÓN (muestreo eficiente)
# ============================================================================

def generate_correlation_matrix(
    profiles: Dict[str, Dict],
    sample_trimestres: int = 5,
    sample_rows: int = 200_000,
) -> Optional[pd.DataFrame]:
    """
    Genera matriz de correlación muestreando Parquets eficientemente.

    Solo lee columnas numéricas clave de unos pocos trimestres.
    """
    log_message(f"Generando matriz de correlación (muestra de {sample_trimestres} trimestres)...")

    # Verificar RAM antes de empezar
    if not check_memory_threshold(min_available_gb=4.0):
        log_message("RAM insuficiente para correlación, omitiendo", level="warn")
        return None

    # Seleccionar trimestres distribuidos uniformemente
    completed = get_completed_trimestres()
    if len(completed) < 3:
        log_message("Insuficientes trimestres completados para correlación", level="warn")
        return None

    step = max(1, len(completed) // sample_trimestres)
    selected = completed[::step][:sample_trimestres]

    # Columnas numéricas clave
    target_cols = KEY_NUMERIC_COLUMNS.copy()

    # Leer muestras de Parquet
    all_samples = []
    for t in selected:
        parquet_files = sorted(PANEL_PATH.glob(f"{t}*.parquet"))
        if not parquet_files:
            continue

        try:
            # Leer solo columnas numéricas del primer archivo
            df = pd.read_parquet(parquet_files[0], columns=target_cols)
            if len(df) > sample_rows:
                df = df.sample(n=sample_rows, random_state=42)
            all_samples.append(df)
            del df
        except Exception as e:
            log_message(f"  Error leyendo {t}: {e}", level="warn")
            continue

    if not all_samples:
        log_message("No se pudieron leer Parquets para correlación", level="warn")
        return None

    combined = pd.concat(all_samples, ignore_index=True)
    del all_samples
    gc.collect()

    # Calcular correlación
    # Convertir a numérico
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    corr = combined.corr()
    del combined
    gc.collect()

    # Guardar
    corr.to_csv(EDA_TABLES_DIR / "eda_matriz_correlacion.csv")

    # Gráfico heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    data = corr.values.copy()
    data[mask] = np.nan

    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    col_labels = [c.replace("_", "\n") for c in corr.columns]
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(col_labels, fontsize=8)

    # Anotar valores
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not mask[i, j]:
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Coeficiente de Correlación de Pearson")
    ax.set_title("Matriz de Correlación — Variables Numéricas Clave\n"
                 f"(Muestra: {sample_trimestres} trimestres, {sample_rows:,} filas/trimestre)",
                 fontsize=14, fontweight="bold")
    save_figure(fig, EDA_FIGURES_DIR / "13_matriz_correlacion.png")

    log_message(f"  Correlación calculada: {len(corr)} × {len(corr)} variables")
    return corr


# ============================================================================
# 8. TABLA DESCRIPTIVA GLOBAL
# ============================================================================

def generate_global_descriptive_table(profiles: Dict[str, Dict]) -> pd.DataFrame:
    """
    Genera tabla descriptiva global consolidada (110 columnas × estadísticas).
    Similar a df.describe() pero calculada desde los perfiles acumulados.
    """
    log_message("Generando tabla descriptiva global consolidada...")

    all_columns = set()
    for p in profiles.values():
        all_columns.update(p.get("columns", {}).keys())

    rows = []
    for col_name in sorted(all_columns):
        stats = _compute_global_stats(profiles, col_name)

        # Obtener tipo y nulidad
        col_types = []
        null_pcts = []
        n_totals = []
        entropies = []
        n_uniques = []
        for p in profiles.values():
            col = p.get("columns", {}).get(col_name, {})
            if col:
                col_types.append(col.get("column_type", "unknown"))
                null_pcts.append(col.get("null_pct", 0))
                n_totals.append(col.get("n_total", 0))
                if col.get("n_unique"):
                    n_uniques.append(col["n_unique"])
                if col.get("entropy"):
                    entropies.append(col["entropy"])

        col_type = max(set(col_types), key=col_types.count) if col_types else "unknown"

        row = {
            "columna": col_name,
            "tipo": col_type,
            "n_total_global": sum(n_totals),
            "null_pct_medio": np.mean(null_pcts) if null_pcts else None,
            "null_pct_max": max(null_pcts) if null_pcts else None,
            "trimestres_con_datos": len(null_pcts),
        }

        if stats:
            row.update({
                "media_global": stats["mean"],
                "std_global": stats["std"],
                "min_global": stats["min"],
                "max_global": stats["max"],
                "n_valid_global": stats["n_valid"],
            })

        if n_uniques:
            row["n_unique_medio"] = np.mean(n_uniques)
        if entropies:
            row["entropia_media"] = np.mean(entropies)

        rows.append(row)

    df_desc = pd.DataFrame(rows)
    df_desc.to_csv(EDA_TABLES_DIR / "eda_tabla_descriptiva_global.csv", index=False)
    log_message(f"  Tabla descriptiva: {len(df_desc)} columnas")

    return df_desc


# ============================================================================
# 9. CONSOLIDAR PERFIL GLOBAL JSON
# ============================================================================

def consolidate_global_profile(profiles: Dict[str, Dict]) -> Dict:
    """
    Consolida todos los perfiles individuales en un único perfil global JSON.
    Este es el input para las fases posteriores.
    """
    log_message("Consolidando perfil global JSON...")

    global_profile = {
        "n_trimestres": len(profiles),
        "trimestres": sorted(profiles.keys()),
        "total_rows": sum(p.get("total_rows", 0) for p in profiles.values()),
        "total_parquet_size_gb": sum(p.get("parquet_size_gb", 0) for p in profiles.values()),
        "total_csv_size_gb": sum(p.get("file_size_gb", 0) for p in profiles.values()),
        "total_processing_seconds": sum(p.get("processing_time_seconds", 0) for p in profiles.values()),
        "columns": {},
    }

    # Consolidar por columna
    all_columns = set()
    for p in profiles.values():
        all_columns.update(p.get("columns", {}).keys())

    for col_name in sorted(all_columns):
        stats = _compute_global_stats(profiles, col_name)

        # Tipo, nulidad
        col_types = []
        null_pcts = []
        for p in profiles.values():
            col = p.get("columns", {}).get(col_name, {})
            if col:
                col_types.append(col.get("column_type", "unknown"))
                null_pcts.append(col.get("null_pct", 0))

        col_entry = {
            "column_type": max(set(col_types), key=col_types.count) if col_types else "unknown",
            "null_pct_mean": round(np.mean(null_pcts), 4) if null_pcts else None,
            "null_pct_max": round(max(null_pcts), 4) if null_pcts else None,
            "n_trimestres": len(null_pcts),
        }

        if stats:
            col_entry.update({
                "mean": round(stats["mean"], 6),
                "std": round(stats["std"], 6),
                "min": stats["min"],
                "max": stats["max"],
                "n_valid": stats["n_valid"],
            })

        global_profile["columns"][col_name] = col_entry

    # Guardar
    output_path = DATA_PROCESSED_PATH / "perfil_global.json"
    with open(output_path, "w") as f:
        json.dump(global_profile, f, indent=2, default=str)

    log_message(f"  Perfil global guardado: {output_path.name}")
    log_message(f"  Total filas: {global_profile['total_rows']:,}")
    log_message(f"  Total Parquet: {global_profile['total_parquet_size_gb']:.2f} GB")

    return global_profile


# ============================================================================
# 10. RESUMEN VISUAL FINAL
# ============================================================================

def generate_summary_dashboard(
    profiles: Dict[str, Dict],
    df_summary: pd.DataFrame,
) -> None:
    """
    Genera un dashboard resumen de una sola página con las métricas principales.
    """
    log_message("Generando dashboard resumen...")

    total_rows = sum(p.get("total_rows", 0) for p in profiles.values())
    total_csv_gb = sum(p.get("file_size_gb", 0) for p in profiles.values())
    total_pq_gb = sum(p.get("parquet_size_gb", 0) for p in profiles.values())
    total_time = sum(p.get("processing_time_seconds", 0) for p in profiles.values())
    n_trimestres = len(profiles)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Dashboard Resumen — EDA Masivo Freddie Mac Performance Data",
                 fontsize=18, fontweight="bold", y=0.98)

    # 1. KPIs
    ax = axes[0, 0]
    ax.axis("off")
    kpis = [
        f"Trimestres procesados:  {n_trimestres}/{TOTAL_CSV_FILES}",
        f"Total filas:  {total_rows:,.0f}",
        f"Tamaño CSV:  {total_csv_gb:.1f} GB",
        f"Tamaño Parquet:  {total_pq_gb:.2f} GB",
        f"Compresión media:  {total_csv_gb/total_pq_gb:.1f}x" if total_pq_gb > 0 else "",
        f"Tiempo total:  {timedelta(seconds=int(total_time))}",
        f"Velocidad media:  {total_rows/total_time:,.0f} filas/s" if total_time > 0 else "",
    ]
    for i, kpi in enumerate(kpis):
        if kpi:
            ax.text(0.05, 0.9 - i * 0.13, kpi, transform=ax.transAxes,
                    fontsize=12, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", alpha=0.8))
    ax.set_title("Indicadores Clave", fontsize=14, fontweight="bold")

    # 2. Filas por trimestre (mini bar)
    ax = axes[0, 1]
    if not df_summary.empty:
        ax.bar(range(len(df_summary)), df_summary["filas"] / 1e6,
               color=PALETTE_CLUSTERS[0], alpha=0.7, width=0.8)
        ax.set_title("Volumen por Trimestre (M filas)", fontsize=11)
        ax.set_xlabel("Trimestre (índice)")
        ax.set_ylabel("Millones")
    ax.grid(axis="y", alpha=0.3)

    # 3. Compresión por trimestre
    ax = axes[0, 2]
    if not df_summary.empty:
        ax.bar(range(len(df_summary)), df_summary["ratio_compresion"],
               color=PALETTE_CLUSTERS[2], alpha=0.7, width=0.8)
        ax.set_title("Ratio Compresión CSV→Parquet", fontsize=11)
        ax.set_xlabel("Trimestre (índice)")
    ax.grid(axis="y", alpha=0.3)

    # 4. FICO medio temporal
    ax = axes[1, 0]
    fico_means = []
    for t in sorted(profiles.keys()):
        col = profiles[t].get("columns", {}).get("borrower_credit_score_at_origination", {})
        val = col.get("mean")
        fico_means.append(val)
    valid = [(i, v) for i, v in enumerate(fico_means) if v is not None]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                color=PALETTE_CLUSTERS[3], linewidth=2, marker=".", markersize=3)
    ax.set_title("FICO Medio por Trimestre", fontsize=11)
    ax.set_xlabel("Trimestre (índice)")
    ax.grid(True, alpha=0.3)

    # 5. LTV medio temporal
    ax = axes[1, 1]
    ltv_means = []
    for t in sorted(profiles.keys()):
        col = profiles[t].get("columns", {}).get("original_loan_to_value_ratio_ltv", {})
        val = col.get("mean")
        ltv_means.append(val)
    valid = [(i, v) for i, v in enumerate(ltv_means) if v is not None]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                color=PALETTE_CLUSTERS[1], linewidth=2, marker=".", markersize=3)
    ax.set_title("LTV Original Medio (%)", fontsize=11)
    ax.set_xlabel("Trimestre (índice)")
    ax.grid(True, alpha=0.3)

    # 6. Tasa de interés media temporal
    ax = axes[1, 2]
    rate_means = []
    for t in sorted(profiles.keys()):
        col = profiles[t].get("columns", {}).get("current_interest_rate", {})
        val = col.get("mean")
        rate_means.append(val)
    valid = [(i, v) for i, v in enumerate(rate_means) if v is not None]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                color=PALETTE_CLUSTERS[4], linewidth=2, marker=".", markersize=3)
    ax.set_title("Tasa Interés Actual Media (%)", fontsize=11)
    ax.set_xlabel("Trimestre (índice)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "00_dashboard_resumen.png")
    log_message("  Dashboard resumen generado")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta el EDA masivo completo."""

    print("=" * 70)
    print("  EDA MASIVO — Portafolio Hipotecario Freddie Mac")
    print("  Script: 02_eda_masivo.py")
    print("=" * 70)
    print_system_summary()

    with ProcessingTimer("EDA Masivo"):

        # ── Verificar que hay perfiles disponibles ──
        completed = get_completed_trimestres()
        log_message(f"Trimestres COMPLETADOS en checkpoint: {len(completed)}/{TOTAL_CSV_FILES}")

        if len(completed) == 0:
            log_message("❌ No hay trimestres completados. Ejecutar primero 01_construccion_panel.py",
                        level="error")
            sys.exit(1)

        if len(completed) < TOTAL_CSV_FILES:
            log_message(
                f"⚠️ Solo {len(completed)}/{TOTAL_CSV_FILES} completados. "
                f"El EDA se generará con datos parciales.",
                level="warn"
            )

        # ── Cargar perfiles ──
        log_message("Cargando perfiles JSON...")
        profiles = load_all_profiles()

        # Filtrar: solo perfiles de trimestres realmente COMPLETADOS
        profiles = {k: v for k, v in profiles.items() if k in completed}
        log_message(f"  {len(profiles)} perfiles cargados y validados")

        if len(profiles) == 0:
            log_message("❌ No hay perfiles válidos para analizar", level="error")
            sys.exit(1)

        # ── Fase 1: Resumen de procesamiento ──
        print("\n" + "─" * 50)
        print("  FASE 1: Resumen de Procesamiento")
        print("─" * 50)
        df_summary = generate_processing_summary(profiles)
        plot_processing_summary(df_summary)
        print_memory_status("Post-resumen")

        # ── Fase 2: Mapa de nulidad ──
        print("\n" + "─" * 50)
        print("  FASE 2: Mapa de Nulidad Temporal")
        print("─" * 50)
        df_null = generate_nullity_heatmap(profiles)
        print_memory_status("Post-nulidad")

        # ── Fase 3: Evolución temporal ──
        print("\n" + "─" * 50)
        print("  FASE 3: Evolución Temporal de Variables Financieras")
        print("─" * 50)
        df_temporal = generate_temporal_evolution(profiles)
        print_memory_status("Post-temporal")

        # ── Fase 4: Distribuciones ──
        print("\n" + "─" * 50)
        print("  FASE 4: Distribuciones de Variables Clave")
        print("─" * 50)
        generate_distribution_plots(profiles)
        clear_memory()
        print_memory_status("Post-distribuciones")

        # ── Fase 5: Ranking de informatividad ──
        print("\n" + "─" * 50)
        print("  FASE 5: Ranking de Informatividad")
        print("─" * 50)
        df_info = generate_informativity_ranking(profiles)
        print_memory_status("Post-ranking")

        # ── Fase 6: Análisis categórico ──
        print("\n" + "─" * 50)
        print("  FASE 6: Análisis de Variables Categóricas")
        print("─" * 50)
        generate_categorical_analysis(profiles)
        clear_memory()
        print_memory_status("Post-categórico")

        # ── Fase 7: Matriz de correlación ──
        print("\n" + "─" * 50)
        print("  FASE 7: Matriz de Correlación")
        print("─" * 50)
        corr = generate_correlation_matrix(profiles)
        clear_memory()
        print_memory_status("Post-correlación")

        # ── Fase 8: Tabla descriptiva global ──
        print("\n" + "─" * 50)
        print("  FASE 8: Tabla Descriptiva Global")
        print("─" * 50)
        df_desc = generate_global_descriptive_table(profiles)

        # ── Fase 9: Perfil global consolidado ──
        print("\n" + "─" * 50)
        print("  FASE 9: Consolidación del Perfil Global")
        print("─" * 50)
        global_profile = consolidate_global_profile(profiles)

        # ── Fase 10: Dashboard resumen ──
        print("\n" + "─" * 50)
        print("  FASE 10: Dashboard Resumen")
        print("─" * 50)
        generate_summary_dashboard(profiles, df_summary)

        # ── Resumen final ──
        print("\n" + "=" * 70)
        print("  RESUMEN FINAL DEL EDA MASIVO")
        print("=" * 70)

        n_figures = len(list(EDA_FIGURES_DIR.glob("*.png")))
        n_tables = len(list(EDA_TABLES_DIR.glob("eda_*.csv")))
        log_message(f"  📊 Figuras generadas: {n_figures} (en {EDA_FIGURES_DIR})")
        log_message(f"  📋 Tablas generadas: {n_tables} (en {EDA_TABLES_DIR})")
        log_message(f"  📁 Perfil global: {DATA_PROCESSED_PATH / 'perfil_global.json'}")
        log_message(f"  📈 Trimestres analizados: {len(profiles)}/{TOTAL_CSV_FILES}")

        if len(profiles) < TOTAL_CSV_FILES:
            log_message(
                f"\n  ⚠️ NOTA: EDA parcial ({len(profiles)}/{TOTAL_CSV_FILES}). "
                f"Re-ejecutar cuando todos los trimestres estén completos "
                f"para obtener el análisis definitivo.",
                level="warn"
            )

    print_memory_status("Final")


if __name__ == "__main__":
    main()
