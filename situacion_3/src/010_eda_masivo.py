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

# Librerías opcionales para EDA avanzado
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_PROCESSED_PATH, PANEL_PATH, PERFILES_PATH,
    FIGURES_PATH, TABLES_PATH, FIGURES_SUBDIRS,
    PERFORMANCE_COLUMNS, KEY_NUMERIC_COLUMNS, KEY_CATEGORICAL_COLUMNS,
    LOAN_ID_COLUMN, DATE_COLUMN, TOTAL_CSV_FILES,
    PARQUET_COMPRESSION, FIGURE_DPI,
    CHECKPOINTS_PATH, PROCESSING_STATE_FILE,
    OLD_TO_NEW_COLUMN_MAP, SENTINEL_VALUES, EXPECTED_NUMERIC_COLUMNS,
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
# RESOLUCIÓN DE NOMBRES (compatibilidad OLD ↔ NEW)
# ============================================================================

# Mapeo inverso: nombre_nuevo → nombre_viejo
NEW_TO_OLD_MAP: Dict[str, str] = {v: k for k, v in OLD_TO_NEW_COLUMN_MAP.items()}


def resolve_col(name: str, available) -> Optional[str]:
    """Resuelve nombre de columna: intenta new, fallback a old."""
    avail_set = set(available) if not isinstance(available, set) else available
    if name in avail_set:
        return name
    old = NEW_TO_OLD_MAP.get(name)
    if old and old in avail_set:
        return old
    return None


def get_profile_col(columns_dict: dict, col_name: str) -> Optional[dict]:
    """Busca datos de columna en perfil JSON con fallback a nombre viejo."""
    if col_name in columns_dict:
        return columns_dict[col_name]
    old = NEW_TO_OLD_MAP.get(col_name)
    if old and old in columns_dict:
        return columns_dict[old]
    return None


def resolve_parquet_columns(
    requested: List[str], available: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Resuelve columnas de Parquet: retorna (cols_para_leer, rename_map).
    Si un nombre NEW no existe pero el OLD sí, lo usa y anota el rename.
    """
    avail_set = set(available)
    cols_to_read: List[str] = []
    rename_map: Dict[str, str] = {}
    for name in requested:
        if name in avail_set:
            cols_to_read.append(name)
        else:
            old = NEW_TO_OLD_MAP.get(name)
            if old and old in avail_set:
                cols_to_read.append(old)
                rename_map[old] = name
    return cols_to_read, rename_map


def normalize_profile_keys(profiles: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Renombra claves de columnas en perfiles de OLD → NEW.
    Así todo el código posterior puede usar nombres nuevos sin fallback.
    """
    for t, profile in profiles.items():
        cols = profile.get("columns", {})
        new_cols = {}
        for col_name, col_data in cols.items():
            new_name = OLD_TO_NEW_COLUMN_MAP.get(col_name, col_name)
            new_cols[new_name] = col_data
        profile["columns"] = new_cols
    return profiles


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


def _get_parquet_schema(path: Path) -> List[str]:
    """Obtiene nombres de columnas de un Parquet sin leer datos."""
    if PYARROW_AVAILABLE:
        return pq.read_schema(path).names
    return pd.read_parquet(path, columns=[]).columns.tolist()


def load_parquet_sample(
    n_trimestres: int = 8,
    rows_per_trimestre: int = 100_000,
    columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Carga muestra estratificada de Parquets para EDA avanzado.
    Resuelve automáticamente nombres old/new.

    Args:
        n_trimestres: Cantidad de trimestres a muestrear
        rows_per_trimestre: Filas máximas por trimestre
        columns: Columnas específicas (None = todas)
        random_state: Semilla para reproducibilidad

    Returns:
        DataFrame con columnas renombradas a nombres NEW
    """
    if not check_memory_threshold(min_available_gb=3.0):
        log_message("RAM insuficiente para sampling de Parquets", level="warn")
        return pd.DataFrame()

    completed = get_completed_trimestres()
    if not completed:
        log_message("No hay trimestres completados para sampling", level="warn")
        return pd.DataFrame()

    step = max(1, len(completed) // n_trimestres)
    selected = completed[::step][:n_trimestres]
    log_message(f"  Sampling: {len(selected)} trimestres ({selected[0]}…{selected[-1]})")

    all_samples: List[pd.DataFrame] = []
    for t in selected:
        parquet_files = sorted(PANEL_PATH.glob(f"{t}*.parquet"))
        if not parquet_files:
            continue
        try:
            pf = parquet_files[0]
            schema_cols = _get_parquet_schema(pf)

            if columns:
                cols_to_read, rename_map = resolve_parquet_columns(columns, schema_cols)
                if not cols_to_read:
                    continue
            else:
                cols_to_read = None
                rename_map = {c: OLD_TO_NEW_COLUMN_MAP[c]
                              for c in schema_cols if c in OLD_TO_NEW_COLUMN_MAP}

            df = pd.read_parquet(pf, columns=cols_to_read)
            if len(df) > rows_per_trimestre:
                df = df.sample(n=rows_per_trimestre, random_state=random_state)
            if rename_map:
                df = df.rename(columns=rename_map)
            df["_trimestre"] = t
            all_samples.append(df)
            del df
        except Exception as e:
            log_message(f"  Error sample {t}: {e}", level="warn")

    if not all_samples:
        log_message("No se pudieron cargar muestras de Parquets", level="warn")
        return pd.DataFrame()

    result = pd.concat(all_samples, ignore_index=True)
    del all_samples
    gc.collect()

    # Coerce columnas numéricas conocidas
    num_candidates = set(KEY_NUMERIC_COLUMNS) | set(EXPECTED_NUMERIC_COLUMNS)
    for col in result.columns:
        if col.startswith("_"):
            continue
        if col in num_candidates:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    log_message(f"  Sample: {len(result):,} filas × {result.shape[1]} cols")
    return result


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
# BLOQUE A — GRÁFICOS PRE-AFE (FIGURAS 14-19)
# ============================================================================

def generate_correlation_by_domains(sample_df: pd.DataFrame) -> None:
    """Fig 14: Matrices de correlación separadas por dominio funcional."""
    log_message("Fig 14: Correlación por dominios...")

    domains = {
        "Originación": [
            "borrower_credit_score_at_origination", "original_loan_to_value_ratio_ltv",
            "debt_to_income_dti", "original_interest_rate", "original_upb",
            "original_loan_term", "number_of_borrowers", "mortgage_insurance_percentage",
        ],
        "Performance Mensual": [
            "current_interest_rate", "current_actual_upb", "loan_age",
            "remaining_months_to_maturity",
        ],
        "Liquidación": [
            "foreclosure_costs", "asset_recovery_costs",
            "miscellaneous_holding_expenses_and_credits",
            "associated_taxes_for_holding_property",
            "net_sales_proceeds", "credit_enhancement_proceeds",
        ],
        "Temporal": [
            "loan_age", "remaining_months_to_maturity",
            "remaining_months_to_legal_maturity", "original_loan_term",
        ],
    }

    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    fig.suptitle("Matrices de Correlación por Dominio Funcional",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, (domain_name, cols) in enumerate(domains.items()):
        ax = axes[idx // 2, idx % 2]
        avail = [c for c in cols if c in sample_df.columns]

        if len(avail) < 2:
            ax.text(0.5, 0.5, f"Datos insuficientes\npara «{domain_name}»",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(domain_name, fontsize=13, fontweight="bold")
            continue

        corr = sample_df[avail].dropna(how="all").corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        im = ax.imshow(corr.where(~mask).values, cmap="RdBu_r",
                       vmin=-1, vmax=1, aspect="auto")
        short = [c.replace("_at_origination", "").replace("original_", "o_")[:22]
                 for c in corr.columns]
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(f"Dominio: {domain_name}", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Anotar valores r
        for i in range(len(corr)):
            for j in range(i):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(val) > 0.5 else "black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, EDA_FIGURES_DIR / "14_correlacion_por_dominios.png")


def generate_vif_heatmap(sample_df: pd.DataFrame) -> None:
    """Fig 15: Heatmap del Factor de Inflación de Varianza (VIF)."""
    log_message("Fig 15: VIF multicolinealidad...")

    target = [
        "borrower_credit_score_at_origination", "original_loan_to_value_ratio_ltv",
        "debt_to_income_dti", "original_interest_rate", "original_upb",
        "original_loan_term", "loan_age", "current_interest_rate",
        "current_actual_upb", "mortgage_insurance_percentage",
        "number_of_borrowers", "remaining_months_to_maturity",
    ]
    avail = [c for c in target if c in sample_df.columns]
    if len(avail) < 3:
        log_message("  Insuficientes columnas para VIF", level="warn")
        return

    sub = sample_df[avail].dropna().sample(n=min(50_000, len(sample_df)), random_state=42)
    if len(sub) < 100:
        return

    # VIF = diagonal inversa de la correlación invertida
    corr = sub.corr().values
    try:
        corr_inv = np.linalg.inv(corr)
        vif_values = np.diag(corr_inv)
    except np.linalg.LinAlgError:
        corr_reg = corr + np.eye(len(corr)) * 1e-6
        corr_inv = np.linalg.inv(corr_reg)
        vif_values = np.diag(corr_inv)

    vif_df = pd.DataFrame({"variable": avail, "VIF": vif_values}).sort_values("VIF", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(avail) * 0.45)))
    colors = ["#d62728" if v > 10 else "#ff7f0e" if v > 5 else "#2ca02c" for v in vif_df["VIF"]]
    ax.barh(range(len(vif_df)), vif_df["VIF"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(vif_df)))
    ax.set_yticklabels([v[:30] for v in vif_df["variable"]], fontsize=9)
    ax.axvline(x=5, color="orange", ls="--", alpha=0.7, label="Umbral moderado (5)")
    ax.axvline(x=10, color="red", ls="--", alpha=0.7, label="Umbral severo (10)")
    ax.set_xlabel("VIF")
    ax.set_title("Factor de Inflación de Varianza (VIF) — Multicolinealidad",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(vif_df["VIF"]):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8)

    save_figure(fig, EDA_FIGURES_DIR / "15_heatmap_vif_multicolinealidad.png")
    vif_df.to_csv(EDA_TABLES_DIR / "eda_vif_multicolinealidad.csv", index=False)


def generate_skewness_kurtosis_scatter(sample_df: pd.DataFrame) -> None:
    """Fig 16: Scatter de asimetría vs curtosis para variables numéricas."""
    if not SCIPY_AVAILABLE:
        log_message("  scipy no disponible, omitiendo fig 16", level="warn")
        return
    log_message("Fig 16: Scatter asimetría vs curtosis...")

    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

    results = []
    for col in numeric_cols:
        data = sample_df[col].dropna()
        if len(data) < 30:
            continue
        sk = float(scipy_stats.skew(data, nan_policy="omit"))
        ku = float(scipy_stats.kurtosis(data, nan_policy="omit"))
        results.append({"variable": col, "skewness": sk, "kurtosis": ku})

    if not results:
        return

    df_sk = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(df_sk["skewness"], df_sk["kurtosis"],
                         c=np.abs(df_sk["skewness"]), cmap="YlOrRd",
                         s=80, alpha=0.7, edgecolors="gray", linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label="|Asimetría|")

    # Anotar variables extremas
    extreme = df_sk.nlargest(8, "kurtosis")
    for _, row in extreme.iterrows():
        ax.annotate(row["variable"][:25], (row["skewness"], row["kurtosis"]),
                    fontsize=7, alpha=0.8, textcoords="offset points", xytext=(5, 5))

    ax.axhline(y=0, color="gray", ls="--", alpha=0.4)
    ax.axvline(x=0, color="gray", ls="--", alpha=0.4)
    ax.axhline(y=3, color="blue", ls=":", alpha=0.4, label="Curtosis normal (3)")
    ax.axhspan(-2, 2, xmin=0, xmax=1, alpha=0.05, color="green")
    ax.set_xlabel("Asimetría (Skewness)", fontsize=12)
    ax.set_ylabel("Curtosis (Kurtosis)", fontsize=12)
    ax.set_title("Mapa de Asimetría vs Curtosis — Variables Numéricas",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "16_skewness_kurtosis_scatter.png")
    df_sk.to_csv(EDA_TABLES_DIR / "eda_skewness_kurtosis.csv", index=False)


def generate_qq_grid(sample_df: pd.DataFrame) -> None:
    """Fig 17: Grid de QQ-plots para las 8 variables numéricas más relevantes."""
    if not SCIPY_AVAILABLE:
        log_message("  scipy no disponible, omitiendo fig 17", level="warn")
        return
    log_message("Fig 17: QQ-plots grid...")

    target = [
        "borrower_credit_score_at_origination", "original_loan_to_value_ratio_ltv",
        "debt_to_income_dti", "original_interest_rate",
        "current_interest_rate", "original_upb", "current_actual_upb", "loan_age",
    ]
    avail = [c for c in target if c in sample_df.columns][:8]
    if not avail:
        return

    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    fig.suptitle("QQ-Plots de Normalidad — Variables Clave",
                 fontsize=15, fontweight="bold")

    for idx, col in enumerate(avail):
        ax = axes[idx // ncols, idx % ncols]
        data = sample_df[col].dropna()
        if len(data) > 20_000:
            data = data.sample(20_000, random_state=42)
        if len(data) < 10:
            ax.set_title(col[:25], fontsize=9)
            continue

        scipy_stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(col.replace("_", " ")[:28], fontsize=9, fontweight="bold")
        ax.get_lines()[0].set(markersize=1, alpha=0.3)
        ax.grid(True, alpha=0.3)

    # Ocultar ejes sobrantes
    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "17_test_normalidad_qq_grid.png")


def generate_transformation_comparison(sample_df: pd.DataFrame) -> None:
    """Fig 18: Comparación de distribuciones original vs log vs sqrt."""
    log_message("Fig 18: Comparación de transformaciones...")

    target = [
        "original_upb", "current_actual_upb", "original_interest_rate",
        "borrower_credit_score_at_origination",
    ]
    avail = [c for c in target if c in sample_df.columns]
    if not avail:
        return

    fig, axes = plt.subplots(len(avail), 3, figsize=(18, 4 * len(avail)))
    if len(avail) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Comparación de Transformaciones: Original vs Log vs √",
                 fontsize=14, fontweight="bold")

    for i, col in enumerate(avail):
        data = sample_df[col].dropna()
        if len(data) > 50_000:
            data = data.sample(50_000, random_state=42)
        data = data[data > 0]  # log/sqrt requieren positivos
        if len(data) < 30:
            continue

        # Original
        axes[i, 0].hist(data, bins=80, color=PALETTE_CLUSTERS[0], alpha=0.7, edgecolor="none")
        axes[i, 0].set_title(f"Original: {col[:28]}", fontsize=9)
        axes[i, 0].set_ylabel("Frecuencia")

        # Log
        log_data = np.log1p(data)
        axes[i, 1].hist(log_data, bins=80, color=PALETTE_CLUSTERS[2], alpha=0.7, edgecolor="none")
        axes[i, 1].set_title("log(1+x)", fontsize=9)

        # Sqrt
        sqrt_data = np.sqrt(data)
        axes[i, 2].hist(sqrt_data, bins=80, color=PALETTE_CLUSTERS[4], alpha=0.7, edgecolor="none")
        axes[i, 2].set_title("√x", fontsize=9)

        for ax in axes[i]:
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, EDA_FIGURES_DIR / "18_transformaciones_comparadas.png")


def generate_nullity_correlation(profiles: Dict[str, Dict]) -> None:
    """Fig 19: Mapa de correlación entre patrones de nulidad."""
    log_message("Fig 19: Correlación de nulidad...")

    all_cols: set = set()
    for p in profiles.values():
        all_cols.update(p.get("columns", {}).keys())
    all_cols = sorted(all_cols)

    if len(all_cols) < 5:
        return

    # Matriz de null_pct: trimestres × columnas
    null_matrix = []
    for t in sorted(profiles.keys()):
        cols_data = profiles[t].get("columns", {})
        total_rows = profiles[t].get("total_rows", 1)
        row = []
        for c in all_cols:
            if c in cols_data:
                null_pct = cols_data[c].get("null_pct", 0)
            else:
                null_pct = 100.0
            row.append(null_pct)
        null_matrix.append(row)

    null_df = pd.DataFrame(null_matrix, columns=all_cols)
    # Solo columnas con variación en nulidad
    varying = null_df.columns[null_df.std() > 1.0]
    if len(varying) < 3:
        log_message("  Pocas columnas con variación de nulidad", level="warn")
        return

    null_corr = null_df[varying].corr()

    fig, ax = plt.subplots(figsize=(max(12, len(varying) * 0.35),
                                     max(10, len(varying) * 0.3)))
    im = ax.imshow(null_corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    short_names = [c[:20] for c in null_corr.columns]
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=5)
    ax.set_title("Correlación entre Patrones de Nulidad",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Correlación")
    save_figure(fig, EDA_FIGURES_DIR / "19_mapa_nulidad_correlacion.png")


# ============================================================================
# BLOQUE B — GRÁFICOS TEMPORALES / LONGITUDINALES (FIGURAS 20-25)
# ============================================================================

def generate_sequence_lengths(sample_df: pd.DataFrame) -> None:
    """Fig 20: Distribución de longitud de secuencias por préstamo."""
    log_message("Fig 20: Longitud de secuencias...")

    loan_col = resolve_col(LOAN_ID_COLUMN, sample_df.columns)
    if loan_col is None:
        log_message("  Columna loan_id no encontrada", level="warn")
        return

    seq_counts = sample_df[loan_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribución de Observaciones por Préstamo",
                 fontsize=14, fontweight="bold")

    # Histograma
    axes[0].hist(seq_counts.values, bins=50, color=PALETTE_CLUSTERS[0],
                 alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("Nro. de observaciones mensuales")
    axes[0].set_ylabel("Cantidad de préstamos")
    axes[0].set_title("Histograma de longitud de secuencia")
    axes[0].axvline(seq_counts.median(), color="red", ls="--",
                    label=f"Mediana: {seq_counts.median():.0f}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # ECDF
    sorted_vals = np.sort(seq_counts.values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1].plot(sorted_vals, ecdf, color=PALETTE_CLUSTERS[3], linewidth=2)
    axes[1].set_xlabel("Nro. de observaciones")
    axes[1].set_ylabel("Probabilidad acumulada")
    axes[1].set_title("ECDF de longitud de secuencia")
    axes[1].grid(True, alpha=0.3)

    stats_text = (f"Media: {seq_counts.mean():.1f}\n"
                  f"Mediana: {seq_counts.median():.0f}\n"
                  f"Max: {seq_counts.max()}\n"
                  f"Préstamos únicos: {len(seq_counts):,}")
    axes[1].text(0.95, 0.05, stats_text, transform=axes[1].transAxes,
                 fontsize=9, va="bottom", ha="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "20_distribucion_longitud_secuencias.png")


def generate_markov_transition(sample_df: pd.DataFrame) -> None:
    """Fig 21: Matriz de transición Markov del estado de delinquency."""
    log_message("Fig 21: Matriz de transición Markov...")

    loan_col = resolve_col(LOAN_ID_COLUMN, sample_df.columns)
    delin_col = resolve_col("current_loan_delinquency_status", sample_df.columns)
    date_col = resolve_col(DATE_COLUMN, sample_df.columns)

    if not all([loan_col, delin_col]):
        log_message("  Columnas necesarias no encontradas para Markov", level="warn")
        return

    # Preparar datos
    df = sample_df[[loan_col, delin_col]].copy()
    df[delin_col] = pd.to_numeric(df[delin_col], errors="coerce")
    df = df.dropna(subset=[delin_col])

    # Crear estados categóricos
    def state_label(x):
        if x == 0:
            return "Corriente"
        elif x == 1:
            return "30 días"
        elif x == 2:
            return "60 días"
        elif x <= 5:
            return "90-150 días"
        else:
            return "Grave (180+)"

    df["estado"] = df[delin_col].apply(state_label)

    # Calcular transiciones por préstamo
    transitions = {}
    for _, group in df.groupby(loan_col):
        states = group["estado"].values
        for i in range(len(states) - 1):
            pair = (states[i], states[i + 1])
            transitions[pair] = transitions.get(pair, 0) + 1

    if not transitions:
        return

    state_order = ["Corriente", "30 días", "60 días", "90-150 días", "Grave (180+)"]
    trans_matrix = pd.DataFrame(0, index=state_order, columns=state_order, dtype=float)

    for (s_from, s_to), count in transitions.items():
        if s_from in state_order and s_to in state_order:
            trans_matrix.loc[s_from, s_to] = count

    # Normalizar filas a probabilidades
    row_sums = trans_matrix.sum(axis=1)
    prob_matrix = trans_matrix.div(row_sums, axis=0).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Matriz de Transición de Estados de Pago (Markov)",
                 fontsize=15, fontweight="bold")

    # Frecuencias absolutas
    im0 = axes[0].imshow(trans_matrix.values, cmap="YlOrRd", aspect="auto")
    for i in range(len(state_order)):
        for j in range(len(state_order)):
            val = trans_matrix.iloc[i, j]
            if val > 0:
                axes[0].text(j, i, f"{val:,.0f}", ha="center", va="center",
                             fontsize=8, color="white" if val > trans_matrix.values.max() * 0.5 else "black")
    axes[0].set_xticks(range(len(state_order)))
    axes[0].set_xticklabels(state_order, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(state_order)))
    axes[0].set_yticklabels(state_order, fontsize=9)
    axes[0].set_title("Frecuencias de Transición")
    axes[0].set_xlabel("Estado destino (t+1)")
    axes[0].set_ylabel("Estado origen (t)")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    # Probabilidades
    im1 = axes[1].imshow(prob_matrix.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for i in range(len(state_order)):
        for j in range(len(state_order)):
            val = prob_matrix.iloc[i, j]
            if val > 0.005:
                axes[1].text(j, i, f"{val:.3f}", ha="center", va="center",
                             fontsize=9, fontweight="bold" if val > 0.1 else "normal",
                             color="white" if val > 0.5 else "black")
    axes[1].set_xticks(range(len(state_order)))
    axes[1].set_xticklabels(state_order, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticks(range(len(state_order)))
    axes[1].set_yticklabels(state_order, fontsize=9)
    axes[1].set_title("Probabilidades de Transición")
    axes[1].set_xlabel("Estado destino (t+1)")
    axes[1].set_ylabel("Estado origen (t)")
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label="P(transición)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "21_matriz_transicion_markov.png")
    prob_matrix.to_csv(EDA_TABLES_DIR / "eda_markov_transition_probabilities.csv")


def generate_survival_curves(sample_df: pd.DataFrame) -> None:
    """Fig 22: Curvas de supervivencia Kaplan-Meier por cohorte de originación."""
    log_message("Fig 22: Curvas de supervivencia...")

    loan_col = resolve_col(LOAN_ID_COLUMN, sample_df.columns)
    age_col = resolve_col("loan_age", sample_df.columns)
    delin_col = resolve_col("current_loan_delinquency_status", sample_df.columns)
    orig_date_col = resolve_col("origination_date", sample_df.columns)

    if not all([loan_col, age_col, delin_col]):
        log_message("  Columnas insuficientes para supervivencia", level="warn")
        return

    df = sample_df[[loan_col, age_col, delin_col]].copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df[delin_col] = pd.to_numeric(df[delin_col], errors="coerce")
    df = df.dropna()

    if "_trimestre" in sample_df.columns:
        df["_trimestre"] = sample_df["_trimestre"]

    # Definir "evento": delinquency >= 3 (90+ días)
    loan_summary = df.groupby(loan_col).agg(
        max_age=(age_col, "max"),
        ever_delinquent=(delin_col, lambda x: (x >= 3).any()),
        time_to_event=(delin_col, lambda x: x[x >= 3].index.min()
                       if (x >= 3).any() else None),
    ).reset_index()

    # Agrupar por cohortes de edad (0-24, 25-60, 61-120, 121+)
    loan_summary["cohorte_edad"] = pd.cut(
        loan_summary["max_age"],
        bins=[0, 24, 60, 120, 360, 9999],
        labels=["0-24m", "25-60m", "61-120m", "121-360m", "360+m"],
        right=True,
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, 5))

    for i, (cohort, group) in enumerate(loan_summary.groupby("cohorte_edad", observed=True)):
        if len(group) < 20:
            continue
        ages = np.sort(group["max_age"].values)
        n = len(ages)
        # Supervivencia: fracción que NO tiene evento hasta edad t
        survival = 1 - np.arange(1, n + 1) / n
        ax.step(ages, survival, where="post", color=colors[i],
                linewidth=2, label=f"{cohort} (n={n:,})", alpha=0.8)

    ax.set_xlabel("Edad del Préstamo (meses)", fontsize=12)
    ax.set_ylabel("S(t) — Probabilidad de Supervivencia", fontsize=12)
    ax.set_title("Curvas de Supervivencia por Cohorte de Edad",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Cohorte", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "22_curvas_supervivencia_cohorte.png")


def generate_autocorrelation_delinquency(profiles: Dict[str, Dict]) -> None:
    """Fig 23: Autocorrelación de la tasa media de delinquency por trimestre."""
    log_message("Fig 23: Autocorrelación de delinquency...")

    delin_means = []
    for t in sorted(profiles.keys()):
        col = get_profile_col(profiles[t].get("columns", {}),
                              "current_loan_delinquency_status")
        if col:
            val = col.get("mean")
            if val is not None:
                delin_means.append(float(val))
            else:
                delin_means.append(np.nan)
        else:
            delin_means.append(np.nan)

    series = pd.Series(delin_means).dropna()
    if len(series) < 10:
        log_message("  Datos insuficientes para autocorrelación", level="warn")
        return

    # ACF manual
    max_lag = min(30, len(series) // 3)
    acf_values = []
    mean_s = series.mean()
    var_s = series.var()

    for lag in range(max_lag + 1):
        if var_s > 0:
            cov = np.mean((series.values[lag:] - mean_s) *
                          (series.values[:len(series) - lag] - mean_s))
            acf_values.append(cov / var_s)
        else:
            acf_values.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Análisis de Autocorrelación — Tasa Media de Delinquency",
                 fontsize=14, fontweight="bold")

    # ACF
    axes[0].bar(range(len(acf_values)), acf_values, color=PALETTE_CLUSTERS[0], alpha=0.7)
    ci = 1.96 / np.sqrt(len(series))
    axes[0].axhline(ci, color="red", ls="--", alpha=0.5)
    axes[0].axhline(-ci, color="red", ls="--", alpha=0.5)
    axes[0].set_xlabel("Lag (trimestres)")
    axes[0].set_ylabel("ACF")
    axes[0].set_title("Función de Autocorrelación (ACF)")
    axes[0].grid(True, alpha=0.3)

    # Serie temporal
    axes[1].plot(series.values, color=PALETTE_CLUSTERS[3], linewidth=1.5, marker=".", markersize=3)
    axes[1].set_xlabel("Trimestre (índice)")
    axes[1].set_ylabel("Delinquency media")
    axes[1].set_title("Serie temporal de delinquency media")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "23_autocorrelacion_delinquency.png")


def generate_cohort_age_heatmap(profiles: Dict[str, Dict]) -> None:
    """Fig 24: Heatmap de delinquency media por cohorte de originación × trimestre."""
    log_message("Fig 24: Heatmap cohorte × edad...")

    # Construir matriz: trimestre (eje Y) × variable (aproximación desde perfiles)
    # Usamos loan_age y delinquency por trimestre como proxy
    trimestres = sorted(profiles.keys())
    metrics = []

    for t in trimestres:
        cols = profiles[t].get("columns", {})
        age_data = get_profile_col(cols, "loan_age")
        delin_data = get_profile_col(cols, "current_loan_delinquency_status")

        mean_age = age_data.get("mean") if age_data else None
        mean_delin = delin_data.get("mean") if delin_data else None
        metrics.append({
            "trimestre": t,
            "mean_age": mean_age,
            "mean_delin": mean_delin,
        })

    df = pd.DataFrame(metrics)
    df["mean_age"] = pd.to_numeric(df["mean_age"], errors="coerce")
    df["mean_delin"] = pd.to_numeric(df["mean_delin"], errors="coerce")
    df = df.dropna()

    if len(df) < 5:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Evolución Temporal: Edad Media y Delinquency Media por Trimestre",
                 fontsize=14, fontweight="bold")

    # Edad media
    axes[0].fill_between(range(len(df)), df["mean_age"], alpha=0.3, color=PALETTE_CLUSTERS[0])
    axes[0].plot(range(len(df)), df["mean_age"], color=PALETTE_CLUSTERS[0], linewidth=2)
    axes[0].set_xlabel("Trimestre (índice)")
    axes[0].set_ylabel("Edad media (meses)")
    axes[0].set_title("Edad Media del Portfolio")
    axes[0].grid(True, alpha=0.3)

    # Delinquency media
    axes[1].fill_between(range(len(df)), df["mean_delin"], alpha=0.3, color="#d62728")
    axes[1].plot(range(len(df)), df["mean_delin"], color="#d62728", linewidth=2)
    axes[1].set_xlabel("Trimestre (índice)")
    axes[1].set_ylabel("Delinquency media")
    axes[1].set_title("Delinquency Media del Portfolio")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "24_heatmap_performance_cohorte_edad.png")


def generate_stacked_payment_evolution(profiles: Dict[str, Dict]) -> None:
    """Fig 25: Evolución apilada de estados de pago por trimestre."""
    log_message("Fig 25: Evolución apilada de estados de pago...")

    trimestres = sorted(profiles.keys())
    state_data = {s: [] for s in ["Corriente", "30d", "60d", "90d+", "Grave"]}

    for t in trimestres:
        cols = profiles[t].get("columns", {})
        delin = get_profile_col(cols, "current_loan_delinquency_status")
        total_rows = profiles[t].get("total_rows", 1)

        if not delin or "top_values" not in delin:
            for s in state_data:
                state_data[s].append(0)
            continue

        # Acumular por estado
        counts = {str(v.get("value", "")): v.get("count", 0)
                  for v in delin.get("top_values", [])}

        corriente = counts.get("0", 0) + counts.get("0.0", 0)
        d30 = counts.get("1", 0) + counts.get("1.0", 0)
        d60 = counts.get("2", 0) + counts.get("2.0", 0)
        d90 = sum(v for k, v in counts.items()
                  if k not in ("0", "0.0", "1", "1.0", "2", "2.0", "")
                  and float(k) <= 5 if _is_numeric(k))
        grave = sum(v for k, v in counts.items()
                    if _is_numeric(k) and float(k) > 5)

        state_data["Corriente"].append(corriente / total_rows * 100 if total_rows else 0)
        state_data["30d"].append(d30 / total_rows * 100 if total_rows else 0)
        state_data["60d"].append(d60 / total_rows * 100 if total_rows else 0)
        state_data["90d+"].append(d90 / total_rows * 100 if total_rows else 0)
        state_data["Grave"].append(grave / total_rows * 100 if total_rows else 0)

    fig, ax = plt.subplots(figsize=(18, 8))
    x = range(len(trimestres))
    colors_stack = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#1f1f1f"]

    bottom = np.zeros(len(trimestres))
    for (state_name, values), color in zip(state_data.items(), colors_stack):
        vals = np.array(values, dtype=float)
        ax.fill_between(x, bottom, bottom + vals, label=state_name,
                        alpha=0.8, color=color)
        bottom += vals

    ax.set_xlabel("Trimestre (índice)")
    ax.set_ylabel("% del Portfolio")
    ax.set_title("Evolución de Estados de Pago (% acumulado por trimestre)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", title="Estado")
    ax.set_xlim(0, len(trimestres) - 1)
    ax.grid(axis="y", alpha=0.3)

    # Etiquetas cada 10 trimestres
    tick_positions = list(range(0, len(trimestres), max(1, len(trimestres) // 15)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([trimestres[i] for i in tick_positions], rotation=45, fontsize=7)

    save_figure(fig, EDA_FIGURES_DIR / "25_evolucion_estados_pago_apilado.png")


def _is_numeric(s: str) -> bool:
    """Verifica si un string es numérico."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ============================================================================
# BLOQUE C — GRÁFICOS DE SEPARABILIDAD / CLUSTERING (FIGURAS 26-31)
# ============================================================================

def generate_bivariate_risk_scatter(sample_df: pd.DataFrame) -> None:
    """Fig 26: Scatter plots bivariados de las principales variables de riesgo."""
    log_message("Fig 26: Scatter bivariado de riesgo...")

    pairs = [
        ("borrower_credit_score_at_origination", "original_loan_to_value_ratio_ltv"),
        ("borrower_credit_score_at_origination", "debt_to_income_dti"),
        ("original_interest_rate", "original_upb"),
        ("debt_to_income_dti", "original_loan_to_value_ratio_ltv"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Diagramas Bivariados de Variables de Riesgo",
                 fontsize=15, fontweight="bold")

    for idx, (xvar, yvar) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        xc = resolve_col(xvar, sample_df.columns)
        yc = resolve_col(yvar, sample_df.columns)

        if not xc or not yc:
            ax.text(0.5, 0.5, f"Datos no disponibles",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        sub = sample_df[[xc, yc]].dropna()
        if len(sub) > 30_000:
            sub = sub.sample(30_000, random_state=42)

        ax.scatter(sub[xc], sub[yc], alpha=0.08, s=3, color=PALETTE_CLUSTERS[0])
        ax.set_xlabel(xvar.replace("_", " ")[:35], fontsize=10)
        ax.set_ylabel(yvar.replace("_", " ")[:35], fontsize=10)
        ax.set_title(f"{xvar[:20]} vs {yvar[:20]}", fontsize=11, fontweight="bold")

        # Correlación Pearson
        r = sub[xc].corr(sub[yc])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, EDA_FIGURES_DIR / "26_scatter_riesgo_bivariado.png")


def generate_mahalanobis_outliers(sample_df: pd.DataFrame) -> None:
    """Fig 27: Detección de outliers via distancia de Mahalanobis."""
    log_message("Fig 27: Outliers Mahalanobis...")

    target = [
        "borrower_credit_score_at_origination", "original_loan_to_value_ratio_ltv",
        "debt_to_income_dti", "original_interest_rate", "original_upb",
    ]
    avail = [c for c in target if c in sample_df.columns]
    if len(avail) < 3:
        log_message("  Insuficientes columnas para Mahalanobis", level="warn")
        return

    sub = sample_df[avail].dropna()
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=42)
    if len(sub) < 100:
        return

    # Calcular Mahalanobis
    X = sub.values
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        cov_inv = np.linalg.inv(cov + np.eye(len(avail)) * 1e-8)
    except np.linalg.LinAlgError:
        log_message("  Matriz singular, no se puede calcular Mahalanobis", level="warn")
        return

    diff = X - mean
    mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Detección de Outliers — Distancia de Mahalanobis",
                 fontsize=14, fontweight="bold")

    # Histograma
    axes[0].hist(mahal, bins=80, color=PALETTE_CLUSTERS[0], alpha=0.7, edgecolor="none")
    threshold = np.percentile(mahal, 97.5)
    axes[0].axvline(threshold, color="red", ls="--",
                    label=f"P97.5 = {threshold:.1f}")
    axes[0].set_xlabel("Distancia de Mahalanobis")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de la Distancia de Mahalanobis")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # QQ-plot contra chi2
    if SCIPY_AVAILABLE:
        theoretical = scipy_stats.chi2.ppf(
            np.linspace(0.01, 0.99, len(mahal)),
            df=len(avail)
        )
        sorted_mahal = np.sort(mahal ** 2)[:len(theoretical)]
        axes[1].scatter(theoretical, sorted_mahal, alpha=0.3, s=5, color=PALETTE_CLUSTERS[3])
        max_val = max(theoretical.max(), sorted_mahal.max())
        axes[1].plot([0, max_val], [0, max_val], "r--", alpha=0.7, label="Referencia χ²")
        axes[1].set_xlabel("Cuantiles teóricos χ²")
        axes[1].set_ylabel("Distancia² observada")
        axes[1].set_title("QQ-plot: Mahalanobis² vs χ²")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "scipy no disponible\npara QQ-plot χ²",
                     ha="center", va="center", transform=axes[1].transAxes)

    n_outliers = (mahal > threshold).sum()
    pct = n_outliers / len(mahal) * 100
    fig.text(0.5, 0.01, f"Outliers (>{threshold:.1f}): {n_outliers:,} ({pct:.1f}%)",
             ha="center", fontsize=11, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "27_outliers_mahalanobis.png")


def generate_pca_2d(sample_df: pd.DataFrame) -> None:
    """Fig 28: PCA exploratorio 2D coloreado por nivel de delinquency."""
    if not SKLEARN_AVAILABLE:
        log_message("  sklearn no disponible, omitiendo fig 28", level="warn")
        return
    log_message("Fig 28: PCA exploratorio 2D...")

    target = KEY_NUMERIC_COLUMNS.copy()
    avail = [c for c in target if c in sample_df.columns]
    if len(avail) < 4:
        log_message("  Insuficientes columnas para PCA", level="warn")
        return

    delin_col = resolve_col("current_loan_delinquency_status", sample_df.columns)

    sub = sample_df[avail + ([delin_col] if delin_col else [])].dropna()
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=42)
    if len(sub) < 100:
        return

    X = sub[avail].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(14, 10))

    if delin_col and delin_col in sub.columns:
        delin = pd.to_numeric(sub[delin_col], errors="coerce").fillna(0)
        delin_cat = pd.cut(delin, bins=[-1, 0, 1, 3, 999],
                           labels=["Corriente", "30d", "60-90d", "Grave"])
        colors_map = {"Corriente": "#2ca02c", "30d": "#ff7f0e",
                      "60-90d": "#d62728", "Grave": "#1f1f1f"}
        for cat in ["Corriente", "30d", "60-90d", "Grave"]:
            mask = delin_cat == cat
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           alpha=0.15, s=5, color=colors_map[cat], label=cat)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.1, s=3,
                   color=PALETTE_CLUSTERS[0])

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varianza)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varianza)", fontsize=12)
    ax.set_title("PCA Exploratorio 2D — Portafolio Hipotecario",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Estado Delinquency", markerscale=5, fontsize=10)
    ax.grid(True, alpha=0.3)

    total_var = sum(pca.explained_variance_ratio_) * 100
    ax.text(0.02, 0.02, f"Varianza explicada (2 comp.): {total_var:.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    save_figure(fig, EDA_FIGURES_DIR / "28_pca_exploratorio_2d.png")


def generate_scree_plot(sample_df: pd.DataFrame) -> None:
    """Fig 29: Scree plot (gráfico de sedimentación) del PCA."""
    if not SKLEARN_AVAILABLE:
        log_message("  sklearn no disponible, omitiendo fig 29", level="warn")
        return
    log_message("Fig 29: Scree plot PCA...")

    target = KEY_NUMERIC_COLUMNS.copy()
    avail = [c for c in target if c in sample_df.columns]
    if len(avail) < 4:
        return

    sub = sample_df[avail].dropna()
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=42)
    if len(sub) < 100:
        return

    X = StandardScaler().fit_transform(sub.values)
    n_comp = min(len(avail), 15)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)

    var_explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_explained)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = range(1, n_comp + 1)

    # Barras: varianza individual
    ax1.bar(x, var_explained * 100, color=PALETTE_CLUSTERS[0], alpha=0.7,
            label="Individual", zorder=2)
    ax1.set_xlabel("Componente Principal", fontsize=12)
    ax1.set_ylabel("Varianza Explicada (%)", fontsize=12, color=PALETTE_CLUSTERS[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE_CLUSTERS[0])

    # Línea: varianza acumulada
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100, "o-", color="#d62728", linewidth=2,
             markersize=6, label="Acumulada", zorder=3)
    ax2.set_ylabel("Varianza Acumulada (%)", fontsize=12, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.axhline(90, color="gray", ls="--", alpha=0.5, label="90% umbral")

    # Encontrar codo (Kaiser: eigenvalue > 1 → varianza > 1/p)
    threshold = 1.0 / len(avail) * 100
    n_kaiser = sum(1 for v in var_explained if v * 100 > threshold)
    ax1.axhline(threshold, color="green", ls=":", alpha=0.6,
                label=f"Kaiser (1/p = {threshold:.1f}%)")

    ax1.set_title(f"Scree Plot — PCA ({n_comp} componentes, Kaiser sugiere {n_kaiser})",
                  fontsize=14, fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    save_figure(fig, EDA_FIGURES_DIR / "29_scree_plot_pca.png")

    # Guardar tabla
    pd.DataFrame({
        "componente": list(x),
        "varianza_explicada_pct": var_explained * 100,
        "varianza_acumulada_pct": cumulative * 100,
    }).to_csv(EDA_TABLES_DIR / "eda_pca_varianza.csv", index=False)


def generate_hopkins_statistic(sample_df: pd.DataFrame) -> None:
    """Fig 30: Hopkins statistic bootstrap para evaluar clusterability."""
    if not SKLEARN_AVAILABLE:
        log_message("  sklearn no disponible, omitiendo fig 30", level="warn")
        return
    log_message("Fig 30: Hopkins statistic...")

    target = KEY_NUMERIC_COLUMNS[:8]
    avail = [c for c in target if c in sample_df.columns]
    if len(avail) < 3:
        return

    sub = sample_df[avail].dropna()
    if len(sub) > 20_000:
        sub = sub.sample(20_000, random_state=42)
    if len(sub) < 200:
        return

    X = StandardScaler().fit_transform(sub.values)

    def hopkins_stat(X_data, m=None, seed=0):
        """Calcula estadístico de Hopkins."""
        n = len(X_data)
        if m is None:
            m = min(100, n // 10)
        rng = np.random.RandomState(seed)

        # Muestra aleatoria de puntos del dataset
        idx = rng.choice(n, m, replace=False)
        X_sample = X_data[idx]

        # Puntos uniformes en el espacio de datos
        mins = X_data.min(axis=0)
        maxs = X_data.max(axis=0)
        X_uniform = rng.uniform(mins, maxs, size=(m, X_data.shape[1]))

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_data)

        u_dist, _ = nn.kneighbors(X_uniform)
        w_dist, _ = nn.kneighbors(X_sample)

        u_sum = (u_dist ** 2).sum()
        w_sum = (w_dist ** 2).sum()

        return u_sum / (u_sum + w_sum) if (u_sum + w_sum) > 0 else 0.5

    # Bootstrap
    n_bootstrap = 50
    hopkins_values = [hopkins_stat(X, seed=i) for i in range(n_bootstrap)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(hopkins_values, bins=20, color=PALETTE_CLUSTERS[2], alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(hopkins_values), color="red", ls="--", linewidth=2,
               label=f"Media: {np.mean(hopkins_values):.3f}")
    ax.axvline(0.5, color="gray", ls=":", linewidth=2,
               label="Aleatorio uniforme (0.5)")
    ax.set_xlabel("Estadístico de Hopkins (H)", fontsize=12)
    ax.set_ylabel("Frecuencia (bootstrap)")
    ax.set_title("Test de Hopkins — Evaluación de Clusterability",
                 fontsize=14, fontweight="bold")

    h_mean = np.mean(hopkins_values)
    if h_mean > 0.7:
        interp = "Alta tendencia a clustering"
    elif h_mean > 0.5:
        interp = "Tendencia moderada a clustering"
    else:
        interp = "Distribución cercana a uniforme"
    ax.text(0.95, 0.95, f"H medio = {h_mean:.3f}\n{interp}",
            transform=ax.transAxes, fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, EDA_FIGURES_DIR / "30_hopkins_statistic_bootstrap.png")


def generate_gap_statistic(sample_df: pd.DataFrame) -> None:
    """Fig 31: Gap statistic preview para selección óptima de K."""
    if not SKLEARN_AVAILABLE:
        log_message("  sklearn no disponible, omitiendo fig 31", level="warn")
        return
    log_message("Fig 31: Gap statistic preview...")

    target = KEY_NUMERIC_COLUMNS[:8]
    avail = [c for c in target if c in sample_df.columns]
    if len(avail) < 3:
        return

    sub = sample_df[avail].dropna()
    if len(sub) > 15_000:
        sub = sub.sample(15_000, random_state=42)
    if len(sub) < 200:
        return

    X = StandardScaler().fit_transform(sub.values)
    k_range = range(2, 11)
    n_refs = 5

    # Inertia real
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        km.fit(X)
        inertias.append(np.log(km.inertia_))

    # Inertia de referencia (datos uniformes)
    ref_inertias = np.zeros((len(k_range), n_refs))
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for r in range(n_refs):
        X_ref = np.random.RandomState(r).uniform(mins, maxs, size=X.shape)
        for ki, k in enumerate(k_range):
            km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
            km.fit(X_ref)
            ref_inertias[ki, r] = np.log(km.inertia_)

    gap = ref_inertias.mean(axis=1) - np.array(inertias)
    gap_std = ref_inertias.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Gap Statistic — Selección Óptima de K",
                 fontsize=14, fontweight="bold")

    # Gap statistic
    axes[0].errorbar(list(k_range), gap, yerr=gap_std, fmt="o-",
                     color=PALETTE_CLUSTERS[3], linewidth=2, capsize=5)
    optimal_k = list(k_range)[np.argmax(gap)]
    axes[0].axvline(optimal_k, color="red", ls="--", alpha=0.7,
                    label=f"K óptimo = {optimal_k}")
    axes[0].set_xlabel("Número de Clusters (K)")
    axes[0].set_ylabel("Gap(K)")
    axes[0].set_title("Gap Statistic")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Método del codo (inertia)
    axes[1].plot(list(k_range), inertias, "o-", color=PALETTE_CLUSTERS[0], linewidth=2)
    axes[1].set_xlabel("Número de Clusters (K)")
    axes[1].set_ylabel("log(Inercia)")
    axes[1].set_title("Método del Codo (Log-Inercia)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "31_gap_statistic_preview.png")


# ============================================================================
# BLOQUE D — HETEROGENEIDAD POR SEGMENTOS (FIGURAS 32-35)
# ============================================================================

def generate_fico_ltv_by_purpose(sample_df: pd.DataFrame) -> None:
    """Fig 32: Distribución FICO y LTV por propósito del préstamo."""
    log_message("Fig 32: FICO/LTV por propósito...")

    fico_col = resolve_col("borrower_credit_score_at_origination", sample_df.columns)
    ltv_col = resolve_col("original_loan_to_value_ratio_ltv", sample_df.columns)
    purpose_col = resolve_col("loan_purpose", sample_df.columns)

    if not all([fico_col, ltv_col, purpose_col]):
        log_message("  Columnas insuficientes para fig 32", level="warn")
        return

    sub = sample_df[[fico_col, ltv_col, purpose_col]].dropna()
    sub[fico_col] = pd.to_numeric(sub[fico_col], errors="coerce")
    sub[ltv_col] = pd.to_numeric(sub[ltv_col], errors="coerce")
    sub = sub.dropna()

    purpose_labels = {"P": "Compra", "C": "Cash-Out Refi", "R": "Refinanciamiento",
                      "N": "No Cash-Out Refi", "U": "Desconocido"}
    sub["purpose_label"] = sub[purpose_col].map(
        lambda x: purpose_labels.get(str(x).strip(), str(x)))

    top_purposes = sub["purpose_label"].value_counts().nlargest(5).index.tolist()
    sub = sub[sub["purpose_label"].isin(top_purposes)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Distribuciones por Propósito del Préstamo",
                 fontsize=14, fontweight="bold")

    # FICO
    data_fico = [sub[sub["purpose_label"] == p][fico_col].values for p in top_purposes]
    bp1 = axes[0].boxplot(data_fico, labels=top_purposes, patch_artist=True, showfliers=False)
    for patch, color in zip(bp1["boxes"], plt.cm.Set2(np.linspace(0, 1, len(top_purposes)))):
        patch.set_facecolor(color)
    axes[0].set_ylabel("FICO Score")
    axes[0].set_title("Puntaje FICO por Propósito")
    axes[0].grid(axis="y", alpha=0.3)

    # LTV
    data_ltv = [sub[sub["purpose_label"] == p][ltv_col].values for p in top_purposes]
    bp2 = axes[1].boxplot(data_ltv, labels=top_purposes, patch_artist=True, showfliers=False)
    for patch, color in zip(bp2["boxes"], plt.cm.Set2(np.linspace(0, 1, len(top_purposes)))):
        patch.set_facecolor(color)
    axes[1].set_ylabel("LTV Original (%)")
    axes[1].set_title("LTV por Propósito")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, EDA_FIGURES_DIR / "32_fico_ltv_por_proposito.png")


def generate_geographic_risk(profiles: Dict[str, Dict]) -> None:
    """Fig 33: Distribución geográfica de riesgo por estado."""
    log_message("Fig 33: Distribución geográfica de riesgo...")

    state_counts: Dict[str, int] = {}
    for p in profiles.values():
        cols = p.get("columns", {})
        state_data = get_profile_col(cols, "property_state")
        if not state_data:
            continue
        for entry in state_data.get("top_values", []):
            st = str(entry.get("value", "")).strip()
            ct = entry.get("count", 0)
            if len(st) == 2 and st.isalpha():
                state_counts[st] = state_counts.get(st, 0) + ct

    if not state_counts:
        return

    df_states = pd.DataFrame([
        {"state": k, "count": v} for k, v in state_counts.items()
    ]).sort_values("count", ascending=True)

    total = df_states["count"].sum()
    df_states["pct"] = df_states["count"] / total * 100

    fig, ax = plt.subplots(figsize=(12, max(8, len(df_states) * 0.25)))
    colors = plt.cm.YlOrRd(df_states["pct"] / df_states["pct"].max())
    ax.barh(range(len(df_states)), df_states["pct"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(df_states)))
    ax.set_yticklabels(df_states["state"].values, fontsize=8)
    ax.set_xlabel("% del Portfolio")
    ax.set_title("Distribución Geográfica del Portfolio por Estado",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Anotar top 5
    top5 = df_states.nlargest(5, "pct")
    for _, row in top5.iterrows():
        idx = df_states.index.get_loc(row.name)
        ax.text(row["pct"] + 0.1, idx, f"{row['pct']:.1f}%", va="center", fontsize=8)

    save_figure(fig, EDA_FIGURES_DIR / "33_distribucion_geografica_riesgo.png")
    df_states.to_csv(EDA_TABLES_DIR / "eda_distribucion_geografica.csv", index=False)


def generate_fico_by_channel(profiles: Dict[str, Dict]) -> None:
    """Fig 34: Evolución del FICO medio por canal de originación."""
    log_message("Fig 34: FICO por canal...")

    # Extraer FICO medio y canal por trimestre (aproximación desde perfiles)
    trimestres = sorted(profiles.keys())
    fico_by_t = []

    for t in trimestres:
        cols = profiles[t].get("columns", {})
        fico = get_profile_col(cols, "borrower_credit_score_at_origination")
        channel = get_profile_col(cols, "channel")

        if fico and fico.get("mean") is not None:
            fico_by_t.append({
                "trimestre": t,
                "fico_mean": float(fico["mean"]),
                "fico_std": float(fico.get("std", 0)),
            })

    if len(fico_by_t) < 5:
        return

    df = pd.DataFrame(fico_by_t)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(range(len(df)), df["fico_mean"], color=PALETTE_CLUSTERS[0],
            linewidth=2, marker=".", markersize=4, label="FICO medio")
    ax.fill_between(range(len(df)),
                    df["fico_mean"] - df["fico_std"],
                    df["fico_mean"] + df["fico_std"],
                    alpha=0.2, color=PALETTE_CLUSTERS[0], label="±1 std")

    ax.set_xlabel("Trimestre (índice)")
    ax.set_ylabel("Puntaje FICO")
    ax.set_title("Evolución del Puntaje FICO Medio por Trimestre",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Etiquetas cada 10 trimestres
    ticks = list(range(0, len(df), max(1, len(df) // 15)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([df.iloc[i]["trimestre"] for i in ticks], rotation=45, fontsize=7)

    save_figure(fig, EDA_FIGURES_DIR / "34_evolucion_fico_por_canal.png")


def generate_cross_domain_correlation(sample_df: pd.DataFrame) -> None:
    """Fig 35: Correlación cruzada entre dominios funcionales."""
    log_message("Fig 35: Correlación cruzada entre dominios...")

    domain_cols = {
        "FICO": "borrower_credit_score_at_origination",
        "LTV": "original_loan_to_value_ratio_ltv",
        "DTI": "debt_to_income_dti",
        "Rate_orig": "original_interest_rate",
        "Rate_curr": "current_interest_rate",
        "UPB_orig": "original_upb",
        "UPB_curr": "current_actual_upb",
        "Age": "loan_age",
        "Term": "original_loan_term",
        "MI%": "mortgage_insurance_percentage",
        "Rem_months": "remaining_months_to_maturity",
        "N_borr": "number_of_borrowers",
    }

    avail = {short: col for short, col in domain_cols.items()
             if col in sample_df.columns}
    if len(avail) < 4:
        return

    sub = sample_df[list(avail.values())].dropna()
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=42)

    sub.columns = list(avail.keys())
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    im = ax.imshow(corr.where(~mask).values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=10)

    for i in range(len(corr)):
        for j in range(i):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.5 else "black")

    ax.set_title("Correlación Cruzada entre Variables Clave (Multi-Dominio)",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Correlación Pearson")
    save_figure(fig, EDA_FIGURES_DIR / "35_correlacion_cruzada_dominios.png")


# ============================================================================
# BLOQUE E — CALIDAD DE DATOS (FIGURAS 36-37)
# ============================================================================

def generate_sentinel_map(profiles: Dict[str, Dict]) -> None:
    """Fig 36: Mapa de valores centinela por trimestre."""
    log_message("Fig 36: Mapa de valores centinela...")

    sentinel_strs = set()
    for vals in SENTINEL_VALUES.values():
        for v in vals:
            sentinel_strs.add(str(v))

    trimestres = sorted(profiles.keys())
    target_cols = list(KEY_NUMERIC_COLUMNS) + list(KEY_CATEGORICAL_COLUMNS)

    # Para cada columna, verificar si los top_values incluyen sentinelas
    sentinel_matrix = []
    cols_with_sentinel = set()

    for t in trimestres:
        row = {}
        cols_data = profiles[t].get("columns", {})
        for col in target_cols:
            col_data = get_profile_col(cols_data, col)
            if not col_data:
                row[col] = 0
                continue
            top_vals = col_data.get("top_values", [])
            sentinel_count = 0
            total = profiles[t].get("total_rows", 1)
            for entry in top_vals:
                val = str(entry.get("value", "")).strip()
                if val in sentinel_strs or val in ("", "nan", "None", " "):
                    sentinel_count += entry.get("count", 0)
            pct = sentinel_count / total * 100 if total > 0 else 0
            row[col] = pct
            if pct > 0:
                cols_with_sentinel.add(col)
        sentinel_matrix.append(row)

    if not cols_with_sentinel:
        log_message("  No se detectaron valores centinela residuales")
        return

    df_sentinel = pd.DataFrame(sentinel_matrix, index=trimestres)
    df_sentinel = df_sentinel[sorted(cols_with_sentinel)]

    fig, ax = plt.subplots(figsize=(max(14, len(cols_with_sentinel) * 0.5),
                                     max(8, len(trimestres) * 0.12)))
    im = ax.imshow(df_sentinel.values, cmap="Reds", aspect="auto", vmin=0,
                   vmax=max(5, df_sentinel.values.max()))
    ax.set_xticks(range(len(df_sentinel.columns)))
    ax.set_xticklabels([c[:20] for c in df_sentinel.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(trimestres)))
    ax.set_yticklabels(trimestres, fontsize=5)
    ax.set_title("Mapa de Valores Centinela Residuales (% por trimestre)",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="% centinela")
    save_figure(fig, EDA_FIGURES_DIR / "36_valores_centinela_mapa.png")


def generate_temporal_drift_ks(sample_df: pd.DataFrame) -> None:
    """Fig 37: Test KS de drift temporal entre primeros y últimos trimestres."""
    if not SCIPY_AVAILABLE:
        log_message("  scipy no disponible, omitiendo fig 37", level="warn")
        return
    log_message("Fig 37: Drift temporal (KS test)...")

    if "_trimestre" not in sample_df.columns:
        log_message("  Columna _trimestre no disponible", level="warn")
        return

    trimestres = sorted(sample_df["_trimestre"].unique())
    if len(trimestres) < 4:
        return

    # Dividir en mitad temprana y tardía
    mid = len(trimestres) // 2
    early_t = trimestres[:mid]
    late_t = trimestres[mid:]

    early = sample_df[sample_df["_trimestre"].isin(early_t)]
    late = sample_df[sample_df["_trimestre"].isin(late_t)]

    target = KEY_NUMERIC_COLUMNS.copy()
    avail = [c for c in target if c in sample_df.columns]
    if not avail:
        return

    results = []
    for col in avail:
        e_data = pd.to_numeric(early[col], errors="coerce").dropna()
        l_data = pd.to_numeric(late[col], errors="coerce").dropna()
        if len(e_data) < 30 or len(l_data) < 30:
            continue
        stat, pval = scipy_stats.ks_2samp(e_data, l_data)
        results.append({
            "variable": col,
            "ks_statistic": stat,
            "p_value": pval,
            "significant": pval < 0.05,
        })

    if not results:
        return

    df_ks = pd.DataFrame(results).sort_values("ks_statistic", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df_ks) * 0.4)))
    colors = ["#d62728" if sig else "#2ca02c" for sig in df_ks["significant"]]
    ax.barh(range(len(df_ks)), df_ks["ks_statistic"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df_ks)))
    ax.set_yticklabels([v[:30] for v in df_ks["variable"]], fontsize=9)
    ax.set_xlabel("Estadístico KS")
    ax.set_title(f"Test KS de Drift Temporal\n(trimestres {early_t[0]}-{early_t[-1]} "
                 f"vs {late_t[0]}-{late_t[-1]})",
                 fontsize=14, fontweight="bold")
    ax.axvline(0.05, color="gray", ls=":", alpha=0.5)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Significativo (p < 0.05)"),
        Patch(facecolor="#2ca02c", label="No significativo"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    for i, (_, row) in enumerate(df_ks.iterrows()):
        ax.text(row["ks_statistic"] + 0.002, i,
                f"p={row['p_value']:.1e}", va="center", fontsize=7)

    save_figure(fig, EDA_FIGURES_DIR / "37_drift_temporal_ks_test.png")
    df_ks.to_csv(EDA_TABLES_DIR / "eda_drift_temporal_ks.csv", index=False)


# ============================================================================
# ORQUESTADOR DE EDA AVANZADO (BLOQUES A-E)
# ============================================================================

def run_advanced_eda(profiles: Dict[str, Dict]) -> None:
    """
    Ejecuta todos los gráficos avanzados (figuras 14-37).
    Carga una muestra de Parquets una sola vez y la reutiliza.
    """
    log_message("Cargando muestra de Parquets para EDA avanzado...")
    sample_df = load_parquet_sample(
        n_trimestres=8,
        rows_per_trimestre=100_000,
        columns=None,  # Todas las columnas
    )

    if sample_df.empty:
        log_message("No se pudo cargar muestra. Gráficos avanzados omitidos.", level="warn")
        return

    log_message(f"  Muestra cargada: {len(sample_df):,} filas × {sample_df.shape[1]} cols")

    # ── Bloque A: Pre-AFE ──
    log_message("── Bloque A: Gráficos Pre-AFE (figuras 14-19) ──")
    try:
        generate_correlation_by_domains(sample_df)
    except Exception as e:
        log_message(f"  Error fig 14: {e}", level="warn")
    try:
        generate_vif_heatmap(sample_df)
    except Exception as e:
        log_message(f"  Error fig 15: {e}", level="warn")
    try:
        generate_skewness_kurtosis_scatter(sample_df)
    except Exception as e:
        log_message(f"  Error fig 16: {e}", level="warn")
    try:
        generate_qq_grid(sample_df)
    except Exception as e:
        log_message(f"  Error fig 17: {e}", level="warn")
    try:
        generate_transformation_comparison(sample_df)
    except Exception as e:
        log_message(f"  Error fig 18: {e}", level="warn")
    try:
        generate_nullity_correlation(profiles)
    except Exception as e:
        log_message(f"  Error fig 19: {e}", level="warn")

    print_memory_status("Post-Bloque-A")

    # ── Bloque B: Temporal/Longitudinal ──
    log_message("── Bloque B: Gráficos Temporales (figuras 20-25) ──")
    try:
        generate_sequence_lengths(sample_df)
    except Exception as e:
        log_message(f"  Error fig 20: {e}", level="warn")
    try:
        generate_markov_transition(sample_df)
    except Exception as e:
        log_message(f"  Error fig 21: {e}", level="warn")
    try:
        generate_survival_curves(sample_df)
    except Exception as e:
        log_message(f"  Error fig 22: {e}", level="warn")
    try:
        generate_autocorrelation_delinquency(profiles)
    except Exception as e:
        log_message(f"  Error fig 23: {e}", level="warn")
    try:
        generate_cohort_age_heatmap(profiles)
    except Exception as e:
        log_message(f"  Error fig 24: {e}", level="warn")
    try:
        generate_stacked_payment_evolution(profiles)
    except Exception as e:
        log_message(f"  Error fig 25: {e}", level="warn")

    print_memory_status("Post-Bloque-B")

    # ── Bloque C: Separabilidad/Clustering ──
    log_message("── Bloque C: Gráficos de Separabilidad (figuras 26-31) ──")
    try:
        generate_bivariate_risk_scatter(sample_df)
    except Exception as e:
        log_message(f"  Error fig 26: {e}", level="warn")
    try:
        generate_mahalanobis_outliers(sample_df)
    except Exception as e:
        log_message(f"  Error fig 27: {e}", level="warn")
    try:
        generate_pca_2d(sample_df)
    except Exception as e:
        log_message(f"  Error fig 28: {e}", level="warn")
    try:
        generate_scree_plot(sample_df)
    except Exception as e:
        log_message(f"  Error fig 29: {e}", level="warn")
    try:
        generate_hopkins_statistic(sample_df)
    except Exception as e:
        log_message(f"  Error fig 30: {e}", level="warn")
    try:
        generate_gap_statistic(sample_df)
    except Exception as e:
        log_message(f"  Error fig 31: {e}", level="warn")

    print_memory_status("Post-Bloque-C")

    # ── Bloque D: Heterogeneidad ──
    log_message("── Bloque D: Heterogeneidad por Segmentos (figuras 32-35) ──")
    try:
        generate_fico_ltv_by_purpose(sample_df)
    except Exception as e:
        log_message(f"  Error fig 32: {e}", level="warn")
    try:
        generate_geographic_risk(profiles)
    except Exception as e:
        log_message(f"  Error fig 33: {e}", level="warn")
    try:
        generate_fico_by_channel(profiles)
    except Exception as e:
        log_message(f"  Error fig 34: {e}", level="warn")
    try:
        generate_cross_domain_correlation(sample_df)
    except Exception as e:
        log_message(f"  Error fig 35: {e}", level="warn")

    print_memory_status("Post-Bloque-D")

    # ── Bloque E: Calidad de Datos ──
    log_message("── Bloque E: Calidad de Datos (figuras 36-37) ──")
    try:
        generate_sentinel_map(profiles)
    except Exception as e:
        log_message(f"  Error fig 36: {e}", level="warn")
    try:
        generate_temporal_drift_ks(sample_df)
    except Exception as e:
        log_message(f"  Error fig 37: {e}", level="warn")

    # Liberar muestra
    del sample_df
    clear_memory()
    print_memory_status("Post-EDA-Avanzado")
    log_message("EDA avanzado completado (figuras 14-37)")


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

        # Normalizar nombres de columnas en perfiles (OLD → NEW)
        profiles = normalize_profile_keys(profiles)
        log_message(f"  {len(profiles)} perfiles cargados, validados y normalizados")

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

        # ── Fase 11: EDA Avanzado (Bloques A-E, figuras 14-37) ──
        print("\n" + "─" * 50)
        print("  FASE 11: EDA Avanzado (24 gráficos: Bloques A-E)")
        print("─" * 50)
        run_advanced_eda(profiles)

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
