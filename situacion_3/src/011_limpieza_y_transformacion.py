#!/usr/bin/env python3
"""
011_limpieza_y_transformacion.py — Limpieza y Preparación para AFE
====================================================================
Situación 3: Portafolio Hipotecario (Fannie Mae Performance Data)

Analiza la calidad de datos de los Parquets consolidados y produce:
  1. Diagnóstico exhaustivo de nulidad por columna y trimestre
  2. Decisiones automáticas de retención/eliminación de columnas
  3. Verificación de valores centinela residuales
  4. Análisis de tipos de datos y coherencia
  5. Configuración de limpieza (JSON) para scripts downstream

Nota: este script NO reescribe los Parquets. Produce un plan de limpieza
que es consumido por 020_analisis_latente.py y posteriores vía
`load_cleaning_config()`. Se puede ejecutar con datos parciales (86/101).

Ejecución:
  python src/011_limpieza_y_transformacion.py
  python src/011_limpieza_y_transformacion.py --threshold 90   # nulos > 90% → drop
  python src/011_limpieza_y_transformacion.py --validate-sample # valida con Parquets

Autor: Nicolás Zapata Obando
Fecha: Febrero 2026
"""

import sys
import gc
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_PROCESSED_PATH, PANEL_PATH, PERFILES_PATH,
    FIGURES_PATH, TABLES_PATH, FIGURES_SUBDIRS,
    PERFORMANCE_COLUMNS, COLUMNS_TO_DROP,
    KEY_NUMERIC_COLUMNS, KEY_CATEGORICAL_COLUMNS,
    SENTINEL_VALUES, EXPECTED_NUMERIC_COLUMNS, DATE_COLUMNS,
    LOAN_ID_COLUMN, DATE_COLUMN, TOTAL_CSV_FILES,
    CHECKPOINTS_PATH, PROCESSING_STATE_FILE,
    FIGURE_DPI, OLD_TO_NEW_COLUMN_MAP,
)
from utils.memory_utils import (
    print_system_summary, print_memory_status,
    ProcessingTimer, clear_memory, check_memory_threshold,
)
from utils.plotting_utils import save_figure, configure_plot_style

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONSTANTES
# ============================================================================

# Directorio de figuras de limpieza
LIMPIEZA_FIGURES_DIR = FIGURES_PATH / "01_limpieza"
LIMPIEZA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Archivo de configuración de limpieza (output principal)
CLEANING_CONFIG_PATH = DATA_PROCESSED_PATH / "cleaning_config.json"

# Mapeo invertido new→old para buscar en perfiles con nombres antiguos
NEW_TO_OLD_MAP = {v: k for k, v in OLD_TO_NEW_COLUMN_MAP.items()}

# Umbrales por defecto
DEFAULT_NULL_DROP_THRESHOLD = 95.0    # Columnas con >95% nulos → candidatas a drop
DEFAULT_NULL_WARN_THRESHOLD = 50.0    # Columnas con >50% nulos → advertencia
DEFAULT_LOW_VARIANCE_CV = 0.5         # CV < 0.5% → baja variabilidad
DEFAULT_LOW_ENTROPY = 0.05            # Entropía normalizada < 0.05 → casi constante

# Columnas ARM: estructuralmente nulas para préstamos FRM (mayoría del dataset)
ARM_COLUMNS = [
    "arm_initial_fixed_rate_period_lte_5_yr_indicator",
    "arm_product_type", "initial_fixed_rate_period",
    "interest_rate_adjustment_frequency", "next_interest_rate_adjustment_date",
    "next_payment_change_date", "arm_index", "arm_cap_structure",
    "initial_interest_rate_cap_up_percent", "periodic_interest_rate_cap_up_percent",
    "lifetime_interest_rate_cap_up_percent", "mortgage_margin",
    "arm_balloon_indicator", "arm_plan_number",
]

# Columnas de liquidación: nulas mientras el préstamo esté vigente
LIQUIDATION_COLUMNS = [
    "zero_balance_code", "zero_balance_effective_date",
    "upb_at_the_time_of_removal", "foreclosure_date", "disposition_date",
    "foreclosure_costs", "property_preservation_and_repair_costs",
    "asset_recovery_costs", "miscellaneous_holding_expenses_and_credits",
    "associated_taxes_for_holding_property", "net_sales_proceeds",
    "credit_enhancement_proceeds", "repurchase_make_whole_proceeds",
    "other_foreclosure_proceeds", "principal_forgiveness_amount",
    "foreclosure_principal_write_off_amount", "delinquent_accrued_interest",
]

# Columnas de precio de propiedad (solo para REO/liquidaciones)
PRICE_COLUMNS = [
    "original_list_start_date", "original_list_price",
    "current_list_start_date", "current_list_price",
]

# Columnas CRT/Deal: solo presentes en pools CAS
CRT_COLUMNS = [
    "upb_at_issuance", "deal_name",
    "borrower_credit_score_at_issuance", "co_borrower_credit_score_at_issuance",
]


# ============================================================================
# UTILIDADES
# ============================================================================

def log_msg(msg: str, level: str = "info") -> None:
    """Imprime mensaje con timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {"info": "INFO", "warn": "⚠️ WARN", "error": "❌ ERROR"}.get(level, "INFO")
    print(f"  {ts} | {prefix:10s} | {msg}", flush=True)


def resolve_column_name(col_new: str, available_cols: set) -> Optional[str]:
    """
    Resuelve un nombre de columna nuevo a lo que esté disponible en los datos.
    Intenta primero el nombre nuevo, luego el antiguo.
    """
    if col_new in available_cols:
        return col_new
    old_name = NEW_TO_OLD_MAP.get(col_new)
    if old_name and old_name in available_cols:
        return old_name
    return None


def load_profiles() -> Dict[str, Dict]:
    """Carga todos los perfiles JSON individuales."""
    profiles = {}
    for pf in sorted(PERFILES_PATH.glob("perfil_*.json")):
        trimestre = pf.stem.replace("perfil_", "")
        if trimestre == "global":
            continue
        try:
            profiles[trimestre] = json.load(open(pf, "r"))
        except Exception as e:
            log_msg(f"Error leyendo {pf.name}: {e}", level="warn")
    return dict(sorted(profiles.items()))


def get_completed_trimestres() -> List[str]:
    """Retorna trimestres completados según checkpoint."""
    if PROCESSING_STATE_FILE.exists():
        cp = json.load(open(PROCESSING_STATE_FILE, "r"))
        files = cp.get("files", {})
        return sorted([
            f.replace(".csv", "")
            for f, info in files.items()
            if info.get("status") == "COMPLETADO"
        ])
    return []


# ============================================================================
# FASE 1: DIAGNÓSTICO DE NULIDAD
# ============================================================================

def analyze_nullity(
    profiles: Dict[str, Dict],
    completed: List[str],
) -> pd.DataFrame:
    """
    Construye la matriz de nulidad: trimestre × columna con % nulos.
    Clasifica cada columna según su patrón de nulidad.

    Returns:
        DataFrame con una fila por columna y estadísticas de nulidad.
    """
    log_msg("Analizando patrones de nulidad...")

    # Recoger todas las columnas disponibles en los perfiles
    all_cols = set()
    for p in profiles.values():
        all_cols.update(p.get("columns", {}).keys())
    all_cols = sorted(all_cols - {"_empty"})

    n_trim = len(profiles)
    rows = []

    for col in all_cols:
        null_pcts = []
        n_valids = []
        n_totals = []
        col_types = []

        for t, p in profiles.items():
            c = p.get("columns", {}).get(col, {})
            if c:
                null_pcts.append(c.get("null_pct", 100.0))
                n_valids.append(c.get("n_valid", 0))
                n_totals.append(c.get("n_total", 0))
                col_types.append(c.get("column_type", "unknown"))

        if not null_pcts:
            continue

        mean_null = np.mean(null_pcts)
        max_null = np.max(null_pcts)
        min_null = np.min(null_pcts)
        std_null = np.std(null_pcts)
        total_valid = sum(n_valids)
        total_rows = sum(n_totals)
        dominant_type = max(set(col_types), key=col_types.count) if col_types else "unknown"

        # Clasificar el patrón de nulidad
        # Mapear al nombre nuevo si es posible
        col_new = OLD_TO_NEW_COLUMN_MAP.get(col, col)

        if mean_null > 99.0:
            pattern = "SIEMPRE_NULO"
        elif mean_null > DEFAULT_NULL_DROP_THRESHOLD:
            if col_new in ARM_COLUMNS or col in ARM_COLUMNS:
                pattern = "ESTRUCTURAL_ARM"
            elif col_new in LIQUIDATION_COLUMNS or col in LIQUIDATION_COLUMNS:
                pattern = "ESTRUCTURAL_LIQUIDACION"
            elif col_new in PRICE_COLUMNS or col in PRICE_COLUMNS:
                pattern = "ESTRUCTURAL_PRECIO"
            elif col_new in CRT_COLUMNS or col in CRT_COLUMNS:
                pattern = "ESTRUCTURAL_CRT"
            else:
                pattern = "ALTO_NULO"
        elif mean_null > DEFAULT_NULL_WARN_THRESHOLD:
            pattern = "MODERADO_NULO"
        elif mean_null > 5.0:
            pattern = "BAJO_NULO"
        elif mean_null > 0:
            pattern = "MINIMO_NULO"
        else:
            pattern = "COMPLETO"

        # Determinar si el % de nulidad varía mucho entre trimestres
        temporal_stability = "ESTABLE" if std_null < 5.0 else "VARIABLE"

        rows.append({
            "columna": col,
            "columna_nueva": col_new,
            "tipo": dominant_type,
            "null_pct_medio": round(mean_null, 2),
            "null_pct_max": round(max_null, 2),
            "null_pct_min": round(min_null, 2),
            "null_pct_std": round(std_null, 2),
            "total_validos": total_valid,
            "total_filas": total_rows,
            "pct_validos_global": round(total_valid / max(total_rows, 1) * 100, 2),
            "trimestres_con_datos": len(null_pcts),
            "patron_nulidad": pattern,
            "estabilidad_temporal": temporal_stability,
        })

    df = pd.DataFrame(rows).sort_values("null_pct_medio", ascending=False)
    df.to_csv(TABLES_PATH / "011_diagnostico_nulidad.csv", index=False)
    log_msg(f"  {len(df)} columnas analizadas")

    # Resumen por patrón
    summary = df["patron_nulidad"].value_counts()
    for pattern, count in summary.items():
        log_msg(f"    {pattern}: {count} columnas")

    return df


# ============================================================================
# FASE 2: DECISIONES DE LIMPIEZA
# ============================================================================

def generate_cleaning_decisions(
    df_null: pd.DataFrame,
    profiles: Dict[str, Dict],
    null_drop_threshold: float = DEFAULT_NULL_DROP_THRESHOLD,
) -> Dict[str, Any]:
    """
    Genera las decisiones de limpieza basadas en el diagnóstico de nulidad
    y la informatividad de cada columna.

    Consideración: los datos están incompletos (86/101 trimestres). Los umbrales
    se ajustan con un margen de seguridad: en vez de descartar a >95%, se exige
    que la nulidad sea >95% en TODOS los trimestres disponibles (null_pct_min).

    Returns:
        Dict con la configuración de limpieza completa.
    """
    log_msg("Generando decisiones de limpieza...")

    n_available = df_null["trimestres_con_datos"].max()
    n_total_expected = TOTAL_CSV_FILES

    decisions = {
        "metadata": {
            "fecha_generacion": datetime.now().isoformat(),
            "trimestres_disponibles": int(n_available),
            "trimestres_totales": n_total_expected,
            "completitud_datos": round(n_available / n_total_expected * 100, 1),
            "umbral_nulo_drop": null_drop_threshold,
            "nota": (
                f"Decisiones basadas en {n_available}/{n_total_expected} trimestres. "
                f"El umbral de eliminación ({null_drop_threshold}%) se aplica sobre "
                f"el null_pct_min para evitar falsos positivos por datos incompletos."
            ),
        },
        "columnas_drop": [],
        "columnas_estructurales": [],
        "columnas_advertencia": [],
        "columnas_retener": [],
        "columnas_transformar": [],
        "sentinel_check": {},
    }

    for _, row in df_null.iterrows():
        col = row["columna"]
        col_new = row["columna_nueva"]
        pattern = row["patron_nulidad"]
        mean_null = row["null_pct_medio"]
        min_null = row["null_pct_min"]

        entry = {
            "columna": col,
            "columna_nueva": col_new,
            "tipo": row["tipo"],
            "null_pct_medio": mean_null,
            "null_pct_min": min_null,
            "patron": pattern,
            "razon": "",
        }

        if pattern == "SIEMPRE_NULO":
            entry["accion"] = "DROP"
            entry["razon"] = f"Nulo >{99}% en todos los trimestres"
            decisions["columnas_drop"].append(entry)

        elif pattern in ("ESTRUCTURAL_ARM", "ESTRUCTURAL_LIQUIDACION",
                         "ESTRUCTURAL_PRECIO", "ESTRUCTURAL_CRT"):
            entry["accion"] = "RETENER_ESTRUCTURAL"
            entry["razon"] = f"Nulidad estructural de negocio ({pattern}): " + {
                "ESTRUCTURAL_ARM": "Solo aplica a préstamos ARM (variable rate), mayoría son FRM",
                "ESTRUCTURAL_LIQUIDACION": "Solo aplica a préstamos liquidados/foreclosed",
                "ESTRUCTURAL_PRECIO": "Solo aplica a propiedades en proceso de venta REO",
                "ESTRUCTURAL_CRT": "Solo aplica a pools CAS/CRT emitidos",
            }.get(pattern, "Nulidad estructural")
            decisions["columnas_estructurales"].append(entry)

        elif pattern == "ALTO_NULO":
            # Usar null_pct_min para evitar descartar columnas que podrían
            # tener datos en los 15 trimestres faltantes
            if min_null > null_drop_threshold:
                entry["accion"] = "DROP"
                entry["razon"] = f"Nulo >{null_drop_threshold}% incluso en trimestres con más datos"
                decisions["columnas_drop"].append(entry)
            else:
                entry["accion"] = "ADVERTENCIA"
                entry["razon"] = (
                    f"Media nulos {mean_null:.1f}% pero mínimo {min_null:.1f}%. "
                    f"Podría mejorar con los {TOTAL_CSV_FILES - int(row['trimestres_con_datos'])} "
                    f"trimestres faltantes."
                )
                decisions["columnas_advertencia"].append(entry)

        elif pattern == "MODERADO_NULO":
            entry["accion"] = "RETENER_CON_ADVERTENCIA"
            entry["razon"] = f"Nulidad moderada ({mean_null:.1f}%). Requiere imputación o exclusión parcial."
            decisions["columnas_advertencia"].append(entry)

        else:
            entry["accion"] = "RETENER"
            entry["razon"] = f"Nulidad aceptable ({mean_null:.1f}%)"
            decisions["columnas_retener"].append(entry)

    # Identificar columnas numéricas con baja variabilidad
    for p in profiles.values():
        for col, info in p.get("columns", {}).items():
            cv = info.get("cv")
            if cv is not None and abs(cv) < DEFAULT_LOW_VARIANCE_CV and info.get("column_type") == "numeric":
                col_new = OLD_TO_NEW_COLUMN_MAP.get(col, col)
                # Solo agregar si no está ya en drop
                if not any(d["columna"] == col for d in decisions["columnas_drop"]):
                    decisions["columnas_transformar"].append({
                        "columna": col,
                        "columna_nueva": col_new,
                        "accion": "REVISAR_BAJA_VARIABILIDAD",
                        "cv": cv,
                        "razon": f"CV={cv:.2f}% — muy baja variabilidad. Considerar eliminar.",
                    })
        break  # Solo necesitamos un trimestre para esto

    # Verificación de valores centinela
    for sentinel_col, sentinel_vals in SENTINEL_VALUES.items():
        resolved = sentinel_col
        # Buscar en perfiles con nombre antiguo si es necesario
        old_name = NEW_TO_OLD_MAP.get(sentinel_col, sentinel_col)
        found_residuals = False
        for p in profiles.values():
            col_info = p.get("columns", {}).get(old_name) or p.get("columns", {}).get(sentinel_col)
            if col_info:
                # Verificar si el max value coincide con el centinela
                max_val = col_info.get("max")
                if max_val is not None and max_val in sentinel_vals:
                    found_residuals = True
                    break
        decisions["sentinel_check"][sentinel_col] = {
            "valores_centinela": sentinel_vals,
            "residuales_detectados": found_residuals,
            "nota": "Posibles centinelas residuales en max value" if found_residuals else "OK",
        }

    # Resumen
    decisions["resumen"] = {
        "total_columnas": len(df_null),
        "drop": len(decisions["columnas_drop"]),
        "estructurales": len(decisions["columnas_estructurales"]),
        "advertencia": len(decisions["columnas_advertencia"]),
        "retener": len(decisions["columnas_retener"]),
        "baja_variabilidad": len(decisions["columnas_transformar"]),
        "centinelas_residuales": sum(
            1 for v in decisions["sentinel_check"].values()
            if v["residuales_detectados"]
        ),
    }

    log_msg(f"  Decisiones: {decisions['resumen']}")

    return decisions


# ============================================================================
# FASE 3: VALIDACIÓN CON MUESTRA DE PARQUETS
# ============================================================================

def validate_with_parquet_sample(
    decisions: Dict,
    sample_trimestres: int = 3,
    sample_rows: int = 100_000,
) -> Dict:
    """
    Valida las decisiones de limpieza leyendo una muestra real de Parquets.
    Verifica tipos reales, centinelas residuales y coherencia.
    """
    log_msg("Validando decisiones con muestra de Parquets...")

    if not check_memory_threshold(min_available_gb=3.0):
        log_msg("RAM insuficiente para validación con Parquets", level="warn")
        return {"validado": False, "razon": "RAM insuficiente"}

    # Seleccionar trimestres distribuidos
    parquet_files = sorted(PANEL_PATH.glob("*.parquet"))
    if not parquet_files:
        log_msg("No hay archivos Parquet disponibles", level="warn")
        return {"validado": False, "razon": "Sin Parquets"}

    # Tomar inicio, medio y final
    step = max(1, len(parquet_files) // sample_trimestres)
    selected = parquet_files[::step][:sample_trimestres]

    validation = {
        "validado": True,
        "archivos_revisados": len(selected),
        "problemas": [],
        "tipos_reales": {},
        "centinelas_residuales": {},
    }

    for pf in selected:
        try:
            df = pd.read_parquet(pf, nrows=sample_rows) if hasattr(pd, '_') else pd.read_parquet(pf)
            if len(df) > sample_rows:
                df = df.head(sample_rows)

            # Registrar tipos reales
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                if col not in validation["tipos_reales"]:
                    validation["tipos_reales"][col] = dtype_str

            # Verificar centinelas residuales
            for sentinel_col, sentinel_vals in SENTINEL_VALUES.items():
                resolved = resolve_column_name(sentinel_col, set(df.columns))
                if resolved and resolved in df.columns:
                    for sv in sentinel_vals:
                        if isinstance(sv, (int, float)):
                            matches = (df[resolved] == sv).sum()
                        else:
                            matches = (df[resolved].astype(str) == str(sv)).sum()
                        if matches > 0:
                            key = f"{sentinel_col}={sv}"
                            prev = validation["centinelas_residuales"].get(key, 0)
                            validation["centinelas_residuales"][key] = prev + int(matches)

            del df
            gc.collect()

        except Exception as e:
            validation["problemas"].append(f"{pf.name}: {str(e)}")
            log_msg(f"  Error validando {pf.name}: {e}", level="warn")

    if validation["centinelas_residuales"]:
        log_msg(f"  ⚠️ Centinelas residuales detectados: {validation['centinelas_residuales']}", level="warn")
    else:
        log_msg("  ✅ Sin centinelas residuales en la muestra")

    log_msg(f"  Validación completada: {len(selected)} archivos revisados")
    return validation


# ============================================================================
# FASE 4: FIGURAS DE DIAGNÓSTICO
# ============================================================================

def plot_nullity_diagnostics(df_null: pd.DataFrame, decisions: Dict) -> None:
    """Genera figuras de diagnóstico de nulidad y limpieza."""
    log_msg("Generando figuras de diagnóstico...")
    configure_plot_style()

    # --- Figura 1: Distribución de nulidad por columna ---
    fig, ax = plt.subplots(figsize=(14, 8))
    df_sorted = df_null.sort_values("null_pct_medio", ascending=True)
    colors = []
    for _, row in df_sorted.iterrows():
        p = row["patron_nulidad"]
        if p == "COMPLETO":
            colors.append("#2ca02c")
        elif p in ("MINIMO_NULO", "BAJO_NULO"):
            colors.append("#17becf")
        elif p == "MODERADO_NULO":
            colors.append("#ff7f0e")
        elif p.startswith("ESTRUCTURAL"):
            colors.append("#9467bd")
        else:
            colors.append("#d62728")

    y_pos = range(len(df_sorted))
    ax.barh(y_pos, df_sorted["null_pct_medio"], color=colors, alpha=0.85, height=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_sorted["columna"], fontsize=5)
    ax.set_xlabel("% Nulos (media entre trimestres)")
    ax.set_title("Diagnóstico de Nulidad por Columna\n"
                 "(Verde=completo, Azul=bajo, Naranja=moderado, Púrpura=estructural, Rojo=alto)",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=DEFAULT_NULL_DROP_THRESHOLD, color="red", linestyle="--",
               alpha=0.7, label=f"Umbral drop ({DEFAULT_NULL_DROP_THRESHOLD}%)")
    ax.axvline(x=DEFAULT_NULL_WARN_THRESHOLD, color="orange", linestyle="--",
               alpha=0.7, label=f"Umbral advertencia ({DEFAULT_NULL_WARN_THRESHOLD}%)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, LIMPIEZA_FIGURES_DIR / "01_diagnostico_nulidad_columnas.png")

    # --- Figura 2: Clasificación de columnas (pie/donut) ---
    pattern_counts = df_null["patron_nulidad"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart de patrones
    colors_pie = {
        "COMPLETO": "#2ca02c", "MINIMO_NULO": "#17becf",
        "BAJO_NULO": "#1f77b4", "MODERADO_NULO": "#ff7f0e",
        "ALTO_NULO": "#d62728", "SIEMPRE_NULO": "#7f7f7f",
        "ESTRUCTURAL_ARM": "#9467bd", "ESTRUCTURAL_LIQUIDACION": "#8c564b",
        "ESTRUCTURAL_PRECIO": "#e377c2", "ESTRUCTURAL_CRT": "#bcbd22",
    }
    pie_colors = [colors_pie.get(p, "#999") for p in pattern_counts.index]
    wedges, texts, autotexts = ax1.pie(
        pattern_counts.values, labels=pattern_counts.index,
        autopct="%1.0f%%", colors=pie_colors, pctdistance=0.8,
        textprops={"fontsize": 8},
    )
    ax1.set_title("Clasificación de Columnas por Patrón de Nulidad", fontsize=11)

    # Resumen de decisiones
    resumen = decisions.get("resumen", {})
    ax2.axis("off")
    decision_text = [
        f"Total columnas analizadas:  {resumen.get('total_columnas', 0)}",
        f"",
        f"✅ Retener (nulidad aceptable):  {resumen.get('retener', 0)}",
        f"🟡 Estructurales (retener con nota):  {resumen.get('estructurales', 0)}",
        f"⚠️  Advertencia (revisar):  {resumen.get('advertencia', 0)}",
        f"❌ Eliminar (>umbral en todos trim):  {resumen.get('drop', 0)}",
        f"📉 Baja variabilidad:  {resumen.get('baja_variabilidad', 0)}",
        f"🔴 Centinelas residuales:  {resumen.get('centinelas_residuales', 0)}",
        f"",
        f"Datos disponibles:  {resumen.get('total_columnas', 0)} cols × {len(df_null)} registros",
        f"Completitud del dataset:  {decisions['metadata'].get('completitud_datos', 0):.0f}%",
    ]
    for i, line in enumerate(decision_text):
        ax2.text(0.05, 0.92 - i * 0.08, line, transform=ax2.transAxes,
                 fontsize=11, fontfamily="monospace",
                 verticalalignment="top")
    ax2.set_title("Resumen de Decisiones de Limpieza", fontsize=11)

    plt.tight_layout()
    save_figure(fig, LIMPIEZA_FIGURES_DIR / "02_clasificacion_columnas.png")

    # --- Figura 3: Nulidad por tipo de dato ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, tipo in enumerate(["numeric", "categorical"]):
        ax = axes[idx]
        df_tipo = df_null[df_null["tipo"] == tipo].sort_values("null_pct_medio", ascending=True)
        if df_tipo.empty:
            ax.text(0.5, 0.5, f"Sin columnas {tipo}", ha="center", va="center")
            continue
        colors_t = ["#2ca02c" if n < 5 else "#ff7f0e" if n < 50 else "#d62728"
                     for n in df_tipo["null_pct_medio"]]
        ax.barh(range(len(df_tipo)), df_tipo["null_pct_medio"],
                color=colors_t, alpha=0.85, height=0.8)
        ax.set_yticks(range(len(df_tipo)))
        ax.set_yticklabels(df_tipo["columna"], fontsize=6)
        ax.set_xlabel("% Nulos")
        ax.set_title(f"Nulidad — Columnas {tipo.title()}s", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, LIMPIEZA_FIGURES_DIR / "03_nulidad_por_tipo.png")

    # --- Figura 4: Estabilidad temporal de nulidad ---
    fig, ax = plt.subplots(figsize=(12, 6))
    df_variable = df_null[df_null["null_pct_std"] > 5].sort_values("null_pct_std", ascending=False).head(30)
    if not df_variable.empty:
        ax.barh(range(len(df_variable)), df_variable["null_pct_std"],
                color="#d62728", alpha=0.8)
        ax.set_yticks(range(len(df_variable)))
        ax.set_yticklabels(df_variable["columna"], fontsize=8)
        ax.set_xlabel("Desviación Estándar del % Nulos entre Trimestres")
        ax.set_title("Columnas con Nulidad Temporalmente Inestable\n"
                     "(nulidad varía mucho entre trimestres → posible data drift)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Todas las columnas tienen nulidad estable", ha="center", va="center")
    plt.tight_layout()
    save_figure(fig, LIMPIEZA_FIGURES_DIR / "04_nulidad_inestable.png")

    # --- Figura 5: Mapa calor decisiones agrupadas ---
    df_decisions = df_null.copy()
    df_decisions["grupo"] = df_decisions["patron_nulidad"].map({
        "COMPLETO": "1_Retener", "MINIMO_NULO": "1_Retener",
        "BAJO_NULO": "1_Retener", "MODERADO_NULO": "2_Advertencia",
        "ALTO_NULO": "3_Eliminar", "SIEMPRE_NULO": "3_Eliminar",
        "ESTRUCTURAL_ARM": "4_Estructural",
        "ESTRUCTURAL_LIQUIDACION": "4_Estructural",
        "ESTRUCTURAL_PRECIO": "4_Estructural",
        "ESTRUCTURAL_CRT": "4_Estructural",
    })
    group_summary = df_decisions.groupby("grupo").agg(
        n_columnas=("columna", "count"),
        null_medio=("null_pct_medio", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    ax.bar(range(len(group_summary)), group_summary["n_columnas"],
           color=colors_bar[:len(group_summary)], alpha=0.85)
    ax.set_xticks(range(len(group_summary)))
    ax.set_xticklabels(group_summary["grupo"], fontsize=10)
    ax.set_ylabel("Número de Columnas")
    ax.set_title("Decisiones de Limpieza por Grupo", fontsize=12, fontweight="bold")
    for i, row in group_summary.iterrows():
        ax.text(i, row["n_columnas"] + 0.5,
                f"{int(row['n_columnas'])} cols\n(nulos: {row['null_medio']:.0f}%)",
                ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, LIMPIEZA_FIGURES_DIR / "05_decisiones_por_grupo.png")

    log_msg(f"  5 figuras de diagnóstico generadas en {LIMPIEZA_FIGURES_DIR}")


# ============================================================================
# FASE 5: GUARDAR CONFIGURACIÓN
# ============================================================================

def save_cleaning_config(decisions: Dict, validation: Optional[Dict] = None) -> Path:
    """
    Guarda la configuración de limpieza como JSON.
    Este archivo es el INPUT de los scripts downstream (020+).
    """
    if validation:
        decisions["validacion_parquet"] = validation

    # Generar lista final de columnas a retener para AFE
    columns_for_afe = []
    all_drop = {d["columna"] for d in decisions["columnas_drop"]}

    for entry in decisions["columnas_retener"]:
        columns_for_afe.append(entry["columna_nueva"])
    for entry in decisions["columnas_advertencia"]:
        columns_for_afe.append(entry["columna_nueva"])
    # Estructurales se retienen pero marcadas
    for entry in decisions["columnas_estructurales"]:
        columns_for_afe.append(entry["columna_nueva"])

    decisions["columnas_para_afe"] = sorted(set(columns_for_afe))
    decisions["columnas_drop_lista"] = sorted({d["columna"] for d in decisions["columnas_drop"]})

    with open(CLEANING_CONFIG_PATH, "w") as f:
        json.dump(decisions, f, indent=2, default=str, ensure_ascii=False)

    log_msg(f"  Configuración guardada: {CLEANING_CONFIG_PATH}")
    log_msg(f"  Columnas para AFE: {len(decisions['columnas_para_afe'])}")
    log_msg(f"  Columnas a eliminar: {len(decisions['columnas_drop_lista'])}")

    return CLEANING_CONFIG_PATH


# ============================================================================
# UTILIDAD PÚBLICA: cargar config desde otros scripts
# ============================================================================

def load_cleaning_config() -> Dict:
    """
    Carga la configuración de limpieza.
    Usable desde 020_analisis_latente.py y otros scripts downstream.

    Ejemplo:
        from011_limpieza_y_transformacion import load_cleaning_config
        config = load_cleaning_config()
        cols_afe = config["columnas_para_afe"]
        cols_drop = config["columnas_drop_lista"]
    """
    if not CLEANING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {CLEANING_CONFIG_PATH}. "
            f"Ejecutar primero: python src/011_limpieza_y_transformacion.py"
        )
    return json.load(open(CLEANING_CONFIG_PATH, "r"))


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta el análisis de limpieza completo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="011 — Limpieza y Transformación del Dataset"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_NULL_DROP_THRESHOLD,
        help=f"Umbral de nulidad para eliminar columnas (default: {DEFAULT_NULL_DROP_THRESHOLD}%)",
    )
    parser.add_argument(
        "--validate-sample", action="store_true",
        help="Validar decisiones con muestra de Parquets",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  011 — LIMPIEZA Y TRANSFORMACIÓN")
    print("  Dataset: Fannie Mae CAS Loan Performance")
    print("=" * 70)
    print_system_summary()

    with ProcessingTimer("Limpieza y Transformación"):

        # ── Cargar datos ──
        completed = get_completed_trimestres()
        log_msg(f"Trimestres completados: {len(completed)}/{TOTAL_CSV_FILES}")

        if len(completed) == 0:
            log_msg("No hay trimestres completados. Ejecutar 001 primero.", level="error")
            sys.exit(1)

        if len(completed) < TOTAL_CSV_FILES:
            log_msg(
                f"⚠️ Solo {len(completed)}/{TOTAL_CSV_FILES} trimestres disponibles. "
                f"Las decisiones se marcan con margen de seguridad.",
                level="warn",
            )

        profiles = load_profiles()
        profiles = {k: v for k, v in profiles.items() if k in completed}
        log_msg(f"  {len(profiles)} perfiles cargados")

        # ── Fase 1: Diagnóstico de nulidad ──
        print("\n" + "─" * 50)
        print("  FASE 1: Diagnóstico de Nulidad")
        print("─" * 50)
        df_null = analyze_nullity(profiles, completed)
        print_memory_status("Post-diagnóstico")

        # ── Fase 2: Decisiones de limpieza ──
        print("\n" + "─" * 50)
        print("  FASE 2: Decisiones de Limpieza")
        print("─" * 50)
        decisions = generate_cleaning_decisions(
            df_null, profiles,
            null_drop_threshold=args.threshold,
        )

        # ── Fase 3: Validación (opcional) ──
        validation = None
        if args.validate_sample:
            print("\n" + "─" * 50)
            print("  FASE 3: Validación con Parquets")
            print("─" * 50)
            validation = validate_with_parquet_sample(decisions)
            print_memory_status("Post-validación")

        # ── Fase 4: Figuras de diagnóstico ──
        print("\n" + "─" * 50)
        print("  FASE 4: Figuras de Diagnóstico")
        print("─" * 50)
        plot_nullity_diagnostics(df_null, decisions)

        # ── Fase 5: Guardar configuración ──
        print("\n" + "─" * 50)
        print("  FASE 5: Guardar Configuración")
        print("─" * 50)
        config_path = save_cleaning_config(decisions, validation)

        # ── Resumen final ──
        print("\n" + "=" * 70)
        print("  RESUMEN DE LIMPIEZA")
        print("=" * 70)
        resumen = decisions["resumen"]
        log_msg(f"  Total columnas:         {resumen['total_columnas']}")
        log_msg(f"  ✅ Retener:              {resumen['retener']}")
        log_msg(f"  🟡 Estructurales:        {resumen['estructurales']}")
        log_msg(f"  ⚠️  Advertencia:          {resumen['advertencia']}")
        log_msg(f"  ❌ Eliminar:              {resumen['drop']}")
        log_msg(f"  📉 Baja variabilidad:     {resumen['baja_variabilidad']}")
        log_msg(f"  🔴 Centinelas residuales: {resumen['centinelas_residuales']}")
        log_msg(f"")
        log_msg(f"  Config guardada en: {config_path}")
        log_msg(f"  Figuras en: {LIMPIEZA_FIGURES_DIR}")

        n_figs = len(list(LIMPIEZA_FIGURES_DIR.glob("*.png")))
        log_msg(f"  📊 Figuras generadas: {n_figs}")

        if len(completed) < TOTAL_CSV_FILES:
            log_msg(
                f"\n  ⚠️ NOTA: Análisis parcial ({len(completed)}/{TOTAL_CSV_FILES}). "
                f"Re-ejecutar cuando todos los trimestres estén completos.",
                level="warn",
            )

    print_memory_status("Final")


if __name__ == "__main__":
    main()
