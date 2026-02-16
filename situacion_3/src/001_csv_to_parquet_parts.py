#!/usr/bin/env python3
"""
001_csv_to_parquet_parts.py — Fase 0: Conversión CSV a Parquet
====================================================================
Situación 3: Portafolio Hipotecario (Fannie Mae Performance Data)

Hace DOS cosas simultáneamente para no leer los ~820 GB dos veces:
  1. Convierte los 101 CSVs (dentro de Performance.zip) a Parquet (Zstd)
  2. Acumula estadísticas del EDA (media, varianza, histogramas, conteos)

Arquitectura:
  - Lee cada CSV en chunks directamente desde el ZIP (sin descomprimir a disco)
  - Limpia valores centinela (9999, 999, "XX", etc.)
  - Infiere tipos (numérico, categórico, fecha) en el primer chunk
  - Escribe cada chunk a Parquet con compresión Zstd nivel 3
  - Genera un perfil estadístico JSON por cada archivo
  - Soporta checkpoint/resume: si se detiene, retoma desde el último archivo completado
  - Distribución opcional con Ray: Master procesa archivos grandes, Worker los pequeños

Ejecución:
  python src/001_csv_to_parquet_parts.py              # Modo local (1 máquina)
  python src/001_csv_to_parquet_parts.py --distributed # Modo Ray (2 máquinas)
  python src/001_csv_to_parquet_parts.py --resume      # Retomar desde checkpoint
  python src/001_csv_to_parquet_parts.py --file 2003Q3.csv  # Procesar 1 solo archivo

Tiempo estimado: 9-15 horas en paralelo (Master + Worker), no supervisado.
"""

import argparse
import gc
import io
import json
import os
import sys
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configurar path del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ZIP_FILE_PATH,
    PERFORMANCE_COLUMNS,
    COLUMNS_TO_DROP,
    CSV_DELIMITER,
    CSV_ENCODING,
    PANEL_PATH,
    PERFILES_PATH,
    CHECKPOINTS_PATH,
    PROCESSING_STATE_FILE,
    TABLES_PATH,
    FIGURES_PATH,
    FIGURES_SUBDIRS,
    LOG_FILE,
    SENTINEL_VALUES,
    EXPECTED_NUMERIC_COLUMNS,
    DATE_COLUMNS,
    TOP_CATEGORIES_LIMIT,
    HISTOGRAM_N_BINS,
    CHUNK_THRESHOLD_GB,
    CHUNK_SIZE_ROWS,
    CHUNK_SIZE_ROWS_LARGE,
    LARGE_FILE_THRESHOLD_GB,
    MIN_AVAILABLE_RAM_GB,
    MEMORY_CRITICAL_THRESHOLD,
    NETWORK_RETRY_ATTEMPTS,
    NETWORK_RETRY_DELAY,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
    PARQUET_MAX_FILE_SIZE_GB,
    RAY_ENABLED,
    RAY_MASTER_IP,
    RAY_WORKER_IP,
    RAY_MASTER_CPUS,
    RAY_WORKER_CPUS,
    LOAN_ID_COLUMN,
    DATE_COLUMN,
)
from utils.memory_utils import (
    print_system_summary,
    print_memory_status,
    get_memory_usage,
    check_memory_threshold,
    wait_for_memory,
    clear_memory,
    aggressive_memory_cleanup,
    ProcessingTimer,
)
from utils.data_loader import ZipDataLoader
from utils.logging_utils import (
    setup_logger,
    get_logger,
    log_memory_snapshot,
    log_io_timing,
    log_exception,
    log_checkpoint as _log_checkpoint_util,
)

# tqdm para barras de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# ACUMULADOR DE ESTADÍSTICAS (Welford Online + Counter para categóricas)
# ============================================================================

class ColumnProfileAccumulator:
    """
    Acumulador de estadísticas en streaming para una columna.
    Permite calcular media, varianza, min, max, nulos, histograma, etc.
    sin tener todos los datos en RAM simultáneamente.

    Para numéricas: algoritmo de Welford (estable numéricamente).
    Para categóricas: Counter de frecuencias (top-N).
    """

    def __init__(self, col_name: str, col_type: str = "unknown"):
        """
        Inicializa el acumulador.

        Args:
            col_name: Nombre de la columna
            col_type: 'numeric', 'categorical', 'date', 'unknown'
        """
        self.col_name = col_name
        self.col_type = col_type

        # Conteos globales
        self.n_total: int = 0       # Total de filas vistas
        self.n_valid: int = 0       # Filas no nulas
        self.n_null: int = 0        # Filas nulas (NaN / vacío)

        # Estadísticas numéricas (Welford online)
        self.sum_x: float = 0.0
        self.sum_x2: float = 0.0   # Suma de cuadrados para varianza
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")

        # Histograma acumulado (bins fijos, se definen en primer chunk)
        self.hist_counts: Optional[np.ndarray] = None
        self.hist_bin_edges: Optional[np.ndarray] = None
        self.hist_initialized: bool = False

        # Categórica: Counter de frecuencias
        self.value_counts: Counter = Counter()

        # Percentiles aproximados: guardar muestra reservorio
        self.reservoir_sample: List[float] = []
        self.reservoir_max_size: int = 50_000   # Muestra para percentiles

        # Tipo de datos dominante
        self.dtype_observed: str = "unknown"

    def update_numeric(self, values: pd.Series) -> None:
        """
        Actualiza estadísticas numéricas con un batch de valores.

        Args:
            values: Serie con valores numéricos (puede contener NaN)
        """
        valid = values.dropna()
        n_new = len(valid)

        if n_new == 0:
            self.n_null += len(values)
            self.n_total += len(values)
            return

        self.n_total += len(values)
        self.n_null += len(values) - n_new
        self.n_valid += n_new

        arr = valid.values.astype(np.float64)
        self.sum_x += arr.sum()
        self.sum_x2 += (arr ** 2).sum()

        batch_min = arr.min()
        batch_max = arr.max()
        if batch_min < self.min_val:
            self.min_val = batch_min
        if batch_max > self.max_val:
            self.max_val = batch_max

        # Histograma acumulado
        if not self.hist_initialized:
            # Definir bins basado en primer batch con datos
            try:
                self.hist_bin_edges = np.linspace(
                    batch_min, batch_max, HISTOGRAM_N_BINS + 1
                )
                self.hist_counts = np.zeros(HISTOGRAM_N_BINS, dtype=np.int64)
                self.hist_initialized = True
            except (ValueError, TypeError):
                pass

        if self.hist_initialized and len(arr) > 0:
            try:
                # Expandir bins si hay valores fuera de rango
                if batch_min < self.hist_bin_edges[0] or batch_max > self.hist_bin_edges[-1]:
                    new_min = min(self.hist_bin_edges[0], batch_min)
                    new_max = max(self.hist_bin_edges[-1], batch_max)
                    new_edges = np.linspace(new_min, new_max, HISTOGRAM_N_BINS + 1)
                    # Re-bin los counts existentes (aproximado)
                    old_centers = (self.hist_bin_edges[:-1] + self.hist_bin_edges[1:]) / 2
                    new_counts = np.zeros(HISTOGRAM_N_BINS, dtype=np.int64)
                    old_indices = np.searchsorted(new_edges, old_centers, side='right') - 1
                    old_indices = np.clip(old_indices, 0, HISTOGRAM_N_BINS - 1)
                    for i, idx in enumerate(old_indices):
                        new_counts[idx] += self.hist_counts[i]
                    self.hist_bin_edges = new_edges
                    self.hist_counts = new_counts

                counts, _ = np.histogram(arr, bins=self.hist_bin_edges)
                self.hist_counts += counts
            except (ValueError, TypeError):
                pass

        # Muestra reservorio para percentiles
        remaining = self.reservoir_max_size - len(self.reservoir_sample)
        if remaining > 0:
            sample_size = min(remaining, n_new, 5000)
            if sample_size > 0:
                indices = np.random.choice(n_new, size=sample_size, replace=False)
                self.reservoir_sample.extend(arr[indices].tolist())
        elif n_new > 0:
            # Reemplazo aleatorio (reservoir sampling)
            for val in arr[np.random.choice(n_new, size=min(500, n_new), replace=False)]:
                j = np.random.randint(0, self.reservoir_max_size)
                self.reservoir_sample[j] = val

    def update_categorical(self, values: pd.Series) -> None:
        """
        Actualiza conteos de frecuencia para una columna categórica.

        Args:
            values: Serie con valores categóricos
        """
        n_row = len(values)
        self.n_total += n_row

        # Contar nulos
        null_mask = values.isna() | (values.astype(str).str.strip() == "")
        self.n_null += null_mask.sum()
        self.n_valid += n_row - null_mask.sum()

        # Contar frecuencias (excluyendo nulos)
        valid = values[~null_mask]
        if len(valid) > 0:
            self.value_counts.update(valid.astype(str).values)

    def finalize(self) -> Dict[str, Any]:
        """
        Calcula las estadísticas finales y retorna un diccionario serializable.

        Returns:
            Dict con todas las estadísticas de la columna
        """
        result: Dict[str, Any] = {
            "column_name": self.col_name,
            "column_type": self.col_type,
            "dtype_observed": self.dtype_observed,
            "n_total": int(self.n_total),
            "n_valid": int(self.n_valid),
            "n_null": int(self.n_null),
            "null_pct": round(self.n_null / max(self.n_total, 1) * 100, 2),
        }

        if self.col_type == "numeric" and self.n_valid > 0:
            mean = self.sum_x / self.n_valid
            variance = (self.sum_x2 / self.n_valid) - (mean ** 2)
            variance = max(variance, 0)  # Corrección numérica
            std = np.sqrt(variance)

            result.update({
                "mean": round(float(mean), 6),
                "std": round(float(std), 6),
                "variance": round(float(variance), 6),
                "min": round(float(self.min_val), 6) if self.min_val != float("inf") else None,
                "max": round(float(self.max_val), 6) if self.max_val != float("-inf") else None,
                "sum": round(float(self.sum_x), 2),
                "cv": round(float(std / abs(mean)) * 100, 2) if mean != 0 else None,
            })

            # Percentiles desde muestra reservorio
            if len(self.reservoir_sample) > 100:
                arr = np.array(self.reservoir_sample)
                result["percentiles"] = {
                    "p1": round(float(np.percentile(arr, 1)), 4),
                    "p5": round(float(np.percentile(arr, 5)), 4),
                    "p10": round(float(np.percentile(arr, 10)), 4),
                    "p25": round(float(np.percentile(arr, 25)), 4),
                    "p50": round(float(np.percentile(arr, 50)), 4),
                    "p75": round(float(np.percentile(arr, 75)), 4),
                    "p90": round(float(np.percentile(arr, 90)), 4),
                    "p95": round(float(np.percentile(arr, 95)), 4),
                    "p99": round(float(np.percentile(arr, 99)), 4),
                }

            # Histograma
            if self.hist_initialized and self.hist_counts is not None:
                result["histogram"] = {
                    "counts": self.hist_counts.tolist(),
                    "bin_edges": self.hist_bin_edges.tolist(),
                }

        elif self.col_type == "categorical":
            top_n = self.value_counts.most_common(TOP_CATEGORIES_LIMIT)
            n_unique = len(self.value_counts)

            result.update({
                "n_unique": n_unique,
                "top_values": [{"value": str(v), "count": int(c)} for v, c in top_n],
                "top_value": str(top_n[0][0]) if top_n else None,
                "top_value_freq": int(top_n[0][1]) if top_n else 0,
            })

            # Entropía de Shannon
            if self.n_valid > 0 and n_unique > 1:
                probs = np.array([c for _, c in self.value_counts.items()], dtype=np.float64)
                probs = probs / probs.sum()
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                max_entropy = np.log2(n_unique)
                result["entropy"] = round(float(entropy), 4)
                result["normalized_entropy"] = round(float(entropy / max_entropy), 4) if max_entropy > 0 else 0

        elif self.col_type == "date":
            # Para fechas, reportar rango
            if self.value_counts:
                top_n = self.value_counts.most_common(TOP_CATEGORIES_LIMIT)
                result.update({
                    "n_unique": len(self.value_counts),
                    "earliest": str(min(self.value_counts.keys())),
                    "latest": str(max(self.value_counts.keys())),
                    "top_values": [{"value": str(v), "count": int(c)} for v, c in top_n[:20]],
                })

        return result


# ============================================================================
# PERFIL POR ARCHIVO
# ============================================================================

class FileProfiler:
    """
    Perfila un archivo CSV completo procesándolo chunk por chunk.
    Mantiene un acumulador por columna y escribe Parquet simultáneamente.
    """

    def __init__(self, filename: str, file_size_gb: float):
        """
        Args:
            filename: Nombre del CSV dentro del ZIP (ej: '2003Q3.csv')
            file_size_gb: Tamaño descomprimido en GB
        """
        self.filename = filename
        self.file_size_gb = file_size_gb
        self.quarter = filename.replace(".csv", "")

        # Acumuladores por columna (se inicializan en primer chunk)
        self.accumulators: Dict[str, ColumnProfileAccumulator] = {}
        self.column_types: Dict[str, str] = {}  # Mapping nombre → tipo
        self.schema_inferred: bool = False

        # Contadores globales del archivo
        self.total_rows: int = 0
        self.total_chunks: int = 0
        self.start_time: float = 0
        self.parquet_bytes_written: int = 0

        # Parquet writer
        self._pq_writer: Optional[pq.ParquetWriter] = None
        self._pq_file_index: int = 0
        self._pq_rows_in_current_file: int = 0
        self._pq_schema: Optional[pa.Schema] = None

        # Calcular chunk size
        if file_size_gb > 20.0:
            self.chunk_size = CHUNK_SIZE_ROWS_LARGE  # 250K
        elif file_size_gb > CHUNK_THRESHOLD_GB:
            self.chunk_size = CHUNK_SIZE_ROWS  # 500K
        else:
            self.chunk_size = 0  # Carga directa

        # Ajustar para máquinas con menos RAM
        mem = get_memory_usage()
        if mem["available_gb"] < 8.0:
            self.chunk_size = max(self.chunk_size // 2, 100_000)

    def _get_parquet_path(self, part_index: int = 0) -> Path:
        """Genera la ruta del archivo Parquet de salida."""
        if part_index == 0:
            return PANEL_PATH / f"{self.quarter}.parquet"
        return PANEL_PATH / f"{self.quarter}_part{part_index:02d}.parquet"

    def _infer_column_types(self, df: pd.DataFrame) -> None:
        """
        Infiere el tipo de cada columna basado en el primer chunk.
        Solo se ejecuta una vez por archivo.

        Args:
            df: DataFrame del primer chunk
        """
        for col in df.columns:
            if col in COLUMNS_TO_DROP:
                continue

            if col in DATE_COLUMNS:
                self.column_types[col] = "date"
            elif col in EXPECTED_NUMERIC_COLUMNS:
                self.column_types[col] = "numeric"
            else:
                # Intentar detectar por contenido
                sample = df[col].dropna().head(1000)
                if len(sample) == 0:
                    self.column_types[col] = "unknown"
                    continue

                try:
                    pd.to_numeric(sample, errors="raise")
                    self.column_types[col] = "numeric"
                except (ValueError, TypeError):
                    self.column_types[col] = "categorical"

        # Inicializar acumuladores
        for col in df.columns:
            if col in COLUMNS_TO_DROP:
                continue
            col_type = self.column_types.get(col, "categorical")
            self.accumulators[col] = ColumnProfileAccumulator(col, col_type)
            self.accumulators[col].dtype_observed = str(df[col].dtype)

        self.schema_inferred = True

    def _clean_sentinel_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores centinela de Freddie Mac por NaN.

        Args:
            df: DataFrame a limpiar

        Returns:
            DataFrame con centinelas reemplazados por NaN
        """
        for col, sentinels in SENTINEL_VALUES.items():
            if col in df.columns:
                df[col] = df[col].replace(sentinels, np.nan)
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fuerza la conversión de tipos según la inferencia del primer chunk.
        Todas las columnas numéricas se guardan como float64 para evitar
        mismatch de schema Parquet entre chunks (int64 vs double cuando
        un chunk tiene NaN y otro no).

        Args:
            df: DataFrame a convertir

        Returns:
            DataFrame con tipos corregidos
        """
        for col in df.columns:
            if col in COLUMNS_TO_DROP:
                continue
            col_type = self.column_types.get(col, "categorical")

            if col_type == "numeric":
                # Siempre float64: evita int64 vs double entre chunks
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif col_type == "categorical":
                df[col] = df[col].astype(str).replace({"nan": np.nan, "": np.nan, " ": np.nan})

        return df

    def _accumulate_stats(self, df: pd.DataFrame) -> None:
        """
        Acumula estadísticas del chunk actual en los acumuladores.

        Args:
            df: DataFrame del chunk actual
        """
        for col, acc in self.accumulators.items():
            if col not in df.columns:
                continue

            if acc.col_type == "numeric":
                acc.update_numeric(df[col])
            elif acc.col_type in ("categorical", "date"):
                acc.update_categorical(df[col])
            else:
                # Intentar como numérico primero
                try:
                    numeric_vals = pd.to_numeric(df[col], errors="coerce")
                    if numeric_vals.notna().sum() > len(df[col]) * 0.5:
                        acc.col_type = "numeric"
                        acc.update_numeric(numeric_vals)
                    else:
                        acc.col_type = "categorical"
                        acc.update_categorical(df[col])
                except Exception:
                    acc.col_type = "categorical"
                    acc.update_categorical(df[col])

    def _write_chunk_to_parquet(self, df: pd.DataFrame) -> None:
        """
        Escribe un chunk al archivo Parquet con compresión Zstd.
        Maneja la división en múltiples archivos si supera el límite.
        Fuerza un schema unificado (int→double) para evitar mismatches.

        Args:
            df: DataFrame del chunk a escribir
        """
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Inicializar writer si es la primera vez
        if self._pq_writer is None:
            # Unificar schema: int→double para consistencia entre chunks
            unified_fields = []
            for field in table.schema:
                if pa.types.is_integer(field.type):
                    unified_fields.append(pa.field(field.name, pa.float64()))
                else:
                    unified_fields.append(field)
            self._pq_schema = pa.schema(unified_fields, metadata=table.schema.metadata)

            # Castear la primera tabla al schema unificado
            table = table.cast(self._pq_schema)

            pq_path = self._get_parquet_path(self._pq_file_index)
            self._pq_writer = pq.ParquetWriter(
                str(pq_path),
                schema=self._pq_schema,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
            )
        else:
            # Castear al schema establecido por el primer chunk
            try:
                table = table.cast(self._pq_schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                # Fallback: castear columna por columna
                arrays = []
                for i, field in enumerate(self._pq_schema):
                    col = table.column(field.name)
                    if col.type != field.type:
                        col = col.cast(field.type)
                    arrays.append(col)
                table = pa.table(
                    {field.name: arr for field, arr in zip(self._pq_schema, arrays)},
                    schema=self._pq_schema,
                )

        # Verificar si necesitamos dividir el archivo
        estimated_size = self._pq_rows_in_current_file * 600 / (1024**3)  # ~600 bytes/fila
        if estimated_size > PARQUET_MAX_FILE_SIZE_GB and self._pq_rows_in_current_file > 0:
            # Cerrar el writer actual y abrir uno nuevo
            self._pq_writer.close()
            self._pq_file_index += 1
            pq_path = self._get_parquet_path(self._pq_file_index)
            self._pq_writer = pq.ParquetWriter(
                str(pq_path),
                schema=self._pq_schema,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
            )
            self._pq_rows_in_current_file = 0

        # Escribir el chunk
        self._pq_writer.write_table(table)
        self._pq_rows_in_current_file += len(df)

        del table
        gc.collect()

    def process_chunk(self, chunk_df: pd.DataFrame, chunk_num: int) -> None:
        """
        Procesa un chunk individual: limpieza → tipos → estadísticas → Parquet.

        Args:
            chunk_df: DataFrame del chunk
            chunk_num: Número secuencial del chunk
        """
        # Eliminar columna vacía
        for col in COLUMNS_TO_DROP:
            if col in chunk_df.columns:
                chunk_df.drop(columns=[col], inplace=True)

        # Primera vez: inferir tipos de columna
        if not self.schema_inferred:
            self._infer_column_types(chunk_df)

        # Paso 1: Limpiar valores centinela
        chunk_df = self._clean_sentinel_values(chunk_df)

        # Paso 2: Forzar tipos
        chunk_df = self._coerce_types(chunk_df)

        # Paso 3: Acumular estadísticas
        self._accumulate_stats(chunk_df)

        # Paso 4: Escribir a Parquet
        self._write_chunk_to_parquet(chunk_df)

        # Contadores
        self.total_rows += len(chunk_df)
        self.total_chunks += 1

    def finalize(self) -> Dict[str, Any]:
        """
        Cierra el Parquet writer y genera el perfil JSON del archivo.

        Returns:
            Dict con el perfil completo del archivo
        """
        # Cerrar writer de Parquet
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None

        elapsed = time.time() - self.start_time

        # Verificar tamaño del Parquet generado
        parquet_size_gb = 0.0
        parquet_files = []
        for i in range(self._pq_file_index + 1):
            pq_path = self._get_parquet_path(i)
            if pq_path.exists():
                size_gb = pq_path.stat().st_size / (1024**3)
                parquet_size_gb += size_gb
                parquet_files.append({
                    "path": str(pq_path.name),
                    "size_gb": round(size_gb, 4),
                })

        # Generar perfil de las columnas
        column_profiles = {}
        high_null_columns = []

        for col_name, acc in self.accumulators.items():
            profile = acc.finalize()
            column_profiles[col_name] = profile

            # Identificar columnas con alta nulidad
            if profile["null_pct"] > 5.0:
                high_null_columns.append({
                    "column": col_name,
                    "null_pct": profile["null_pct"],
                })

        # Perfil general del archivo
        file_profile = {
            "filename": self.filename,
            "quarter": self.quarter,
            "file_size_gb": round(self.file_size_gb, 4),
            "parquet_size_gb": round(parquet_size_gb, 4),
            "compression_ratio": round(self.file_size_gb / max(parquet_size_gb, 0.001), 2),
            "parquet_files": parquet_files,
            "total_rows": self.total_rows,
            "total_chunks": self.total_chunks,
            "total_columns": len(self.accumulators),
            "processing_time_seconds": round(elapsed, 1),
            "processing_time_human": str(timedelta(seconds=int(elapsed))),
            "rows_per_second": round(self.total_rows / max(elapsed, 0.001)),
            "high_null_columns": sorted(high_null_columns, key=lambda x: -x["null_pct"]),
            "column_type_counts": {
                "numeric": sum(1 for a in self.accumulators.values() if a.col_type == "numeric"),
                "categorical": sum(1 for a in self.accumulators.values() if a.col_type == "categorical"),
                "date": sum(1 for a in self.accumulators.values() if a.col_type == "date"),
                "unknown": sum(1 for a in self.accumulators.values() if a.col_type == "unknown"),
            },
            "columns": column_profiles,
            "processed_at": datetime.now().isoformat(),
        }

        return file_profile


# ============================================================================
# GESTIÓN DE CHECKPOINT
# ============================================================================

class CheckpointManager:
    """Gestiona el estado de procesamiento de los 101 archivos."""

    STATES = ("PENDIENTE", "EN_PROCESO", "COMPLETADO", "ERROR")

    def __init__(self):
        self.state_file = PROCESSING_STATE_FILE
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Crea el directorio de checkpoints si no existe."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict[str, Any]:
        """Carga el estado desde disco."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"files": {}, "started_at": datetime.now().isoformat()}

    def save_state(self, state: Dict[str, Any]) -> None:
        """Guarda el estado a disco de forma atómica."""
        tmp_file = self.state_file.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(self.state_file)

    def get_file_status(self, filename: str) -> str:
        """Obtiene el estado de un archivo específico."""
        state = self.load_state()
        return state.get("files", {}).get(filename, {}).get("status", "PENDIENTE")

    def set_file_status(
        self,
        filename: str,
        status: str,
        extra: Optional[Dict] = None
    ) -> None:
        """
        Actualiza el estado de un archivo.

        Args:
            filename: Nombre del CSV
            status: PENDIENTE | EN_PROCESO | COMPLETADO | ERROR
            extra: Datos adicionales (filas, tiempo, error, etc.)
        """
        state = self.load_state()
        if "files" not in state:
            state["files"] = {}

        file_entry = state["files"].get(filename, {})
        file_entry["status"] = status
        file_entry["updated_at"] = datetime.now().isoformat()

        if extra:
            file_entry.update(extra)

        state["files"][filename] = file_entry
        self.save_state(state)

    def get_pending_files(self, all_files: List[str]) -> List[str]:
        """Retorna la lista de archivos pendientes de procesar.
        
        Incluye archivos:
        - Con estado PENDIENTE o ERROR
        - Con estado EN_PROCESO (sesiones anteriores interrumpidas)
        - Con estado COMPLETADO pero sin Parquets válidos en disco
        """
        import glob as _glob
        state = self.load_state()
        files_state = state.get("files", {})
        pending = []
        modified = False

        for f in all_files:
            entry = files_state.get(f, {})
            status = entry.get("status", "PENDIENTE")

            # EN_PROCESO de sesiones anteriores → resetear a PENDIENTE
            if status == "EN_PROCESO":
                log_message(f"   🔄 Reseteando {f} de EN_PROCESO → PENDIENTE")
                files_state.setdefault(f, {})["status"] = "PENDIENTE"
                files_state[f]["updated_at"] = datetime.now().isoformat()
                modified = True
                pending.append(f)
                continue

            # COMPLETADO → verificar que existen Parquets válidos
            if status == "COMPLETADO":
                quarter = f.replace(".csv", "")
                panel_dir = str(PANEL_PATH)
                parquets = _glob.glob(f"{panel_dir}/{quarter}.parquet") + \
                           _glob.glob(f"{panel_dir}/{quarter}_part*.parquet")
                if not parquets:
                    log_message(
                        f"   ⚠️ {f} marcado COMPLETADO pero sin Parquets → re-procesar"
                    )
                    files_state.setdefault(f, {})["status"] = "PENDIENTE"
                    files_state[f]["updated_at"] = datetime.now().isoformat()
                    modified = True
                    pending.append(f)
                    continue
                # Verificar integridad mínima de los Parquets
                try:
                    for p in parquets:
                        pq.ParquetFile(p)  # Lanza excepción si es corrupto
                except Exception:
                    log_message(
                        f"   ⚠️ {f} tiene Parquets corruptos → re-procesar"
                    )
                    # Borrar Parquets corruptos
                    for p in parquets:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    files_state.setdefault(f, {})["status"] = "PENDIENTE"
                    files_state[f]["updated_at"] = datetime.now().isoformat()
                    modified = True
                    pending.append(f)
                    continue
                # Parquets válidos → saltar
                continue

            if status in ("PENDIENTE", "ERROR"):
                pending.append(f)

        # Guardar si hubo cambios
        if modified:
            state["files"] = files_state
            self.save_state(state)

        return pending

    def get_completed_count(self) -> int:
        """Retorna cuántos archivos se han completado."""
        state = self.load_state()
        return sum(
            1 for f in state.get("files", {}).values()
            if f.get("status") == "COMPLETADO"
        )

    def print_summary(self) -> None:
        """Imprime un resumen del estado de procesamiento."""
        state = self.load_state()
        files = state.get("files", {})

        counts = Counter(f.get("status", "PENDIENTE") for f in files.values())
        total = sum(counts.values())

        print(f"\n  📊 Estado del procesamiento:")
        print(f"     COMPLETADO:  {counts.get('COMPLETADO', 0):>3} / {total}")
        print(f"     EN_PROCESO:  {counts.get('EN_PROCESO', 0):>3}")
        print(f"     PENDIENTE:   {counts.get('PENDIENTE', 0):>3}")
        print(f"     ERROR:       {counts.get('ERROR', 0):>3}")


# ============================================================================
# LOGGER DE PROCESAMIENTO
# ============================================================================

def log_message(msg: str, level: str = "info") -> None:
    """Escribe un mensaje al logger de Python (archivo rotativo + consola).

    Reemplaza la versión anterior que escribía manualmente a LOG_FILE.
    Ahora usa logging.RotatingFileHandler con timestamps y tracebacks.

    Args:
        msg: Mensaje a registrar
        level: Nivel del log ('debug', 'info', 'warning', 'error', 'critical')
    """
    logger = get_logger()
    log_func = getattr(logger, level, logger.info)
    log_func(msg)

    # Compatibilidad: también escribir al LOG_FILE legacy (opcional)
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except IOError:
        pass


# ============================================================================
# PROCESAMIENTO DE UN SOLO ARCHIVO
# ============================================================================

def process_single_file(
    filename: str,
    zip_path: Path = ZIP_FILE_PATH,
    checkpoint: Optional[CheckpointManager] = None,
) -> Dict[str, Any]:
    """
    Procesa un archivo CSV del ZIP: limpieza → perfilado → Parquet.

    Args:
        filename: Nombre del CSV dentro del ZIP (ej: '2003Q3.csv')
        zip_path: Ruta al archivo ZIP
        checkpoint: Gestor de checkpoints (None para no usar)

    Returns:
        Dict con el perfil estadístico del archivo
    """
    # Verificar checkpoint
    if checkpoint:
        current_status = checkpoint.get_file_status(filename)
        if current_status == "COMPLETADO":
            log_message(f"⏭️  {filename} ya está COMPLETADO — saltando")
            return {"filename": filename, "status": "SKIPPED"}
        checkpoint.set_file_status(filename, "EN_PROCESO")

    log_message(f"🚀 Iniciando procesamiento de {filename}")
    print_memory_status(f"Antes de {filename}")

    try:
        # Abrir el ZIP y obtener info del archivo
        with ZipDataLoader(zip_path) as loader:
            inventory = loader.get_file_inventory()
            file_info = next(
                (f for f in inventory if f["filename"] == filename), None
            )

            if file_info is None:
                raise FileNotFoundError(f"Archivo '{filename}' no encontrado en el ZIP")

            file_size_gb = file_info["uncompressed_gb"]

            # Crear el profiler
            profiler = FileProfiler(filename, file_size_gb)
            profiler.start_time = time.time()

            # Determinar si usar chunks
            if profiler.chunk_size > 0:
                # Lectura por chunks con callback
                log_message(
                    f"   📦 Modo chunked: {profiler.chunk_size:,} filas/chunk "
                    f"(archivo {file_size_gb:.2f} GB)"
                )

                chunk_num = 0
                z = loader._open_zip()

                for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
                    try:
                        with z.open(filename) as f:
                            reader = pd.read_csv(
                                f,
                                sep=CSV_DELIMITER,
                                header=None,
                                names=PERFORMANCE_COLUMNS,
                                encoding=CSV_ENCODING,
                                chunksize=profiler.chunk_size,
                                low_memory=True,
                                dtype=str,  # Leer todo como string inicialmente
                            )

                            for chunk in reader:
                                chunk_num += 1

                                # Verificar RAM
                                if not check_memory_threshold(MIN_AVAILABLE_RAM_GB):
                                    log_message(
                                        f"   ⏳ RAM baja antes del chunk {chunk_num}, "
                                        f"esperando..."
                                    )
                                    if not wait_for_memory(MIN_AVAILABLE_RAM_GB):
                                        log_message(
                                            f"   ⚠️ RAM insuficiente, reduciendo chunk size"
                                        )
                                        aggressive_memory_cleanup()

                                # Procesar chunk
                                profiler.process_chunk(chunk, chunk_num)

                                # Log periódico
                                if chunk_num % 10 == 0:
                                    elapsed = time.time() - profiler.start_time
                                    rate = profiler.total_rows / max(elapsed, 0.001)
                                    mem = get_memory_usage()
                                    log_message(
                                        f"   📊 {filename} chunk {chunk_num}: "
                                        f"{profiler.total_rows:,} filas, "
                                        f"{rate:,.0f} filas/s, "
                                        f"RAM {mem['percent_used']}%"
                                    )

                                # Liberar memoria del chunk
                                del chunk
                                gc.collect()

                        break  # Éxito

                    except (IOError, OSError) as e:
                        if attempt < NETWORK_RETRY_ATTEMPTS:
                            log_message(
                                f"   ⚠️ Error de red '{e}' — reintentando "
                                f"({attempt}/{NETWORK_RETRY_ATTEMPTS})"
                            )
                            time.sleep(NETWORK_RETRY_DELAY)
                            loader._zip_ref = None
                            z = loader._open_zip()
                        else:
                            raise

            else:
                # Archivo pequeño: carga directa
                log_message(f"   📂 Carga directa ({file_size_gb:.2f} GB)")
                df = loader.read_file(filename, dtype=str)
                profiler.process_chunk(df, 1)
                del df
                gc.collect()

        # Finalizar perfil (cierra ParquetWriter → flush a HDD)
        t_finalize = time.time()
        file_profile = profiler.finalize()
        log_io_timing(
            f"finalize+parquet_close({filename})",
            time.time() - t_finalize,
        )

        # Guardar perfil JSON
        profile_path = PERFILES_PATH / f"perfil_{profiler.quarter}.json"
        t_json = time.time()
        with open(profile_path, "w") as f:
            json.dump(file_profile, f, indent=2, ensure_ascii=False)
        profile_bytes = profile_path.stat().st_size
        log_io_timing(
            f"perfil_json({filename})",
            time.time() - t_json,
            profile_bytes,
        )

        # Actualizar checkpoint
        if checkpoint:
            t_ckpt = time.time()
            checkpoint.set_file_status(filename, "COMPLETADO", {
                "total_rows": profiler.total_rows,
                "total_chunks": profiler.total_chunks,
                "processing_time": round(time.time() - profiler.start_time, 1),
                "parquet_size_gb": file_profile.get("parquet_size_gb", 0),
            })
            log_io_timing(
                f"checkpoint({filename})",
                time.time() - t_ckpt,
            )

        log_message(
            f"✅ {filename}: {profiler.total_rows:,} filas → "
            f"{file_profile.get('parquet_size_gb', 0):.2f} GB Parquet "
            f"({file_profile.get('processing_time_human', '?')})"
        )

        # Limpiar
        del profiler
        gc.collect()
        print_memory_status(f"Después de {filename}")

        return file_profile

    except Exception as e:
        error_msg = f"❌ Error procesando {filename}: {e}"
        log_message(error_msg, level="error")
        log_exception(f"process_single_file({filename})")

        if checkpoint:
            checkpoint.set_file_status(filename, "ERROR", {
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

        gc.collect()
        return {"filename": filename, "status": "ERROR", "error": str(e)}


# ============================================================================
# ASIGNACIÓN ESTÁTICA DE ARCHIVOS
# ============================================================================

def assign_files_to_nodes(
    inventory: List[Dict],
    pending_files: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Asigna archivos a Master y Worker de forma estática.
    Master: archivos grandes (>10 GB) — tiene más RAM y CPU.
    Worker: archivos pequeños y medianos — más cantidad, menos peso.

    Args:
        inventory: Lista de dicts con info de cada archivo
        pending_files: Archivos pendientes de procesar

    Returns:
        (master_files, worker_files)
    """
    # Filtrar solo los pendientes
    pending_set = set(pending_files)
    pending_inventory = [f for f in inventory if f["filename"] in pending_set]

    # Ordenar por tamaño descendente
    pending_inventory.sort(key=lambda x: x["uncompressed_gb"], reverse=True)

    master_files = []
    worker_files = []
    master_gb = 0.0
    worker_gb = 0.0

    for f in pending_inventory:
        fname = f["filename"]
        size = f["uncompressed_gb"]

        # Archivos grandes (>10 GB) al Master (más capacidad)
        if size > LARGE_FILE_THRESHOLD_GB:
            master_files.append(fname)
            master_gb += size
        else:
            # Balancear entre Worker y Master
            # El Worker debería procesar más archivos pero más pequeños
            if worker_gb < master_gb * 0.7:  # Worker carga ~70% del volumen del Master
                worker_files.append(fname)
                worker_gb += size
            else:
                master_files.append(fname)
                master_gb += size

    log_message(
        f"📋 Asignación de archivos:\n"
        f"     Master: {len(master_files)} archivos ({master_gb:.1f} GB)\n"
        f"     Worker: {len(worker_files)} archivos ({worker_gb:.1f} GB)"
    )

    return master_files, worker_files


# ============================================================================
# EJECUCIÓN LOCAL (1 MÁQUINA)
# ============================================================================

def _process_file_for_pool(filename: str) -> Dict:
    """
    Wrapper para ProcessPoolExecutor - debe ser función de nivel módulo.
    No usa checkpoint para evitar conflictos de escritura concurrente.
    """
    return process_single_file(filename, checkpoint=None)


def run_local_parallel(
    files_to_process: Optional[List[str]] = None,
    resume: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict]:
    """
    Ejecuta el procesamiento en la máquina local con paralelismo (ProcessPoolExecutor).
    
    Usa todos los núcleos menos 1 (deja 1 para el SO y monitoreo).
    
    Args:
        files_to_process: Lista de archivos a procesar (None = todos)
        resume: Si True, salta archivos ya completados
        max_workers: Número de procesos paralelos (None = cpu_count - 1)
    
    Returns:
        Lista de perfiles generados
    """
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)  # Todos - 1
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    checkpoint = CheckpointManager()

    with ZipDataLoader() as loader:
        inventory = loader.get_file_inventory()
        all_files = [f["filename"] for f in inventory]

    # Determinar qué procesar
    if files_to_process:
        target_files = files_to_process
    elif resume:
        target_files = checkpoint.get_pending_files(all_files)
        if len(target_files) < len(all_files):
            completed = len(all_files) - len(target_files)
            log_message(f"🔄 Modo resume: {completed} ya completados, {len(target_files)} pendientes")
    else:
        target_files = all_files

    if not target_files:
        log_message("✅ Todos los archivos ya fueron procesados")
        checkpoint.print_summary()
        return []

    # Ordenar: procesar los más pequeños primero
    with ZipDataLoader() as loader:
        inv = loader.get_file_inventory()
    size_map = {f["filename"]: f["uncompressed_gb"] for f in inv}
    target_files.sort(key=lambda x: size_map.get(x, 0))

    log_message(
        f"📦 Procesando {len(target_files)} archivos en modo LOCAL PARALELO\n"
        f"   {max_workers} workers, procesamiento simultáneo"
    )

    profiles = []
    total_start = time.time()
    completed_count = checkpoint.get_completed_count()

    # Procesar en paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las tareas
        future_to_file = {
            executor.submit(_process_file_for_pool, f): f
            for f in target_files
        }
        
        # Recolectar resultados a medida que terminan
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                profiles.append(result)
                
                # Actualizar checkpoint desde proceso principal
                status = result.get("status", "OK")
                if status != "ERROR":
                    checkpoint.set_file_status(filename, "COMPLETADO", {
                        "total_rows": result.get("total_rows", 0),
                        "processing_time": result.get("processing_time", 0),
                        "parquet_size_gb": result.get("parquet_size_gb", 0),
                    })
                    log_message(
                        f"   ✅ {filename} → {result.get('total_rows', '?'):,} filas, "
                        f"{result.get('parquet_size_gb', 0):.3f} GB"
                    )
                else:
                    checkpoint.set_file_status(filename, "ERROR", {
                        "error": result.get("error", "Unknown")
                    })
                    log_message(f"   ❌ {filename}: {result.get('error', 'Unknown')}", level="error")
                
                completed_count += 1
                
                # Reporte de progreso cada 5 archivos
                if completed_count % 5 == 0:
                    elapsed = time.time() - total_start
                    pct = completed_count / len(all_files) * 100
                    log_message(
                        f"   📊 Progreso: {completed_count}/{len(all_files)} "
                        f"({pct:.1f}%) — {timedelta(seconds=int(elapsed))}"
                    )
                    
            except Exception as e:
                log_message(f"   ❌ {filename}: Error en ProcessPool: {e}", level="error")
                log_exception(f"_process_file_for_pool({filename})")
                checkpoint.set_file_status(filename, "ERROR", {"error": str(e)})

    total_elapsed = time.time() - total_start
    log_message(
        f"\n🏁 Procesamiento LOCAL PARALELO completado:\n"
        f"     {len(profiles)} archivos procesados\n"
        f"     Tiempo total: {timedelta(seconds=int(total_elapsed))}"
    )
    checkpoint.print_summary()

    return profiles


def run_local(
    files_to_process: Optional[List[str]] = None,
    resume: bool = True,
) -> List[Dict]:
    """
    Ejecuta el procesamiento en la máquina local (sin Ray).

    Args:
        files_to_process: Lista de archivos a procesar (None = todos)
        resume: Si True, salta archivos ya completados

    Returns:
        Lista de perfiles generados
    """
    checkpoint = CheckpointManager()

    with ZipDataLoader() as loader:
        inventory = loader.get_file_inventory()
        all_files = [f["filename"] for f in inventory]

    # Determinar qué procesar
    if files_to_process:
        target_files = files_to_process
    elif resume:
        target_files = checkpoint.get_pending_files(all_files)
        if len(target_files) < len(all_files):
            completed = len(all_files) - len(target_files)
            log_message(f"🔄 Modo resume: {completed} ya completados, {len(target_files)} pendientes")
    else:
        target_files = all_files

    if not target_files:
        log_message("✅ Todos los archivos ya fueron procesados")
        checkpoint.print_summary()
        return []

    # Ordenar: procesar los más pequeños primero (validar el pipeline rápido)
    with ZipDataLoader() as loader:
        inv = loader.get_file_inventory()
    size_map = {f["filename"]: f["uncompressed_gb"] for f in inv}
    target_files.sort(key=lambda x: size_map.get(x, 0))

    log_message(f"📦 Procesando {len(target_files)} archivos en modo LOCAL")

    profiles = []
    total_start = time.time()

    for i, filename in enumerate(target_files, 1):
        size = size_map.get(filename, 0)
        log_message(
            f"\n{'='*60}\n"
            f"  [{i}/{len(target_files)}] {filename} ({size:.2f} GB)\n"
            f"{'='*60}"
        )

        profile = process_single_file(filename, checkpoint=checkpoint)
        profiles.append(profile)

        # Progreso global
        elapsed = time.time() - total_start
        completed = checkpoint.get_completed_count()
        pct = completed / len(all_files) * 100
        log_message(
            f"  📈 Progreso global: {completed}/{len(all_files)} "
            f"({pct:.1f}%) — {timedelta(seconds=int(elapsed))}"
        )

        # Limpieza agresiva entre archivos
        aggressive_memory_cleanup()

    total_elapsed = time.time() - total_start
    log_message(
        f"\n🏁 Procesamiento LOCAL completado: "
        f"{len(profiles)} archivos en {timedelta(seconds=int(total_elapsed))}"
    )
    checkpoint.print_summary()

    return profiles


# ============================================================================
# EJECUCIÓN DISTRIBUIDA CON RAY
# ============================================================================

def run_distributed(resume: bool = True) -> List[Dict]:
    """
    Ejecuta el procesamiento distribuyendo archivos entre Master y Worker via Ray.

    Arquitectura:
      - Master: 8 hilos, usa 7 para procesamiento (todos - 1)
      - Worker: 4 hilos, usa 3 para procesamiento (todos - 1)
      - Ambos nodos procesan archivos en paralelo
      - SSH al Worker requiere llave: ~/.ssh/id_ed25519_proyecto

    Args:
        resume: Si True, salta archivos ya completados

    Returns:
        Lista de perfiles generados
    """
    try:
        import ray
    except ImportError:
        log_message("⚠️ Ray no está instalado. Ejecutando en modo local.")
        return run_local(resume=resume)

    checkpoint = CheckpointManager()

    # Obtener inventario
    with ZipDataLoader() as loader:
        inventory = loader.get_file_inventory()
        all_files = [f["filename"] for f in inventory]

    # Archivos pendientes
    if resume:
        pending = checkpoint.get_pending_files(all_files)
    else:
        pending = all_files

    if not pending:
        log_message("✅ Todos los archivos ya fueron procesados")
        checkpoint.print_summary()
        return []

    # Asignar archivos a nodos
    master_files, worker_files = assign_files_to_nodes(inventory, pending)

    # Inicializar Ray (Master usa todos-1 CPUs)
    if not ray.is_initialized():
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            resources = ray.cluster_resources()
            log_message(f"🟢 Ray conectado: {resources}")
            log_message(
                f"   Nota: Para conectar el Worker, ejecutar en 192.168.1.15:\n"
                f"   ray start --address='192.168.1.17:6379' --num-cpus=3"
            )

            # Verificar que el Worker está presente
            if "node:192.168.1.15" not in resources:
                log_message("⚠️ Worker (192.168.1.15) no detectado en el clúster Ray")
                log_message(
                    "   SSH al Worker: ssh -i ~/.ssh/id_ed25519_proyecto "
                    "mrsasayo_mesa@192.168.1.15"
                )
                log_message("   Los archivos del Worker se procesarán en local")
                master_files = master_files + worker_files
                worker_files = []
        except Exception as e:
            log_message(f"⚠️ No se pudo conectar al clúster Ray: {e}")
            log_message("   Ejecutando en modo local...")
            return run_local(files_to_process=pending, resume=resume)

    # Definir tareas remotas autocontenidas para ambos nodos
    # NOTA: las funciones son 100% autocontenidas — importan todo internamente
    #   porque el módulo '001_csv_to_parquet_parts' no es importable por nombre
    #   (comienza con dígito). Ray necesita recrear las funciones en cada nodo.
    
    # Estrategia CPU: todos - 1 núcleo por nodo
    #   Master: 8 hilos → 7 para procesamiento → num_cpus=1 → 7 tareas paralelas
    #   Worker: 4 hilos → 3 para procesamiento → num_cpus=1 → 3 tareas paralelas
    # Se reserva 1 hilo para el SO, Ray overhead y monitoreo.
    
    @ray.remote(
        num_cpus=1,
        max_retries=2,
        resources={"node:192.168.1.17": 0.001},
    )
    def process_file_on_master(filename: str, zip_path_str: str) -> dict:
        """Tarea Ray: procesar un archivo en el Master (192.168.1.17).
        
        num_cpus=1: permite 7 tareas en paralelo (8 CPUs - 1 reservado = 7)
        """
        import sys
        import importlib
        from pathlib import Path

        # Configurar sys.path
        src_dir = str(Path(zip_path_str).parent.parent / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Configurar logging
        from utils.logging_utils import setup_logger
        setup_logger("001_master_parallel")

        # Importar el módulo completo
        mod = importlib.import_module("001_csv_to_parquet_parts")

        # Procesar archivo SIN CHECKPOINT (evita conflictos de escritura paralela)
        # El checkpoint se actualizará desde el proceso principal
        return mod.process_single_file(
            filename,
            zip_path=Path(zip_path_str),
            checkpoint=None,
        )
    
    @ray.remote(
        num_cpus=1,
        max_retries=2,
        resources={"node:192.168.1.15": 0.001},
    )
    def process_file_on_worker(filename: str, zip_path_str: str) -> dict:
        """Tarea Ray: procesar un archivo en el Worker (192.168.1.15).
        
        num_cpus=1: permite 3 tareas en paralelo (4 CPUs - 1 reservado = 3)
        Conexión SSH al Worker requiere llave: ~/.ssh/id_ed25519_proyecto
        """
        import sys
        import importlib
        from pathlib import Path

        # Configurar sys.path para el Worker
        src_dir = str(Path(zip_path_str).parent.parent / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Configurar logging en el Worker
        from utils.logging_utils import setup_logger
        setup_logger("001_worker")

        # Importar el módulo completo (nombre empieza con dígito → importlib)
        mod = importlib.import_module("001_csv_to_parquet_parts")

        # Procesar archivo SIN CHECKPOINT (evita conflictos de escritura paralela)
        # El checkpoint se actualizará desde el proceso principal
        return mod.process_single_file(
            filename,
            zip_path=Path(zip_path_str),
            checkpoint=None,
        )

    log_message(
        f"🚀 Modo DISTRIBUIDO PARALELO:\n"
        f"     Master: {len(master_files)} archivos (7 en paralelo, 1 CPU/tarea)\n"
        f"     Worker: {len(worker_files)} archivos (3 en paralelo, 1 CPU/tarea)\n"
        f"     SSH Worker: llave ~/.ssh/id_ed25519_proyecto"
    )
    total_start = time.time()
    zip_path_str = str(ZIP_FILE_PATH)

    # Lanzar TODAS las tareas en Ray (ambos nodos)
    # Master: max 3 en paralelo (6 CPUs / 2 CPUs por tarea)
    # Worker: max 1 a la vez (3 CPUs / 3 CPUs por tarea)
    all_futures = []
    
    for f in master_files:
        future = process_file_on_master.remote(f, zip_path_str)
        all_futures.append((f, future, "master"))
    
    for f in worker_files:
        future = process_file_on_worker.remote(f, zip_path_str)
        all_futures.append((f, future, "worker"))
    
    log_message(
        f"   📤 {len(master_files)} tareas Master + "
        f"{len(worker_files)} tareas Worker = {len(all_futures)} totales"
    )

    # Esperar todas las tareas con monitoreo de progreso
    all_profiles = []
    completed_count = 0
    last_report = 0
    
    log_message("\n⏳ Procesando en paralelo (Master + Worker)...")
    
    while all_futures:
        # Esperar que se complete al menos 1 tarea (timeout 60s para reportar progreso)
        ready_refs = [f for _, f, _ in all_futures]
        ready, _ = ray.wait(ready_refs, num_returns=1, timeout=60)
        
        if not ready:
            # Reporte de progreso cada 60s
            log_message(
                f"   📊 Progreso: {completed_count}/{len(master_files) + len(worker_files)} "
                f"archivos completados ({completed_count*100/(len(master_files) + len(worker_files)):.1f}%)"
            )
            continue
        
        # Procesar tareas completadas
        for ready_ref in ready:
            # Encontrar qué tarea se completó
            for i, (filename, future, node) in enumerate(all_futures):
                if future == ready_ref:
                    try:
                        result = ray.get(future)
                        status = result.get("status", "OK")
                        rows = result.get("total_rows", "?")
                        pq_gb = result.get("parquet_size_gb", 0)
                        elapsed_file = result.get("processing_time_human", "?")
                        
                        log_message(
                            f"   ✅ [{node.upper()}] {filename} → {rows:,} filas, "
                            f"{pq_gb:.3f} GB Parquet ({elapsed_file})"
                        )
                        all_profiles.append(result)
                        
                        # Actualizar checkpoint (desde proceso principal, no hay conflictos)
                        if status != "ERROR":
                            checkpoint.set_file_status(filename, "COMPLETADO", {
                                "total_rows": rows,
                                "processing_time": result.get("processing_time", 0),
                                "parquet_size_gb": pq_gb,
                            })
                    except Exception as e:
                        log_message(f"❌ [{node.upper()}] {filename}: {e}", level="error")
                        log_exception(f"ray_get({filename})")
                        all_profiles.append({
                            "filename": filename,
                            "status": "ERROR",
                            "error": str(e),
                        })
                        # Marcar como error en checkpoint
                        checkpoint.set_file_status(filename, "ERROR", {"error": str(e)})
                    
                    # Remover de la lista
                    all_futures.pop(i)
                    completed_count += 1
                    
                    # Reporte cada 5 archivos
                    if completed_count - last_report >= 5:
                        last_report = completed_count
                        log_message(
                            f"   📊 Progreso: {completed_count}/{len(master_files) + len(worker_files)} "
                            f"completados ({completed_count*100/(len(master_files) + len(worker_files)):.1f}%)"
                        )
                    break
    total_elapsed = time.time() - total_start

    # Estadísticas finales
    ok_count = sum(1 for p in all_profiles if p.get("status") != "ERROR")
    err_count = sum(1 for p in all_profiles if p.get("status") == "ERROR")
    total_rows = sum(p.get("total_rows", 0) for p in all_profiles if p.get("status") != "ERROR")

    log_message(
        f"\n🏁 Procesamiento DISTRIBUIDO completado:\n"
        f"     Archivos: {ok_count} OK, {err_count} errores\n"
        f"     Filas totales: {total_rows:,}\n"
        f"     Tiempo total: {timedelta(seconds=int(total_elapsed))}"
    )
    checkpoint.print_summary()

    return all_profiles


# ============================================================================
# CONSOLIDACIÓN DEL PERFIL GLOBAL (Sub-Fase 0.4)
# ============================================================================

def consolidate_profiles() -> Dict[str, Any]:
    """
    Consolida los 101 perfiles individuales en un perfil global.
    Genera:
    - Tabla maestra de perfil de columnas  
    - Reporte de evolución temporal de variables clave
    - Mapa de nulidad (trimestre × columna)
    - Ranking de columnas por informatividad

    Returns:
        Dict con el perfil global consolidado
    """
    log_message("🔄 Consolidando perfiles estadísticos...")

    # Cargar todos los perfiles JSON
    profile_files = sorted(PERFILES_PATH.glob("perfil_*.json"))
    if not profile_files:
        log_message("⚠️ No se encontraron perfiles para consolidar")
        return {}

    profiles = []
    for pf in profile_files:
        with open(pf, "r") as f:
            profiles.append(json.load(f))

    log_message(f"   Cargados {len(profiles)} perfiles")

    # ====================================================================
    # 1. Tabla maestra de perfil de columnas (global)
    # ====================================================================
    column_master = {}
    all_quarters = [p["quarter"] for p in profiles]

    # Obtener nombres de columnas del primer perfil
    first_columns = list(profiles[0].get("columns", {}).keys())

    for col_name in first_columns:
        col_stats = {
            "column_name": col_name,
            "n_total_global": 0,
            "n_valid_global": 0,
            "n_null_global": 0,
            "col_type": "unknown",
            "present_in_quarters": 0,
        }

        # Acumular a través de todos los trimestres
        type_votes = Counter()
        sum_x_global = 0.0
        sum_x2_global = 0.0
        global_min = float("inf")
        global_max = float("-inf")
        global_counter = Counter()
        reservoir = []

        for prof in profiles:
            col_data = prof.get("columns", {}).get(col_name)
            if col_data is None:
                continue

            col_stats["present_in_quarters"] += 1
            col_stats["n_total_global"] += col_data.get("n_total", 0)
            col_stats["n_valid_global"] += col_data.get("n_valid", 0)
            col_stats["n_null_global"] += col_data.get("n_null", 0)

            ctype = col_data.get("column_type", "unknown")
            type_votes[ctype] += 1

            if ctype == "numeric":
                if "mean" in col_data and "std" in col_data and col_data.get("n_valid", 0) > 0:
                    n = col_data["n_valid"]
                    mean = col_data["mean"]
                    std = col_data["std"]
                    sum_x_global += mean * n
                    sum_x2_global += (std**2 + mean**2) * n

                if col_data.get("min") is not None:
                    global_min = min(global_min, col_data["min"])
                if col_data.get("max") is not None:
                    global_max = max(global_max, col_data["max"])

            elif ctype == "categorical":
                for tv in col_data.get("top_values", []):
                    global_counter[tv["value"]] += tv["count"]

        # Tipo dominante
        if type_votes:
            dominant_type = type_votes.most_common(1)[0][0]
        else:
            dominant_type = "unknown"
        col_stats["col_type"] = dominant_type

        # Estadísticas globales numéricas
        if dominant_type == "numeric" and col_stats["n_valid_global"] > 0:
            n = col_stats["n_valid_global"]
            global_mean = sum_x_global / n
            global_var = (sum_x2_global / n) - (global_mean ** 2)
            global_var = max(global_var, 0)

            col_stats.update({
                "global_mean": round(float(global_mean), 6),
                "global_std": round(float(np.sqrt(global_var)), 6),
                "global_min": round(float(global_min), 6) if global_min != float("inf") else None,
                "global_max": round(float(global_max), 6) if global_max != float("-inf") else None,
                "global_cv": round(float(np.sqrt(global_var) / abs(global_mean)) * 100, 2) if global_mean != 0 else None,
            })

        # Estadísticas globales categóricas
        elif dominant_type == "categorical" and global_counter:
            n_unique = len(global_counter)
            col_stats.update({
                "global_n_unique": n_unique,
                "global_top_value": str(global_counter.most_common(1)[0][0]),
                "global_top_count": int(global_counter.most_common(1)[0][1]),
            })

            # Entropía global
            if n_unique > 1:
                probs = np.array(list(global_counter.values()), dtype=np.float64)
                probs = probs / probs.sum()
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                col_stats["global_entropy"] = round(float(entropy), 4)

        # Nulidad global
        if col_stats["n_total_global"] > 0:
            col_stats["global_null_pct"] = round(
                col_stats["n_null_global"] / col_stats["n_total_global"] * 100, 2
            )

        column_master[col_name] = col_stats

    # ====================================================================
    # 2. Mapa de nulidad (trimestre × columna)
    # ====================================================================
    nullity_map = {}
    for prof in profiles:
        quarter = prof["quarter"]
        nullity_map[quarter] = {}
        for col_name, col_data in prof.get("columns", {}).items():
            nullity_map[quarter][col_name] = col_data.get("null_pct", 0)

    # ====================================================================
    # 3. Evolución temporal de variables clave
    # ====================================================================
    key_vars = [
        "borrower_credit_score", "original_dti", "original_ltv",
        "original_interest_rate", "original_upb", "current_actual_upb",
        "current_interest_rate", "loan_age", "current_loan_delinquency_status",
        "original_cltv", "co_borrower_credit_score", "mortgage_insurance_pct",
    ]

    temporal_evolution = {}
    for var in key_vars:
        temporal_evolution[var] = []
        for prof in profiles:
            col_data = prof.get("columns", {}).get(var)
            if col_data is None:
                continue

            entry = {
                "quarter": prof["quarter"],
                "n_valid": col_data.get("n_valid", 0),
                "null_pct": col_data.get("null_pct", 0),
            }

            if col_data.get("column_type") == "numeric":
                entry.update({
                    "mean": col_data.get("mean"),
                    "std": col_data.get("std"),
                    "min": col_data.get("min"),
                    "max": col_data.get("max"),
                })
                if "percentiles" in col_data:
                    entry["p25"] = col_data["percentiles"].get("p25")
                    entry["p50"] = col_data["percentiles"].get("p50")
                    entry["p75"] = col_data["percentiles"].get("p75")

            temporal_evolution[var].append(entry)

    # ====================================================================
    # 4. Ranking de informatividad
    # ====================================================================
    informatividad = []
    for col_name, stats in column_master.items():
        score = 0.0
        reason = ""

        if stats["col_type"] == "numeric":
            cv = stats.get("global_cv")
            if cv is not None:
                score = abs(cv)
                reason = f"CV={cv:.2f}%"
            else:
                score = 0
                reason = "CV no calculable"

        elif stats["col_type"] == "categorical":
            ent = stats.get("global_entropy", 0)
            score = ent
            reason = f"Entropía={ent:.4f}"

        informatividad.append({
            "column": col_name,
            "type": stats["col_type"],
            "score": round(score, 4),
            "reason": reason,
            "null_pct": stats.get("global_null_pct", 0),
            "n_valid": stats.get("n_valid_global", 0),
        })

    informatividad.sort(key=lambda x: -x["score"])

    # ====================================================================
    # Construir perfil global
    # ====================================================================
    global_profile = {
        "total_files_processed": len(profiles),
        "total_rows_global": sum(p.get("total_rows", 0) for p in profiles),
        "total_parquet_gb": round(sum(p.get("parquet_size_gb", 0) for p in profiles), 2),
        "total_csv_gb": round(sum(p.get("file_size_gb", 0) for p in profiles), 2),
        "quarters_covered": all_quarters,
        "column_master": column_master,
        "nullity_map": nullity_map,
        "temporal_evolution": temporal_evolution,
        "informatividad_ranking": informatividad,
        "consolidated_at": datetime.now().isoformat(),
    }

    # Guardar perfil global
    global_profile_path = PERFILES_PATH.parent / "perfil_global.json"
    with open(global_profile_path, "w") as f:
        json.dump(global_profile, f, indent=2, ensure_ascii=False)
    log_message(f"💾 Perfil global guardado: {global_profile_path}")

    # Guardar tabla maestra como CSV
    master_rows = []
    for col_name, stats in column_master.items():
        row = {
            "columna": col_name,
            "tipo": stats["col_type"],
            "n_total": stats["n_total_global"],
            "n_valid": stats["n_valid_global"],
            "null_pct": stats.get("global_null_pct", 0),
            "trimestres_presentes": stats["present_in_quarters"],
        }
        if stats["col_type"] == "numeric":
            row.update({
                "media": stats.get("global_mean"),
                "std": stats.get("global_std"),
                "min": stats.get("global_min"),
                "max": stats.get("global_max"),
                "cv_pct": stats.get("global_cv"),
            })
        elif stats["col_type"] == "categorical":
            row.update({
                "n_unique": stats.get("global_n_unique"),
                "top_valor": stats.get("global_top_value"),
                "entropia": stats.get("global_entropy"),
            })
        master_rows.append(row)

    df_master = pd.DataFrame(master_rows)
    master_csv_path = TABLES_PATH / "perfil_columnas_global.csv"
    df_master.to_csv(master_csv_path, index=False)
    log_message(f"💾 Tabla maestra guardada: {master_csv_path}")

    # Guardar ranking de informatividad
    df_ranking = pd.DataFrame(informatividad)
    ranking_path = TABLES_PATH / "ranking_informatividad.csv"
    df_ranking.to_csv(ranking_path, index=False)
    log_message(f"💾 Ranking de informatividad: {ranking_path}")

    # Guardar mapa de nulidad como CSV
    df_null = pd.DataFrame(nullity_map).T
    df_null.index.name = "quarter"
    null_path = TABLES_PATH / "mapa_nulidad.csv"
    df_null.to_csv(null_path)
    log_message(f"💾 Mapa de nulidad: {null_path}")

    # Imprimir resumen
    print("\n" + "=" * 60)
    print("  RESUMEN DEL PERFIL GLOBAL")
    print("=" * 60)
    print(f"  Archivos procesados:  {len(profiles)}")
    print(f"  Filas totales:        {global_profile['total_rows_global']:,}")
    print(f"  Tamaño CSV original:  {global_profile['total_csv_gb']:.1f} GB")
    print(f"  Tamaño Parquet Zstd:  {global_profile['total_parquet_gb']:.1f} GB")

    comp = global_profile['total_csv_gb'] / max(global_profile['total_parquet_gb'], 0.001)
    print(f"  Ratio compresión:     {comp:.1f}x")

    # Top 10 columnas más informativas
    print(f"\n  Top 10 columnas más informativas:")
    for i, col in enumerate(informatividad[:10], 1):
        print(f"    {i:2d}. {col['column']:<35s} ({col['type']:>12s}) score={col['score']:.4f}")

    # Columnas con >50% nulidad
    high_null = [c for c in informatividad if c["null_pct"] > 50]
    if high_null:
        print(f"\n  ⚠️ {len(high_null)} columnas con >50% nulidad:")
        for c in high_null[:10]:
            print(f"     {c['column']:<35s} {c['null_pct']:.1f}% nulos")

    print("=" * 60)

    return global_profile


# ============================================================================
# VALIDACIÓN POST-CONVERSIÓN (Sub-Fase 0.3)
# ============================================================================

def validate_parquet_files() -> pd.DataFrame:
    """
    Valida los archivos Parquet generados contra los registros del checkpoint.

    Verifica:
    - Que el archivo existe
    - Que el row count coincide con el log
    - Que el schema es correcto
    - Que las columnas numéricas no tienen centinelas residuales

    Returns:
        DataFrame con resultados de validación
    """
    log_message("🔍 Validando archivos Parquet...")

    checkpoint = CheckpointManager()
    state = checkpoint.load_state()
    results = []

    parquet_files = sorted(PANEL_PATH.glob("*.parquet"))
    log_message(f"   Encontrados {len(parquet_files)} archivos Parquet")

    for pq_path in parquet_files:
        fname = pq_path.stem
        result = {
            "filename": pq_path.name,
            "quarter": fname.split("_part")[0],
            "exists": True,
            "size_gb": round(pq_path.stat().st_size / (1024**3), 4),
        }

        try:
            # Leer metadata del Parquet
            pf = pq.ParquetFile(str(pq_path))
            metadata = pf.metadata

            result["num_rows"] = metadata.num_rows
            result["num_columns"] = metadata.num_columns
            result["num_row_groups"] = metadata.num_row_groups

            # Verificar se puede leer el schema
            schema = pf.schema_arrow
            result["schema_valid"] = True
            result["schema_columns"] = schema.names

            # Verificar row count vs checkpoint
            csv_name = f"{result['quarter']}.csv"
            checkpoint_entry = state.get("files", {}).get(csv_name, {})
            expected_rows = checkpoint_entry.get("total_rows")

            if expected_rows is not None:
                result["rows_match"] = metadata.num_rows == expected_rows
                result["expected_rows"] = expected_rows
            else:
                result["rows_match"] = None  # No hay referencia

            # Lectura rápida de primera fila para verificar datos
            first_batch = pf.read_row_group(0).to_pandas().head(5)
            result["readable"] = True

            # Verificar centinelas residuales en columnas clave
            sentinel_check = {}
            for col, sentinels in SENTINEL_VALUES.items():
                if col in first_batch.columns:
                    for sv in sentinels:
                        has_sentinel = (first_batch[col] == sv).any()
                        sentinel_check[col] = not has_sentinel
            result["sentinel_clean"] = all(sentinel_check.values()) if sentinel_check else True

            result["status"] = "OK"

        except Exception as e:
            result["status"] = f"ERROR: {e}"
            result["schema_valid"] = False
            result["readable"] = False

        results.append(result)

    df_results = pd.DataFrame(results)

    # Resumen
    n_ok = (df_results["status"] == "OK").sum()
    n_err = len(df_results) - n_ok
    log_message(f"   ✅ {n_ok} archivos válidos, ❌ {n_err} con errores")

    # Guardar resultados
    val_path = TABLES_PATH / "validacion_parquet.csv"
    df_results.to_csv(val_path, index=False)
    log_message(f"   💾 Resultados guardados: {val_path}")

    return df_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description="Fase 0: Construcción del Panel Analítico (CSV→Parquet + EDA)"
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Usar Ray para distribuir entre Master y Worker"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Retomar desde el último checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignorar checkpoints y reprocesar todo"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Procesar un solo archivo (ej: --file 2003Q3.csv)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Solo validar los Parquet existentes"
    )
    parser.add_argument(
        "--consolidate-only", action="store_true",
        help="Solo consolidar los perfiles JSON existentes"
    )

    args = parser.parse_args()

    # ── Configurar logging ──
    logger = setup_logger("001_csv_to_parquet_parts")

    # Banner
    print("\n" + "=" * 70)
    print("  FASE 0: CONSTRUCCIÓN DEL PANEL ANALÍTICO")
    print("  CSV → Parquet (Zstd) + Perfilado Estadístico Simultáneo")
    print("=" * 70)

    print_system_summary()

    # Modo solo-validación
    if args.validate_only:
        with ProcessingTimer("Validación de Parquet"):
            validate_parquet_files()
        return

    # Modo solo-consolidación
    if args.consolidate_only:
        with ProcessingTimer("Consolidación de perfiles"):
            consolidate_profiles()
        return

    # Verificar que el ZIP existe
    if not ZIP_FILE_PATH.exists():
        print(f"\n  ❌ ERROR: No se encontró el archivo ZIP en: {ZIP_FILE_PATH}")
        print(f"     Verifica que el dataset esté montado vía NFS.")
        sys.exit(1)

    zip_size_gb = ZIP_FILE_PATH.stat().st_size / (1024**3)
    log_message(f"📦 ZIP encontrado: {ZIP_FILE_PATH} ({zip_size_gb:.2f} GB)")

    resume = args.resume and not args.no_resume

    with ProcessingTimer("Fase 0 — Construcción del Panel"):

        # Modo archivo individual
        if args.file:
            log_message(f"🎯 Modo single-file: {args.file}")
            profile = process_single_file(args.file, checkpoint=CheckpointManager())
            if profile.get("status") != "ERROR":
                log_message("✅ Archivo procesado exitosamente")
            return

        # Modo distribuido con Ray
        if args.distributed:
            profiles = run_distributed(resume=resume)
        else:
            # Modo local paralelo: todos los núcleos menos 1
            n_workers = max(1, os.cpu_count() - 1)
            log_message(f"   🖥️ CPUs disponibles: {os.cpu_count()}, usando {n_workers} workers")
            profiles = run_local_parallel(resume=resume, max_workers=n_workers)

        # ── Verificar completitud antes de continuar ──
        checkpoint_final = CheckpointManager()
        all_completed = checkpoint_final.get_completed_count()
        with ZipDataLoader() as _loader:
            total_files = len(_loader.get_file_inventory())
        
        if all_completed < total_files:
            log_message(
                f"\n  ⚠️ FASE 0 INCOMPLETA: {all_completed}/{total_files} archivos completados."
                f"\n     No se procederá a la consolidación hasta completar los {total_files} archivos."
                f"\n     Re-ejecuta con --resume para procesar los {total_files - all_completed} restantes."
            )
            checkpoint_final.print_summary()
            print_memory_status("Final del pipeline (incompleto)")
            sys.exit(1)

        # Sub-Fase 0.3: Validación (solo si todos los 101 archivos están completos)
        log_message("\n" + "=" * 60)
        log_message("  SUB-FASE 0.3: Validación post-conversión")
        log_message("=" * 60)
        validate_parquet_files()

        # Sub-Fase 0.4: Consolidación
        log_message("\n" + "=" * 60)
        log_message("  SUB-FASE 0.4: Consolidación del perfil global")
        log_message("=" * 60)
        consolidate_profiles()

    print_memory_status("Final del pipeline")
    log_message("🏁 Fase 0 completada exitosamente — 101/101 archivos.")


if __name__ == "__main__":
    main()
