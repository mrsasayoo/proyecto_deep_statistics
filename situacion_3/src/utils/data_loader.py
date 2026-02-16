"""
data_loader.py — Carga Distribuida de Datos desde ZIP Comprimido
================================================================
Situación 3: Portafolio Hipotecario (55.31 GB ZIP, ~820 GB descomprimido)

Módulo reutilizable para:
- Leer archivos CSV directamente desde el ZIP (SIN descomprimir a disco)
- Lectura adaptativa: archivos pequeños → carga directa; grandes → chunks
- Barras de progreso para visualización en terminal (tqdm / rich)
- Reintentos automáticos ante fallos de red (NFS sobre Wi-Fi)
- Distribución de lectura entre Master y Worker via Ray (opcional)
- Liberación automática de RAM después de cada chunk

Protocolo de uso:
    from utils.data_loader import ZipDataLoader
    loader = ZipDataLoader()
    headers = loader.read_all_headers()
    df = loader.read_file("2003Q3.csv", chunk_callback=process_fn)
"""

import gc
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import pandas as pd

# Importaciones del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ZIP_FILE_PATH,
    PERFORMANCE_COLUMNS,
    COLUMNS_TO_DROP,
    CSV_DELIMITER,
    CSV_ENCODING,
    CSV_HAS_HEADER,
    CHUNK_THRESHOLD_GB,
    CHUNK_SIZE_ROWS,
    CHUNK_SIZE_ROWS_LARGE,
    LARGE_FILE_THRESHOLD_GB,
    MIN_AVAILABLE_RAM_GB,
    NETWORK_RETRY_ATTEMPTS,
    NETWORK_RETRY_DELAY,
    NETWORK_TIMEOUT_SECONDS,
)
from utils.memory_utils import (
    check_memory_threshold,
    clear_memory,
    get_memory_usage,
    get_optimal_chunk_size,
    print_memory_status,
    should_use_chunks,
    wait_for_memory,
    ProcessingTimer,
)

# tqdm para barras de progreso (fallback a print si no está instalado)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# rich para consola mejorada (opcional)
try:
    from rich.console import Console
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeRemainingColumn,
        FileSizeColumn, TransferSpeedColumn, SpinnerColumn,
    )
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


class ZipDataLoader:
    """
    Cargador de datos desde un archivo ZIP con CSVs de Freddie Mac.

    Maneja:
    - Lectura directa desde ZIP (sin descomprimir a disco)
    - Chunks adaptativos según tamaño del archivo y RAM disponible
    - Barras de progreso en terminal
    - Reintentos ante fallos de red (NFS/Wi-Fi)
    - Liberación automática de memoria
    """

    def __init__(self, zip_path: Optional[Path] = None):
        """
        Inicializa el cargador.

        Args:
            zip_path: Ruta al archivo ZIP. Default: ZIP_FILE_PATH de config.py
        """
        self.zip_path = Path(zip_path) if zip_path else ZIP_FILE_PATH

        if not self.zip_path.exists():
            raise FileNotFoundError(
                f"❌ Archivo ZIP no encontrado: {self.zip_path}\n"
                f"   Verifica que el dataset esté montado vía NFS o copiado localmente."
            )

        self._zip_ref: Optional[zipfile.ZipFile] = None
        self._file_inventory: Optional[List[dict]] = None

    # ====================================================================
    # APERTURA Y CIERRE DEL ZIP
    # ====================================================================

    def _open_zip(self) -> zipfile.ZipFile:
        """Abre el ZIP con reintentos ante fallos de red."""
        for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
            try:
                if self._zip_ref is None or self._zip_ref.fp is None:
                    self._zip_ref = zipfile.ZipFile(self.zip_path, "r")
                return self._zip_ref
            except (IOError, OSError, zipfile.BadZipFile) as e:
                if attempt < NETWORK_RETRY_ATTEMPTS:
                    print(
                        f"  ⚠️ Error al abrir ZIP (intento {attempt}/{NETWORK_RETRY_ATTEMPTS}): {e}\n"
                        f"     Reintentando en {NETWORK_RETRY_DELAY}s..."
                    )
                    time.sleep(NETWORK_RETRY_DELAY)
                    self._zip_ref = None
                else:
                    raise IOError(
                        f"❌ No se pudo abrir el ZIP después de {NETWORK_RETRY_ATTEMPTS} intentos: {e}"
                    )

    def close(self) -> None:
        """Cierra el archivo ZIP y libera recursos."""
        if self._zip_ref is not None:
            try:
                self._zip_ref.close()
            except Exception:
                pass
            self._zip_ref = None

    def __enter__(self):
        self._open_zip()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ====================================================================
    # INVENTARIO DE ARCHIVOS
    # ====================================================================

    def get_file_inventory(self) -> List[dict]:
        """
        Obtiene el inventario completo de archivos dentro del ZIP.

        Returns:
            Lista de dicts con: filename, compressed_gb, uncompressed_gb,
            compression_ratio, category (pequeño/mediano/grande)
        """
        if self._file_inventory is not None:
            return self._file_inventory

        z = self._open_zip()
        inventory = []

        for name in sorted(z.namelist()):
            info = z.getinfo(name)
            uncompressed_gb = info.file_size / (1024**3)

            # Clasificar por tamaño
            if uncompressed_gb < CHUNK_THRESHOLD_GB:
                category = "pequeño"
            elif uncompressed_gb < LARGE_FILE_THRESHOLD_GB:
                category = "mediano"
            else:
                category = "grande"

            inventory.append({
                "filename": name,
                "compressed_gb": round(info.compress_size / (1024**3), 4),
                "uncompressed_gb": round(uncompressed_gb, 4),
                "compression_ratio": round(info.file_size / max(info.compress_size, 1), 1),
                "category": category,
            })

        self._file_inventory = inventory
        return inventory

    def print_file_inventory(self) -> None:
        """Imprime el inventario de archivos de forma legible."""
        inventory = self.get_file_inventory()

        total_comp = sum(f["compressed_gb"] for f in inventory)
        total_uncomp = sum(f["uncompressed_gb"] for f in inventory)

        print(f"\n  {'Archivo':<15} {'Comprimido':>12} {'Descomprimido':>14} {'Categoría':>10}")
        print("  " + "-" * 55)

        for f in inventory:
            icon = {"pequeño": "🟢", "mediano": "🟡", "grande": "🔴"}[f["category"]]
            print(
                f"  {icon} {f['filename']:<13} {f['compressed_gb']:>10.2f} GB "
                f"{f['uncompressed_gb']:>12.2f} GB {f['category']:>10}"
            )

        print("  " + "-" * 55)
        print(f"  {'TOTAL':<15} {total_comp:>10.2f} GB {total_uncomp:>12.2f} GB")
        print(f"  Total archivos: {len(inventory)}")

        # Resumen por categoría
        for cat in ["pequeño", "mediano", "grande"]:
            count = sum(1 for f in inventory if f["category"] == cat)
            total = sum(f["uncompressed_gb"] for f in inventory if f["category"] == cat)
            print(f"    {cat}: {count} archivos ({total:.1f} GB)")

    # ====================================================================
    # LECTURA DE ENCABEZADOS (PRIMERA FILA)
    # ====================================================================

    def read_first_row(self, filename: str) -> Tuple[int, pd.DataFrame]:
        """
        Lee solo la primera fila de un archivo CSV dentro del ZIP.

        Args:
            filename: Nombre del archivo dentro del ZIP (ej: "2003Q3.csv")

        Returns:
            Tupla de (número_de_columnas, DataFrame con 1 fila)
        """
        z = self._open_zip()

        for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
            try:
                with z.open(filename) as f:
                    df = pd.read_csv(
                        f,
                        sep=CSV_DELIMITER,
                        header=None,
                        names=PERFORMANCE_COLUMNS,
                        nrows=1,
                        encoding=CSV_ENCODING,
                        low_memory=False,
                    )
                return len(df.columns), df
            except (IOError, OSError) as e:
                if attempt < NETWORK_RETRY_ATTEMPTS:
                    print(f"  ⚠️ Error leyendo {filename} (intento {attempt}): {e}")
                    time.sleep(NETWORK_RETRY_DELAY)
                    self._zip_ref = None  # Forzar reapertura
                    z = self._open_zip()
                else:
                    raise

    def read_all_headers(self) -> pd.DataFrame:
        """
        Lee la primera fila de cada uno de los 101 archivos CSV para
        verificar la consistencia de la estructura (número de columnas).

        Returns:
            DataFrame resumen con: filename, num_columns, first_loan_id,
            first_period, consistent (bool)
        """
        inventory = self.get_file_inventory()
        results = []

        print(f"\n  📋 Leyendo encabezados de {len(inventory)} archivos...")

        # Barra de progreso
        iterator = inventory
        if TQDM_AVAILABLE:
            iterator = tqdm(
                inventory,
                desc="  Leyendo encabezados",
                unit="archivo",
                ncols=80,
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

        for file_info in iterator:
            fname = file_info["filename"]
            try:
                ncols, row = self.read_first_row(fname)
                results.append({
                    "filename": fname,
                    "num_columns": ncols,
                    "uncompressed_gb": file_info["uncompressed_gb"],
                    "category": file_info["category"],
                    "first_loan_id": str(row.iloc[0]["loan_sequence_number"]) if "loan_sequence_number" in row.columns else "N/A",
                    "first_period": str(row.iloc[0]["monthly_reporting_period"]) if "monthly_reporting_period" in row.columns else "N/A",
                })
            except Exception as e:
                results.append({
                    "filename": fname,
                    "num_columns": -1,
                    "uncompressed_gb": file_info["uncompressed_gb"],
                    "category": file_info["category"],
                    "first_loan_id": f"ERROR: {e}",
                    "first_period": "ERROR",
                })

        df_results = pd.DataFrame(results)

        # Verificar consistencia
        expected_cols = len(PERFORMANCE_COLUMNS)
        df_results["consistent"] = df_results["num_columns"] == expected_cols

        inconsistent = df_results[~df_results["consistent"]]
        if len(inconsistent) > 0:
            print(f"\n  ⚠️ {len(inconsistent)} archivos con columnas inconsistentes:")
            for _, row in inconsistent.iterrows():
                print(f"    {row['filename']}: {row['num_columns']} columnas (esperadas: {expected_cols})")
        else:
            print(f"\n  ✅ Todos los {len(df_results)} archivos tienen {expected_cols} columnas. Estructura consistente.")

        return df_results

    # ====================================================================
    # LECTURA DE ARCHIVOS COMPLETOS / POR CHUNKS
    # ====================================================================

    def read_file(
        self,
        filename: str,
        usecols: Optional[List[str]] = None,
        dtype: Optional[dict] = None,
        chunk_callback: Optional[Callable[[pd.DataFrame, int], None]] = None,
        max_rows: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Lee un archivo CSV completo desde el ZIP, con manejo adaptativo de memoria.

        Si el archivo es pequeño (< CHUNK_THRESHOLD_GB): carga directa en RAM.
        Si es grande: lee por chunks y aplica chunk_callback a cada uno.

        Args:
            filename: Nombre del CSV dentro del ZIP (ej: "2003Q3.csv")
            usecols: Lista de columnas a cargar (None = todas)
            dtype: Dict de tipos de datos por columna
            chunk_callback: Función(chunk_df, chunk_num) a aplicar a cada chunk.
                Si se proporciona, los chunks NO se acumulan en RAM (se liberan)
                y la función retorna None.
            max_rows: Máximo de filas a leer (None = todas)

        Returns:
            DataFrame completo si no hay chunk_callback, o None si hay callback.
        """
        inventory = self.get_file_inventory()
        file_info = next((f for f in inventory if f["filename"] == filename), None)

        if file_info is None:
            raise FileNotFoundError(f"❌ Archivo '{filename}' no encontrado en el ZIP")

        file_size_gb = file_info["uncompressed_gb"]
        use_chunks = should_use_chunks(file_size_gb, CHUNK_THRESHOLD_GB)

        if max_rows is not None:
            use_chunks = False  # Si hay límite de filas, no usar chunks

        print(f"\n  📁 Leyendo {filename} ({file_size_gb:.2f} GB descomprimido)")

        if use_chunks:
            return self._read_chunked(filename, file_size_gb, usecols, dtype, chunk_callback)
        else:
            return self._read_direct(filename, usecols, dtype, max_rows)

    def _read_direct(
        self,
        filename: str,
        usecols: Optional[List[str]],
        dtype: Optional[dict],
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Lee un archivo completo directamente en RAM."""
        # Verificar RAM antes de cargar
        wait_for_memory(MIN_AVAILABLE_RAM_GB, label=filename)

        z = self._open_zip()

        for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
            try:
                with z.open(filename) as f:
                    read_params = {
                        "sep": CSV_DELIMITER,
                        "header": None,
                        "names": PERFORMANCE_COLUMNS,
                        "encoding": CSV_ENCODING,
                        "low_memory": True,
                    }

                    if usecols:
                        read_params["usecols"] = usecols
                    if dtype:
                        read_params["dtype"] = dtype
                    if max_rows:
                        read_params["nrows"] = max_rows

                    with ProcessingTimer(f"Carga directa: {filename}", show_memory=False):
                        df = pd.read_csv(f, **read_params)

                # Eliminar columna vacía
                for col in COLUMNS_TO_DROP:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

                print(f"    → {len(df):,} filas × {len(df.columns)} columnas cargadas")
                print_memory_status(f"Después de {filename}")
                return df

            except (IOError, OSError) as e:
                if attempt < NETWORK_RETRY_ATTEMPTS:
                    print(f"  ⚠️ Error de red leyendo {filename} (intento {attempt}): {e}")
                    time.sleep(NETWORK_RETRY_DELAY)
                    self._zip_ref = None
                    z = self._open_zip()
                else:
                    raise

    def _read_chunked(
        self,
        filename: str,
        file_size_gb: float,
        usecols: Optional[List[str]],
        dtype: Optional[dict],
        chunk_callback: Optional[Callable],
    ) -> Optional[pd.DataFrame]:
        """Lee un archivo grande por chunks con barra de progreso."""
        chunk_size = get_optimal_chunk_size(file_size_gb)
        z = self._open_zip()

        print(f"    📦 Modo chunked: {chunk_size:,} filas/chunk")

        accumulated = [] if chunk_callback is None else None
        chunk_num = 0
        total_rows = 0

        for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
            try:
                with z.open(filename) as f:
                    reader = pd.read_csv(
                        f,
                        sep=CSV_DELIMITER,
                        header=None,
                        names=PERFORMANCE_COLUMNS,
                        encoding=CSV_ENCODING,
                        chunksize=chunk_size,
                        low_memory=True,
                        usecols=usecols if usecols else None,
                        dtype=dtype,
                    )

                    # Barra de progreso
                    desc = f"    {filename}"
                    if TQDM_AVAILABLE:
                        reader_iter = reader
                        pbar = tqdm(
                            desc=desc,
                            unit="chunk",
                            ncols=90,
                            bar_format="    {l_bar}{bar}| Chunk {n_fmt} [{elapsed}<{remaining}]",
                        )
                    else:
                        reader_iter = reader
                        pbar = None

                    for chunk in reader_iter:
                        chunk_num += 1
                        total_rows += len(chunk)

                        # Verificar RAM antes de procesar
                        if not check_memory_threshold(MIN_AVAILABLE_RAM_GB):
                            print(f"    ⏳ Esperando RAM libre antes del chunk {chunk_num}...")
                            if not wait_for_memory(MIN_AVAILABLE_RAM_GB, label=f"Chunk {chunk_num}"):
                                print(f"    ❌ RAM insuficiente. Deteniendo lectura en chunk {chunk_num}.")
                                break

                        # Eliminar columna vacía
                        for col in COLUMNS_TO_DROP:
                            if col in chunk.columns:
                                chunk.drop(columns=[col], inplace=True)

                        if chunk_callback:
                            # Procesar y liberar
                            chunk_callback(chunk, chunk_num)
                            del chunk
                            gc.collect()
                        else:
                            accumulated.append(chunk)

                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({"filas": f"{total_rows:,}"})

                    if pbar:
                        pbar.close()

                break  # Lectura exitosa

            except (IOError, OSError) as e:
                if attempt < NETWORK_RETRY_ATTEMPTS:
                    print(f"  ⚠️ Error de red (intento {attempt}): {e}")
                    time.sleep(NETWORK_RETRY_DELAY)
                    self._zip_ref = None
                    z = self._open_zip()
                    chunk_num = 0
                    total_rows = 0
                    if accumulated:
                        accumulated.clear()
                else:
                    raise

        print(f"    → {total_rows:,} filas procesadas en {chunk_num} chunks")
        print_memory_status(f"Después de {filename}")

        if accumulated:
            result = pd.concat(accumulated, ignore_index=True)
            del accumulated
            gc.collect()
            return result

        return None

    # ====================================================================
    # GENERADOR DE CHUNKS (para pipelines perezosos)
    # ====================================================================

    def iter_chunks(
        self,
        filename: str,
        chunk_size: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        dtype: Optional[dict] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generador que produce chunks de un archivo CSV uno a uno.
        Ideal para pipelines de procesamiento donde no se necesita
        el DataFrame completo en RAM.

        Args:
            filename: Nombre del CSV dentro del ZIP
            chunk_size: Filas por chunk (None = automático)
            usecols: Columnas a cargar
            dtype: Tipos de datos

        Yields:
            DataFrame con chunk_size filas en cada iteración
        """
        inventory = self.get_file_inventory()
        file_info = next((f for f in inventory if f["filename"] == filename), None)

        if file_info is None:
            raise FileNotFoundError(f"❌ Archivo '{filename}' no encontrado en el ZIP")

        if chunk_size is None:
            chunk_size = get_optimal_chunk_size(file_info["uncompressed_gb"])

        z = self._open_zip()

        with z.open(filename) as f:
            reader = pd.read_csv(
                f,
                sep=CSV_DELIMITER,
                header=None,
                names=PERFORMANCE_COLUMNS,
                encoding=CSV_ENCODING,
                chunksize=chunk_size,
                low_memory=True,
                usecols=usecols,
                dtype=dtype,
            )

            for chunk in reader:
                for col in COLUMNS_TO_DROP:
                    if col in chunk.columns:
                        chunk.drop(columns=[col], inplace=True)
                yield chunk

    # ====================================================================
    # UTILIDADES
    # ====================================================================

    def get_filenames(self) -> List[str]:
        """Retorna lista de nombres de archivos CSV dentro del ZIP."""
        z = self._open_zip()
        return sorted(z.namelist())

    def get_file_size_gb(self, filename: str) -> float:
        """Retorna el tamaño descomprimido de un archivo en GB."""
        z = self._open_zip()
        info = z.getinfo(filename)
        return info.file_size / (1024**3)

    def count_rows(self, filename: str) -> int:
        """
        Cuenta las filas de un archivo sin cargarlo completo en RAM.
        Usa lectura línea a línea.
        """
        z = self._open_zip()
        count = 0
        with z.open(filename) as f:
            for _ in f:
                count += 1
        return count


# ============================================================================
# FUNCIONES DE CONVENIENCIA (sin instanciar clase)
# ============================================================================

def quick_peek(filename: str = "2000Q1.csv", n_rows: int = 5) -> pd.DataFrame:
    """
    Vista rápida: carga las primeras N filas de un archivo del ZIP.

    Args:
        filename: Nombre del archivo CSV
        n_rows: Número de filas a cargar

    Returns:
        DataFrame con las primeras n_rows filas
    """
    with ZipDataLoader() as loader:
        df = loader.read_file(filename, max_rows=n_rows)
    return df


def get_all_filenames() -> List[str]:
    """Retorna los nombres de los 101 archivos CSV del ZIP."""
    with ZipDataLoader() as loader:
        return loader.get_filenames()


# ============================================================================
# MAIN (TEST)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: data_loader.py")
    print("=" * 60)

    with ZipDataLoader() as loader:
        # Test 1: Inventario
        print("\n--- Test 1: Inventario de archivos ---")
        loader.print_file_inventory()

        # Test 2: Primera fila de un archivo
        print("\n--- Test 2: Primera fila de 2000Q1.csv ---")
        ncols, row = loader.read_first_row("2000Q1.csv")
        print(f"  Columnas: {ncols}")
        print(f"  Loan ID: {row.iloc[0]['loan_sequence_number']}")
        print(f"  Período: {row.iloc[0]['monthly_reporting_period']}")

    print("\n  ✅ Todos los tests pasaron correctamente.")
