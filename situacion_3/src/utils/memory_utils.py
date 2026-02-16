"""
memory_utils.py — Gestión de Memoria y Recursos del Sistema
============================================================
Situación 3: Portafolio Hipotecario (~820 GB descomprimido)

Funciones para:
- Monitorear uso de RAM en tiempo real
- Liberar memoria de forma agresiva (del + gc.collect)
- Pausar ejecución si la RAM está saturada
- Verificar umbrales antes de cargar datos pesados
- Medir tiempos de procesamiento

Protocolo de memoria:
  1. Verificar RAM disponible ANTES de cargar un chunk
  2. Procesar el chunk
  3. del variable_pesada
  4. gc.collect()
  5. Verificar RAM de nuevo
"""

import gc
import os
import sys
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ============================================================================
# MONITOREO DE MEMORIA
# ============================================================================

def get_memory_usage() -> dict:
    """
    Obtiene el uso actual de memoria RAM del sistema.

    Returns:
        dict con: total_gb, used_gb, available_gb, percent_used, swap_used_gb
    """
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "percent_used": mem.percent,
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "swap_total_gb": round(swap.total / (1024**3), 2),
    }


def get_process_memory_mb() -> float:
    """
    Obtiene la memoria RAM usada por el proceso Python actual (en MB).

    Returns:
        Memoria RSS del proceso en MB.
    """
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024**2), 1)


def print_memory_status(label: str = "", show_swap: bool = False) -> dict:
    """
    Imprime el estado actual de la memoria RAM de forma visual.

    Args:
        label: Etiqueta descriptiva del momento (ej: "Antes de cargar chunk 3")
        show_swap: Si True, también muestra uso de swap

    Returns:
        dict con el estado de memoria actual
    """
    mem = get_memory_usage()
    proc_mb = get_process_memory_mb()
    prefix = f"[{label}] " if label else ""

    # Barra visual de uso de RAM
    bar_len = 20
    filled = int(bar_len * mem["percent_used"] / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Color semántico basado en porcentaje
    if mem["percent_used"] >= 90:
        status_icon = "🔴"
    elif mem["percent_used"] >= 80:
        status_icon = "🟡"
    else:
        status_icon = "🟢"

    print(
        f"  {status_icon} {prefix}RAM: [{bar}] {mem['percent_used']}% | "
        f"{mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB | "
        f"Disponible: {mem['available_gb']:.1f} GB | "
        f"Proceso: {proc_mb:.0f} MB"
    )

    if show_swap and mem["swap_total_gb"] > 0:
        print(
            f"     SWAP: {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB"
        )

    return mem


# ============================================================================
# LIBERACIÓN DE MEMORIA
# ============================================================================

def clear_memory(*args) -> int:
    """
    Fuerza la recolección de basura de Python para liberar RAM.
    Llámese DESPUÉS de hacer `del variable_pesada`.

    Args:
        *args: Ignorados (para compatibilidad con llamadas genéricas)

    Returns:
        Número de objetos liberados por gc.collect()
    """
    collected = gc.collect()
    return collected


def aggressive_memory_cleanup() -> dict:
    """
    Limpieza agresiva de memoria:
    1. Forzar gc.collect() múltiples veces
    2. Compactar si es posible

    Returns:
        dict con memoria antes y después de la limpieza
    """
    mem_before = get_memory_usage()

    # Múltiples pasadas de gc para objetos con referencias circulares
    for _ in range(3):
        gc.collect()

    mem_after = get_memory_usage()
    freed_mb = (mem_before["used_gb"] - mem_after["used_gb"]) * 1024

    if freed_mb > 1:
        print(f"  🧹 Limpieza agresiva: {freed_mb:.0f} MB liberados")

    return {"before": mem_before, "after": mem_after, "freed_mb": round(freed_mb, 1)}


# ============================================================================
# VERIFICACIÓN DE UMBRALES
# ============================================================================

def check_memory_threshold(
    min_available_gb: float = 3.0,
    raise_error: bool = False,
) -> bool:
    """
    Verifica que haya suficiente RAM disponible antes de una operación pesada.

    Args:
        min_available_gb: Mínimo de GB libres requeridos.
        raise_error: Si True, lanza MemoryError. Si False, retorna bool.

    Returns:
        True si hay suficiente memoria, False si no.

    Raises:
        MemoryError: Solo si raise_error=True y no hay suficiente RAM.
    """
    mem = get_memory_usage()
    if mem["available_gb"] >= min_available_gb:
        return True

    msg = (
        f"⚠️ RAM insuficiente: {mem['available_gb']:.1f} GB disponibles, "
        f"se requieren al menos {min_available_gb} GB. "
        f"(Uso actual: {mem['percent_used']}%)"
    )

    if raise_error:
        raise MemoryError(msg)

    print(f"  {msg}")
    return False


def wait_for_memory(
    min_available_gb: float = 3.0,
    max_retries: int = 5,
    wait_seconds: int = 10,
    label: str = "",
) -> bool:
    """
    Espera hasta que haya suficiente RAM disponible.
    Útil cuando se procesan chunks en paralelo y la RAM se satura temporalmente.

    Args:
        min_available_gb: Mínimo de GB libres requeridos
        max_retries: Número máximo de intentos de espera
        wait_seconds: Segundos entre cada verificación
        label: Etiqueta para los mensajes de log

    Returns:
        True si se liberó suficiente RAM, False si se agotaron los reintentos
    """
    for attempt in range(1, max_retries + 1):
        if check_memory_threshold(min_available_gb):
            return True

        prefix = f"[{label}] " if label else ""
        print(
            f"  ⏳ {prefix}Esperando RAM libre ({attempt}/{max_retries})... "
            f"Reintentando en {wait_seconds}s"
        )

        # Intentar liberar memoria mientras esperamos
        aggressive_memory_cleanup()
        time.sleep(wait_seconds)

    print(f"  ❌ No se liberó suficiente RAM después de {max_retries} intentos.")
    return False


def should_use_chunks(file_size_gb: float, threshold_gb: float = 2.0) -> bool:
    """
    Determina si un archivo debe leerse por chunks o completo.

    Args:
        file_size_gb: Tamaño del archivo descomprimido en GB
        threshold_gb: Umbral a partir del cual usar chunks

    Returns:
        True si debe usarse lectura por chunks
    """
    return file_size_gb > threshold_gb


def get_optimal_chunk_size(
    file_size_gb: float,
    chunk_size_normal: int = 500_000,
    chunk_size_large: int = 250_000,
    large_threshold_gb: float = 10.0,
) -> int:
    """
    Calcula el tamaño óptimo de chunk basado en el tamaño del archivo
    y la RAM disponible.

    Args:
        file_size_gb: Tamaño del archivo en GB
        chunk_size_normal: Filas por chunk para archivos medianos
        chunk_size_large: Filas por chunk para archivos grandes
        large_threshold_gb: Umbral para considerar archivo como "grande"

    Returns:
        Número de filas por chunk recomendado
    """
    mem = get_memory_usage()

    # Si queda poca RAM, reducir aún más el chunk
    if mem["available_gb"] < 5.0:
        return chunk_size_large // 2  # 125,000 filas mínimo

    if file_size_gb > large_threshold_gb:
        return chunk_size_large

    return chunk_size_normal


# ============================================================================
# TIMER DE PROCESAMIENTO
# ============================================================================

class ProcessingTimer:
    """
    Cronómetro para medir y reportar tiempos de procesamiento.

    Uso:
        with ProcessingTimer("Cargando archivo Q3 2003"):
            data = load_file(...)
    """

    def __init__(self, label: str = "Procesamiento", show_memory: bool = True):
        self.label = label
        self.show_memory = show_memory
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"\n  ⏱️  Iniciando: {self.label}")
        if self.show_memory:
            print_memory_status("Inicio")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if exc_type:
            print(f"  ❌ Error en '{self.label}' después de {elapsed_str}")
        else:
            print(f"  ✅ Completado: {self.label} ({elapsed_str})")

        if self.show_memory:
            print_memory_status("Final")

        return False  # No suprimir excepciones

    @property
    def elapsed_seconds(self) -> float:
        """Retorna el tiempo transcurrido en segundos."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


# ============================================================================
# ESTIMACIÓN DE RECURSOS NECESARIOS
# ============================================================================

def estimate_memory_for_file(
    file_size_gb: float,
    overhead_factor: float = 2.5,
) -> dict:
    """
    Estima la RAM necesaria para procesar un archivo CSV.
    La regla general: un CSV en pandas ocupa ~2-3x su tamaño en disco.

    Args:
        file_size_gb: Tamaño del archivo descomprimido en GB
        overhead_factor: Factor multiplicador (pandas overhead)

    Returns:
        dict con estimaciones de RAM necesaria
    """
    estimated_ram_gb = file_size_gb * overhead_factor
    mem = get_memory_usage()
    fits_in_ram = estimated_ram_gb < mem["available_gb"]

    return {
        "file_size_gb": file_size_gb,
        "estimated_ram_gb": round(estimated_ram_gb, 2),
        "available_ram_gb": mem["available_gb"],
        "fits_in_ram": fits_in_ram,
        "recommended_chunks": max(1, int(estimated_ram_gb / (mem["available_gb"] * 0.6))),
    }


def print_system_summary() -> None:
    """
    Imprime un resumen completo de los recursos del sistema.
    Útil al inicio de cada script para documentar el entorno.
    """
    import platform

    mem = get_memory_usage()
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)

    print("\n" + "=" * 60)
    print("  RESUMEN DEL SISTEMA")
    print("=" * 60)
    print(f"  SO:              {platform.system()} {platform.release()}")
    print(f"  Hostname:        {platform.node()}")
    print(f"  Python:          {platform.python_version()}")
    print(f"  CPU (físicos):   {cpu_physical} núcleos")
    print(f"  CPU (lógicos):   {cpu_count} hilos")
    print(f"  RAM Total:       {mem['total_gb']:.1f} GB")
    print(f"  RAM Disponible:  {mem['available_gb']:.1f} GB")
    print(f"  RAM Uso:         {mem['percent_used']}%")
    if mem["swap_total_gb"] > 0:
        print(f"  SWAP:            {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB")
    print(f"  PID del proceso: {os.getpid()}")
    print("=" * 60 + "\n")


# ============================================================================
# MAIN (TEST)
# ============================================================================

if __name__ == "__main__":
    print_system_summary()

    with ProcessingTimer("Test de memoria"):
        print_memory_status("Estado actual", show_swap=True)

        # Simular carga pesada
        print("\n  Simulando carga pesada (100 MB)...")
        heavy_data = bytearray(100 * 1024 * 1024)  # 100 MB
        print_memory_status("Después de cargar 100 MB")

        # Liberar
        del heavy_data
        clear_memory()
        print_memory_status("Después de liberar")

    # Test de estimación
    print("\n  --- Estimación para archivo de 37 GB ---")
    estimate = estimate_memory_for_file(37.0)
    for k, v in estimate.items():
        print(f"  {k}: {v}")
