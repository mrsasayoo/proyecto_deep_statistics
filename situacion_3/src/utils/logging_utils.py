"""
logging_utils.py — Sistema de Logging Estructurado
====================================================
Situación 3: Portafolio Hipotecario

Proporciona logging dual (consola + archivo) con:
- Rotación automática de archivos de log (max 50 MB, 5 backups)
- Timestamps precisos (milisegundos) para diagnosticar latencia de I/O
- Niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Contexto automático: PID, RAM disponible, hostname
- Trazas completas de excepciones

Uso:
    from utils.logging_utils import setup_logger, get_logger

    # Al inicio del script:
    setup_logger("01_construccion_panel")

    # Dentro de funciones:
    logger = get_logger()
    logger.info("Procesando archivo %s", filename)
    logger.error("Fallo en chunk %d", chunk_num, exc_info=True)
"""

import logging
import os
import platform
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import psutil


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Referencia al directorio de logs (importado al inicializar)
_LOGS_PATH: Optional[Path] = None
_logger: Optional[logging.Logger] = None

# Formato con timestamp preciso (milisegundos) para diagnóstico de I/O en HDD
LOG_FORMAT_FILE = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)
LOG_FORMAT_CONSOLE = (
    "  %(asctime)s | %(levelname)-8s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotación: 50 MB por archivo, 5 backups = máximo ~300 MB de logs
LOG_MAX_BYTES = 50 * 1024 * 1024   # 50 MB
LOG_BACKUP_COUNT = 5


# ============================================================================
# SETUP
# ============================================================================

def setup_logger(
    script_name: str,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    logs_path: Optional[Path] = None,
) -> logging.Logger:
    """
    Configura el logger principal para un script del pipeline.

    Crea dos handlers:
    1. Archivo rotativo en logs/<script_name>.log (nivel DEBUG)
    2. Consola (nivel INFO o superior)

    Args:
        script_name: Nombre del script (sin .py) — se usa como nombre del log
        level: Nivel mínimo para el archivo (default: DEBUG)
        console_level: Nivel mínimo para consola (default: INFO)
        logs_path: Directorio de logs (default: desde config.LOGS_PATH)

    Returns:
        Logger configurado
    """
    global _LOGS_PATH, _logger

    # Resolver directorio de logs
    if logs_path is None:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import LOGS_PATH
            _LOGS_PATH = LOGS_PATH
        except ImportError:
            _LOGS_PATH = Path(__file__).parent.parent.parent / "logs"
    else:
        _LOGS_PATH = logs_path

    _LOGS_PATH.mkdir(parents=True, exist_ok=True)

    # Crear logger
    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    # Evitar handlers duplicados si se llama varias veces
    if logger.handlers:
        logger.handlers.clear()

    # ── Handler 1: Archivo rotativo ──
    log_file = _LOGS_PATH / f"{script_name}.log"
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT_FILE, LOG_DATE_FORMAT))
    logger.addHandler(file_handler)

    # ── Handler 2: Consola ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT_CONSOLE, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # Banner de inicio
    logger.info("=" * 70)
    logger.info("INICIO: %s", script_name)
    logger.info("Timestamp: %s", datetime.now().isoformat())
    logger.info("Host: %s | PID: %d", platform.node(), os.getpid())
    logger.info("Python: %s", platform.python_version())

    mem = psutil.virtual_memory()
    logger.info(
        "RAM: %.1f GB total, %.1f GB disponible (%.1f%% usado)",
        mem.total / (1024**3),
        mem.available / (1024**3),
        mem.percent,
    )

    cpu = psutil.cpu_count(logical=True)
    logger.info("CPU: %d hilos lógicos", cpu)
    logger.info("Log file: %s", log_file)
    logger.info("=" * 70)

    _logger = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Obtiene el logger actual. Si no se ha configurado, crea uno básico.

    Args:
        name: Nombre del logger (None = el último configurado con setup_logger)

    Returns:
        Logger
    """
    global _logger

    if name:
        return logging.getLogger(name)

    if _logger is None:
        # Fallback: logger básico a consola
        _logger = logging.getLogger("situacion3")
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(LOG_FORMAT_CONSOLE, LOG_DATE_FORMAT))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)

    return _logger


# ============================================================================
# HELPERS DE LOGGING CON CONTEXTO
# ============================================================================

def log_memory_snapshot(label: str = "") -> None:
    """
    Escribe un snapshot de memoria al log (nivel DEBUG).
    Útil para rastrear memory leaks o puntos de alto uso.

    Args:
        label: Etiqueta descriptiva del momento
    """
    logger = get_logger()
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_mb = proc.memory_info().rss / (1024**2)

    logger.debug(
        "MEM_SNAPSHOT [%s] sistema=%.1f%% (%.1f/%.1f GB disp.) | "
        "proceso=%.0f MB | swap=%.1f GB",
        label,
        mem.percent,
        mem.available / (1024**3),
        mem.total / (1024**3),
        proc_mb,
        psutil.swap_memory().used / (1024**3),
    )


def log_io_timing(operation: str, elapsed_sec: float, bytes_processed: int = 0) -> None:
    """
    Registra tiempos de I/O para diagnosticar cuellos de botella en HDD/NFS.

    Args:
        operation: Descripción de la operación (ej: "Lectura 2003Q3.csv")
        elapsed_sec: Tiempo transcurrido en segundos
        bytes_processed: Bytes leídos/escritos (0 si no aplica)
    """
    logger = get_logger()

    if bytes_processed > 0:
        mb = bytes_processed / (1024**2)
        throughput = mb / max(elapsed_sec, 0.001)
        logger.info(
            "IO_TIMING [%s] %.2fs | %.1f MB | %.1f MB/s",
            operation, elapsed_sec, mb, throughput,
        )
    else:
        logger.info("IO_TIMING [%s] %.2fs", operation, elapsed_sec)


def log_exception(context: str) -> None:
    """
    Registra una excepción con traceback completo al log.

    Args:
        context: Contexto donde ocurrió la excepción
    """
    logger = get_logger()
    logger.error("EXCEPCIÓN en [%s]", context, exc_info=True)
    log_memory_snapshot(f"post-error:{context}")


def log_checkpoint(filename: str, status: str, details: str = "") -> None:
    """
    Registra un cambio de estado de checkpoint.

    Args:
        filename: Archivo afectado
        status: Nuevo estado (PENDIENTE, EN_PROCESO, COMPLETADO, ERROR)
        details: Detalles adicionales
    """
    logger = get_logger()
    logger.info("CHECKPOINT [%s] → %s %s", filename, status, details)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger("test_logging")
    logger.debug("Mensaje DEBUG — solo en archivo")
    logger.info("Mensaje INFO — consola + archivo")
    logger.warning("Mensaje WARNING")
    logger.error("Mensaje ERROR")

    log_memory_snapshot("test")
    log_io_timing("test_read", 2.5, 500 * 1024 * 1024)

    try:
        1 / 0
    except ZeroDivisionError:
        log_exception("test_division")

    log_checkpoint("2003Q3.csv", "COMPLETADO", "100,000 filas, 0.5 GB Parquet")
    logger.info("Test de logging completado")
