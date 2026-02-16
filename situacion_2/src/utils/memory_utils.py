"""
memory_utils.py - Gestión de memoria y recursos del sistema
============================================================
Funciones para monitorear y liberar RAM durante el procesamiento
de datos pesados (2.1 GB de imágenes multiespectrales).

Regla: Después de cada proceso pesado → del variable + gc.collect()
"""

import gc
import os
import psutil
import time
from datetime import datetime
from pathlib import Path


def get_memory_usage() -> dict:
    """
    Obtiene el uso actual de memoria RAM del sistema.
    
    Returns:
        dict con: total_gb, used_gb, available_gb, percent_used
    """
    mem = psutil.virtual_memory()
    return {
        'total_gb': round(mem.total / (1024**3), 2),
        'used_gb': round(mem.used / (1024**3), 2),
        'available_gb': round(mem.available / (1024**3), 2),
        'percent_used': mem.percent
    }


def print_memory_status(label: str = "") -> None:
    """
    Imprime el estado actual de la memoria RAM.
    
    Args:
        label: Etiqueta descriptiva del momento (ej: "Antes de cargar datos")
    """
    mem = get_memory_usage()
    prefix = f"[{label}] " if label else ""
    print(f"💾 {prefix}RAM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB "
          f"({mem['percent_used']}%) | Disponible: {mem['available_gb']:.1f} GB")


def clear_memory(*variables) -> None:
    """
    Libera memoria eliminando variables y forzando la recolección de basura.
    
    Args:
        *variables: Variables a eliminar (ya eliminadas externamente con del)
    """
    gc.collect()


def check_memory_threshold(min_available_gb: float = 2.0) -> bool:
    """
    Verifica que haya suficiente memoria RAM disponible.
    
    Args:
        min_available_gb: Mínimo de GB disponibles requeridos
        
    Returns:
        True si hay suficiente memoria
        
    Raises:
        MemoryError si no hay suficiente memoria
    """
    mem = get_memory_usage()
    if mem['available_gb'] < min_available_gb:
        raise MemoryError(
            f"⚠️ Memoria insuficiente: {mem['available_gb']:.1f} GB disponibles, "
            f"se requieren al menos {min_available_gb} GB.\n"
            f"Cierra aplicaciones y reintenta."
        )
    return True


class ProcessingTimer:
    """
    Contexto para medir tiempos de procesamiento.
    
    Uso:
        with ProcessingTimer("Carga de datos"):
            # ... código pesado ...
    """
    
    def __init__(self, task_name: str, log_file: Path = None):
        self.task_name = task_name
        self.log_file = log_file
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  Iniciando: {self.task_name}")
        print_memory_status("Inicio")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if duration < 60:
            time_str = f"{duration:.1f} segundos"
        elif duration < 3600:
            time_str = f"{duration/60:.1f} minutos"
        else:
            time_str = f"{duration/3600:.1f} horas"
        
        status = "✅" if exc_type is None else "❌"
        print(f"{status} Completado: {self.task_name} en {time_str}")
        print_memory_status("Fin")
        
        # Registrar en log si se proporcionó ruta
        if self.log_file:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {status} {self.task_name}: {time_str}\n")
        
        return False  # No suprimir excepciones
