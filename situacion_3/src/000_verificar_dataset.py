#!/usr/bin/env python3
"""
00_test_headers.py — Prueba de Lectura de Encabezados
======================================================
Situación 3: Portafolio Hipotecario

Lee la primera fila y los encabezados de los 101 archivos CSV
dentro del ZIP para verificar la estructura del dataset.

Genera:
  - Tabla resumen en terminal (con barra de progreso)
  - outputs/tables/file_inventory.csv
  - outputs/tables/column_consistency_check.csv
  - outputs/figures/00_exploratorio/file_sizes_inventory.png

Uso:
  cd situacion_3/
  python src/00_test_headers.py
"""

import sys
import os
from pathlib import Path

# Agregar src/ al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ZIP_FILE_PATH,
    PERFORMANCE_COLUMNS,
    TABLES_PATH,
    FIGURES_SUBDIRS,
    TABLE_NAMES,
)
from utils.memory_utils import (
    print_system_summary,
    print_memory_status,
    ProcessingTimer,
)
from utils.data_loader import ZipDataLoader
from utils.plotting_utils import plot_file_inventory, save_figure
from utils.logging_utils import (
    setup_logger,
    get_logger,
    log_memory_snapshot,
    log_exception,
)


def main():
    """Punto de entrada principal."""
    # ── Configurar logging ──
    logger = setup_logger("00_test_headers")

    print("=" * 70)
    print("  FASE 0: PRUEBA DE LECTURA DE ENCABEZADOS")
    print("  Dataset: Freddie Mac Performance Data (101 CSVs)")
    print("=" * 70)

    # Resumen del sistema
    print_system_summary()

    # Verificar que el ZIP existe
    logger.info("Verificando archivo ZIP: %s", ZIP_FILE_PATH)
    print(f"\n  📦 Archivo ZIP: {ZIP_FILE_PATH}")
    print(f"     Existe: {ZIP_FILE_PATH.exists()}")

    if not ZIP_FILE_PATH.exists():
        logger.error("El archivo ZIP no existe: %s", ZIP_FILE_PATH)
        print("\n  ❌ ERROR: El archivo ZIP no existe.")
        print("     Verifica que el dataset esté montado vía NFS o copiado localmente.")
        print(f"     Ruta esperada: {ZIP_FILE_PATH}")
        sys.exit(1)

    # Tamaño del ZIP
    zip_size_gb = ZIP_FILE_PATH.stat().st_size / (1024**3)
    logger.info("ZIP encontrado: %.2f GB", zip_size_gb)
    print(f"     Tamaño: {zip_size_gb:.2f} GB")

    with ProcessingTimer("Lectura completa de encabezados"):
      try:
        with ZipDataLoader() as loader:

            # ============================================================
            # 1. INVENTARIO DE ARCHIVOS
            # ============================================================
            print("\n" + "=" * 70)
            print("  1. INVENTARIO DE ARCHIVOS EN EL ZIP")
            print("=" * 70)

            loader.print_file_inventory()
            inventory = loader.get_file_inventory()
            logger.info("Inventario: %d archivos encontrados", len(inventory))

            # Guardar inventario como CSV
            import pandas as pd
            df_inventory = pd.DataFrame(inventory)
            inventory_path = TABLES_PATH / TABLE_NAMES["file_inventory"]
            df_inventory.to_csv(inventory_path, index=False)
            logger.info("Inventario guardado: %s", inventory_path)
            print(f"\n  💾 Inventario guardado: {inventory_path.name}")

            # ============================================================
            # 2. VERIFICACIÓN DE CONSISTENCIA DE COLUMNAS
            # ============================================================
            print("\n" + "=" * 70)
            print("  2. VERIFICACIÓN DE CONSISTENCIA DE COLUMNAS")
            print("=" * 70)
            print(f"  Columnas esperadas: {len(PERFORMANCE_COLUMNS)}")
            print(f"  Columnas definidas en config.py:")
            print(f"    Primeras 5: {PERFORMANCE_COLUMNS[1:6]}")
            print(f"    Últimas 5:  {PERFORMANCE_COLUMNS[-5:]}")

            print_memory_status("Antes de leer encabezados")
            log_memory_snapshot("antes_de_leer_encabezados")

            df_headers = loader.read_all_headers()

            # Guardar resultados
            consistency_path = TABLES_PATH / TABLE_NAMES["column_consistency"]
            df_headers.to_csv(consistency_path, index=False)
            logger.info("Consistencia guardada: %s", consistency_path)
            print(f"\n  💾 Consistencia guardada: {consistency_path.name}")

            # Estadísticas
            consistent_count = df_headers["consistent"].sum()
            total_count = len(df_headers)
            error_count = (df_headers["num_columns"] == -1).sum()

            logger.info(
                "Consistencia: %d/%d archivos OK, %d errores",
                consistent_count, total_count, error_count,
            )

            print(f"\n  📊 Resultados:")
            print(f"     Archivos consistentes: {consistent_count}/{total_count}")
            print(f"     Archivos con error:    {error_count}/{total_count}")

            if consistent_count == total_count:
                logger.info("TODOS los archivos tienen la estructura esperada")
                print(f"\n  ✅ TODOS los archivos tienen la estructura esperada.")
            else:
                logger.warning(
                    "HAY archivos con estructura diferente: %d/%d inconsistentes",
                    total_count - consistent_count, total_count,
                )
                print(f"\n  ⚠️ HAY archivos con estructura diferente. Revisar manualmente.")

            # ============================================================
            # 3. MUESTRA DE DATOS (primera fila del primer archivo)
            # ============================================================
            print("\n" + "=" * 70)
            print("  3. MUESTRA DE DATOS — Primera fila del primer archivo")
            print("=" * 70)

            filenames = loader.get_filenames()
            first_file = filenames[0]
            ncols, sample_row = loader.read_first_row(first_file)
            logger.info("Muestra: %s (%d columnas)", first_file, ncols)

            print(f"\n  Archivo: {first_file}")
            print(f"  Columnas: {ncols}")
            print(f"\n  Datos de la primera fila:")

            # Mostrar columnas con sus valores (solo las primeras 35)
            for i, col in enumerate(PERFORMANCE_COLUMNS[:35]):
                if col == "_empty":
                    continue
                val = sample_row.iloc[0].get(col, "N/A")
                print(f"    [{i:>3}] {col:<45} = {val}")

            print(f"\n    ... y {ncols - 35} columnas más (campos extendidos)")

            # ============================================================
            # 4. GRÁFICO DE INVENTARIO
            # ============================================================
            print("\n" + "=" * 70)
            print("  4. GENERANDO GRÁFICO DE INVENTARIO")
            print("=" * 70)

            fig_path = FIGURES_SUBDIRS["exploratorio"] / "file_sizes_inventory.png"
            plot_file_inventory(
                filenames=[f["filename"] for f in inventory],
                sizes_gb=[f["uncompressed_gb"] for f in inventory],
                title="Inventario de Archivos — Performance.zip (Freddie Mac)",
                filepath=fig_path,
            )
            logger.info("Gráfico guardado: %s", fig_path)

      except Exception:
        log_exception("00_test_headers — error durante procesamiento")
        raise

    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print("\n" + "=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    print(f"  Archivos en el ZIP:    {total_count}")
    print(f"  Consistencia:          {consistent_count}/{total_count} ✅")
    print(f"  Columnas por archivo:  {len(PERFORMANCE_COLUMNS)}")
    print(f"  Tamaño comprimido:     {zip_size_gb:.2f} GB")
    total_uncomp = sum(f["uncompressed_gb"] for f in inventory)
    print(f"  Tamaño descomprimido:  {total_uncomp:.2f} GB")
    print(f"\n  Archivos generados:")
    print(f"    📄 {inventory_path}")
    print(f"    📄 {consistency_path}")
    print(f"    📊 {fig_path}")
    print_memory_status("Final del script")
    logger.info("Fase 00 completada exitosamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
