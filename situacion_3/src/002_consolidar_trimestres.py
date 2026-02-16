#!/usr/bin/env python3
"""
002 — Consolidar Parquets por Trimestre
========================================
Script temporal para:
1. Renombrar columnas de Parquets existentes (esquema antiguo → nuevo).
2. Fusionar archivos _partXX.parquet en un solo archivo por trimestre.
3. Validar integridad de los archivos consolidados.

Ejecutar después de haber procesado los trimestres con 001_csv_to_parquet_parts.py
y antes de ejecutar el EDA u otros análisis.

Uso:
    python 002_consolidar_trimestres.py
"""

import sys
import os
import gc
import glob
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PANEL_PATH, DATA_PROCESSED_PATH, OLD_TO_NEW_COLUMN_MAP,
    PERFORMANCE_COLUMNS, COLUMNS_TO_DROP,
    PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE, PARQUET_MAX_FILE_SIZE_GB,
)
from utils.memory_utils import (
    print_system_summary, print_memory_status,
    ProcessingTimer, clear_memory, check_memory_threshold,
)

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================================
# Funciones auxiliares
# ============================================================================

def get_trimestre_groups() -> dict[str, list[Path]]:
    """
    Agrupa los archivos Parquet por trimestre.
    Retorna dict: trimestre → lista de archivos (incluye parts y no-parts).
    Ejemplo: {'2000Q1': [Path('2000Q1.parquet'), Path('2000Q1_part01.parquet'), ...]}
    """
    all_files = sorted(PANEL_PATH.glob("*.parquet"))
    groups: dict[str, list[Path]] = defaultdict(list)

    for f in all_files:
        # Extraer trimestre del nombre: 2000Q1.parquet o 2000Q1_part01.parquet
        match = re.match(r"^(\d{4}Q\d)(?:_part\d+)?\.parquet$", f.name)
        if match:
            trimestre = match.group(1)
            groups[trimestre].append(f)

    return dict(sorted(groups.items()))


def detect_schema_type(columns: list[str]) -> str:
    """
    Detecta si un Parquet usa el esquema antiguo o nuevo.
    Retorna 'old', 'new', o 'unknown'.
    """
    old_markers = {"loan_sequence_number", "current_deferred_upb", "original_ltv", "col_61"}
    new_markers = {"loan_identifier", "channel", "original_loan_to_value_ratio_ltv"}
    col_set = set(columns)

    if old_markers & col_set:
        return "old"
    elif new_markers & col_set:
        return "new"
    else:
        return "unknown"


def rename_columns_table(table: pa.Table) -> pa.Table:
    """
    Renombra columnas de un Table de PyArrow usando OLD_TO_NEW_COLUMN_MAP.
    Solo renombra si el esquema es antiguo.
    """
    columns = table.column_names
    schema_type = detect_schema_type(columns)

    if schema_type == "new":
        return table  # Ya tiene nombres correctos

    new_names = []
    for col in columns:
        new_name = OLD_TO_NEW_COLUMN_MAP.get(col, col)
        new_names.append(new_name)

    return table.rename_columns(new_names)


def consolidate_trimestre(
    trimestre: str,
    files: list[Path],
    output_dir: Path,
    backup_dir: Path | None = None,
) -> dict:
    """
    Consolida todos los archivos Parquet de un trimestre en uno o más
    archivos finales con columnas renombradas.

    Retorna dict con estadísticas del proceso.
    """
    stats = {
        "trimestre": trimestre,
        "input_files": len(files),
        "input_rows": 0,
        "output_rows": 0,
        "schema_type": "unknown",
        "renamed": False,
        "consolidated": False,
        "error": None,
    }

    try:
        # -------------------------------------------------------------------
        # Verificar memoria antes de procesar
        # -------------------------------------------------------------------
        check_memory_threshold(min_available_gb=2.0)

        # -------------------------------------------------------------------
        # Leer todos los archivos del trimestre
        # -------------------------------------------------------------------
        tables = []
        for f in sorted(files):
            try:
                t = pq.read_table(f)
                tables.append(t)
                stats["input_rows"] += len(t)
            except Exception as e:
                stats["error"] = f"Error leyendo {f.name}: {e}"
                return stats

        if not tables:
            stats["error"] = "Sin archivos válidos"
            return stats

        # Detectar esquema
        stats["schema_type"] = detect_schema_type(tables[0].column_names)

        # -------------------------------------------------------------------
        # Concatenar si hay múltiples partes
        # -------------------------------------------------------------------
        if len(tables) == 1:
            combined = tables[0]
        else:
            combined = pa.concat_tables(tables, promote=True)
            stats["consolidated"] = True

        del tables
        gc.collect()

        # -------------------------------------------------------------------
        # Renombrar columnas si es esquema antiguo
        # -------------------------------------------------------------------
        if stats["schema_type"] == "old":
            combined = rename_columns_table(combined)
            stats["renamed"] = True

        stats["output_rows"] = len(combined)

        # -------------------------------------------------------------------
        # Escribir archivo(s) de salida
        # -------------------------------------------------------------------
        output_base = output_dir / f"{trimestre}.parquet"

        # Estimar tamaño: si es muy grande, dividir en partes
        estimated_size_gb = combined.nbytes / (1024**3)

        if estimated_size_gb > PARQUET_MAX_FILE_SIZE_GB:
            # Dividir en partes
            total_rows = len(combined)
            rows_per_part = int(total_rows * (PARQUET_MAX_FILE_SIZE_GB / estimated_size_gb) * 0.9)
            part_num = 0

            for start in range(0, total_rows, rows_per_part):
                end = min(start + rows_per_part, total_rows)
                part_table = combined.slice(start, end - start)
                part_path = output_dir / f"{trimestre}_part{part_num:02d}.parquet"

                pq.write_table(
                    part_table,
                    part_path,
                    compression=PARQUET_COMPRESSION,
                    compression_level=PARQUET_COMPRESSION_LEVEL,
                    row_group_size=PARQUET_ROW_GROUP_SIZE,
                    write_statistics=True,
                )
                part_num += 1
                del part_table
                gc.collect()
        else:
            pq.write_table(
                combined,
                output_base,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                row_group_size=PARQUET_ROW_GROUP_SIZE,
                write_statistics=True,
            )

        del combined
        gc.collect()

        # -------------------------------------------------------------------
        # Mover originales a backup (si se proporcionó backup_dir)
        # -------------------------------------------------------------------
        if backup_dir and (stats["renamed"] or stats["consolidated"]):
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                backup_path = backup_dir / f.name
                if f.exists() and not backup_path.exists():
                    shutil.move(str(f), str(backup_path))

    except Exception as e:
        stats["error"] = str(e)

    return stats


def validate_consolidated(output_dir: Path) -> dict:
    """
    Valida todos los Parquets consolidados en output_dir.
    Retorna diccionario con estadísticas.
    """
    files = sorted(output_dir.glob("*.parquet"))
    valid = 0
    corrupt = 0
    total_rows = 0
    schema_issues = []

    expected_cols = [c for c in PERFORMANCE_COLUMNS if c not in COLUMNS_TO_DROP]

    for f in tqdm(files, desc="Validando consolidados"):
        try:
            schema = pq.read_schema(f)
            pf = pq.read_metadata(f)

            # Verificar columnas
            if set(schema.names) != set(expected_cols):
                missing = set(expected_cols) - set(schema.names)
                extra = set(schema.names) - set(expected_cols)
                if missing or extra:
                    schema_issues.append({
                        "file": f.name,
                        "missing": missing,
                        "extra": extra,
                    })

            total_rows += pf.num_rows
            valid += 1
        except Exception as e:
            corrupt += 1
            print(f"  ✗ CORRUPTO: {f.name} — {e}")

    return {
        "valid": valid,
        "corrupt": corrupt,
        "total_rows": total_rows,
        "total_files": len(files),
        "schema_issues": schema_issues,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada principal."""
    print_system_summary()

    with ProcessingTimer("Consolidación de trimestres"):
        # ------------------------------------------------------------------
        # Paso 1: Obtener grupos de trimestres
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PASO 1: Inventario de archivos Parquet")
        print("=" * 70)

        groups = get_trimestre_groups()
        total_files = sum(len(v) for v in groups.values())

        print(f"  Trimestres encontrados: {len(groups)}")
        print(f"  Archivos Parquet totales: {total_files}")

        # Clasificar
        need_consolidation = {k: v for k, v in groups.items() if len(v) > 1}
        single_files = {k: v for k, v in groups.items() if len(v) == 1}

        print(f"  Con múltiples partes (a consolidar): {len(need_consolidation)}")
        print(f"  Archivo único (solo renombrar cols): {len(single_files)}")

        # Verificar esquemas
        sample_file = next(iter(groups.values()))[0]
        sample_schema = pq.read_schema(sample_file)
        schema_type = detect_schema_type(sample_schema.names)
        print(f"  Esquema detectado: {schema_type}")

        if schema_type == "new" and not need_consolidation:
            print("\n  ✓ Todos los archivos ya tienen el esquema correcto y están consolidados.")
            print("    No hay nada que hacer.")
            return

        # ------------------------------------------------------------------
        # Paso 2: Crear directorio temporal de salida
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PASO 2: Procesando trimestres")
        print("=" * 70)

        # Usamos un directorio temporal para escritura segura
        temp_output = DATA_PROCESSED_PATH / "panel_analitico_consolidated"
        temp_output.mkdir(parents=True, exist_ok=True)

        backup_dir = DATA_PROCESSED_PATH / "panel_analitico_backup"

        results = []
        errors = []

        for trimestre, files in tqdm(groups.items(), desc="Consolidando"):
            # Verificar si ya existe el consolidado
            expected_output = temp_output / f"{trimestre}.parquet"
            if expected_output.exists():
                # Ya procesado, verificar
                try:
                    schema = pq.read_schema(expected_output)
                    if detect_schema_type(schema.names) == "new":
                        results.append({
                            "trimestre": trimestre,
                            "input_files": len(files),
                            "schema_type": "new",
                            "renamed": False,
                            "consolidated": False,
                            "error": None,
                            "input_rows": 0,
                            "output_rows": pq.read_metadata(expected_output).num_rows,
                        })
                        continue
                except Exception:
                    pass  # Re-procesar

            stats = consolidate_trimestre(
                trimestre=trimestre,
                files=files,
                output_dir=temp_output,
                backup_dir=backup_dir,
            )
            results.append(stats)

            if stats["error"]:
                errors.append(stats)
                print(f"\n  ✗ ERROR en {trimestre}: {stats['error']}")

            # Liberar memoria periódicamente
            if len(results) % 10 == 0:
                gc.collect()

        # ------------------------------------------------------------------
        # Paso 3: Validación
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PASO 3: Validación de archivos consolidados")
        print("=" * 70)

        validation = validate_consolidated(temp_output)

        print(f"\n  Archivos válidos: {validation['valid']}")
        print(f"  Archivos corruptos: {validation['corrupt']}")
        print(f"  Total filas: {validation['total_rows']:,}")
        print(f"  Problemas de esquema: {len(validation['schema_issues'])}")

        if validation["schema_issues"]:
            for issue in validation["schema_issues"][:5]:
                print(f"    {issue['file']}: missing={issue['missing']}, extra={issue['extra']}")

        # ------------------------------------------------------------------
        # Paso 4: Reemplazar directorio original
        # ------------------------------------------------------------------
        if not errors and validation["corrupt"] == 0:
            print("\n" + "=" * 70)
            print("PASO 4: Reemplazando directorio original")
            print("=" * 70)

            # Mover originales restantes a backup
            remaining_originals = list(PANEL_PATH.glob("*.parquet"))
            if remaining_originals:
                backup_dir.mkdir(parents=True, exist_ok=True)
                for f in remaining_originals:
                    backup_path = backup_dir / f.name
                    if not backup_path.exists():
                        shutil.move(str(f), str(backup_path))

            # Mover consolidados al directorio original
            for f in temp_output.glob("*.parquet"):
                shutil.move(str(f), str(PANEL_PATH / f.name))

            # Eliminar directorio temporal
            if temp_output.exists():
                shutil.rmtree(temp_output)

            print(f"  ✓ {validation['valid']} archivos consolidados movidos a {PANEL_PATH}")
            print(f"  ✓ Originales respaldados en {backup_dir}")

            # Verificación final
            final_count = len(list(PANEL_PATH.glob("*.parquet")))
            print(f"  ✓ Archivos finales en panel_analitico/: {final_count}")

        else:
            print("\n  ⚠ Hubo errores. Los archivos consolidados están en:")
            print(f"    {temp_output}")
            print("  Los originales NO fueron modificados.")

            if errors:
                print(f"\n  Trimestres con error ({len(errors)}):")
                for e in errors:
                    print(f"    {e['trimestre']}: {e['error']}")

        # ------------------------------------------------------------------
        # Resumen
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("RESUMEN DE CONSOLIDACIÓN")
        print("=" * 70)

        renamed_count = sum(1 for r in results if r.get("renamed"))
        consolidated_count = sum(1 for r in results if r.get("consolidated"))
        total_input_rows = sum(r.get("input_rows", 0) for r in results)
        total_output_rows = sum(r.get("output_rows", 0) for r in results)

        print(f"  Trimestres procesados: {len(results)}")
        print(f"  Columnas renombradas: {renamed_count} trimestres")
        print(f"  Partes consolidadas: {consolidated_count} trimestres")
        print(f"  Filas entrada: {total_input_rows:,}")
        print(f"  Filas salida: {total_output_rows:,}")
        print(f"  Errores: {len(errors)}")

    print_memory_status("Final")


if __name__ == "__main__":
    main()
