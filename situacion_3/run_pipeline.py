#!/usr/bin/env python3
"""
run_pipeline.py — Ejecutor Principal del Pipeline
===================================================
Situación 3: Portafolio Hipotecario (Freddie Mac)

Ejecuta secuencialmente las fases del pipeline analítico.
Cada fase es un script independiente que puede ejecutarse por separado.

Uso:
  python run_pipeline.py              # Pipeline local (1 máquina)
  python run_pipeline.py --distributed # Pipeline distribuido (Ray cluster)
  python run_pipeline.py --phase 0    # Ejecutar solo una fase específica
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Directorio de scripts fuente
SRC_DIR = Path(__file__).parent / "src"

# Fases del pipeline en orden secuencial
PIPELINE_PHASES = [
    {
        "phase": 0,
        "script": "00_test_headers.py",
        "name": "Prueba de Lectura de Encabezados",
        "description": "Lee encabezados de los 101 archivos y verifica consistencia",
    },
    {
        "phase": 1,
        "script": "01_construccion_panel.py",
        "name": "Construcción del Panel Analítico",
        "description": "Integra datos longitudinales y exporta en Parquet",
    },
    {
        "phase": 2,
        "script": "02_analisis_latente.py",
        "name": "Extracción de Componentes Latentes",
        "description": "AFE/AFC y reducción dimensional",
    },
    {
        "phase": 3,
        "script": "03_deep_learning.py",
        "name": "Deep Learning — Autoencoders (VAE)",
        "description": "Generación de embeddings de riesgo",
    },
    {
        "phase": 4,
        "script": "04_clustering.py",
        "name": "Segmentación (Clustering)",
        "description": "K-Means, GMM y clustering jerárquico",
    },
    {
        "phase": 5,
        "script": "05_perfilado_riesgo.py",
        "name": "Caracterización de Perfiles de Riesgo",
        "description": "Perfilado financiero e interpretación de clusters",
    },
]


def run_phase(phase_info: dict) -> bool:
    """
    Ejecuta una fase del pipeline.

    Args:
        phase_info: Dict con script, name, description

    Returns:
        True si la fase se completó exitosamente
    """
    script_path = SRC_DIR / phase_info["script"]

    if not script_path.exists():
        print(f"  ⏭️  Fase {phase_info['phase']}: {phase_info['script']} — AÚN NO IMPLEMENTADO, saltando...")
        return True  # No es un error, simplemente aún no existe

    print(f"\n{'=' * 70}")
    print(f"  FASE {phase_info['phase']}: {phase_info['name']}")
    print(f"  {phase_info['description']}")
    print(f"  Script: {phase_info['script']}")
    print(f"{'=' * 70}\n")

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SRC_DIR.parent),
    )

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  ✅ Fase {phase_info['phase']} completada en {minutes}m {seconds}s")
        return True
    else:
        print(f"\n  ❌ Fase {phase_info['phase']} FALLÓ (código: {result.returncode}) en {minutes}m {seconds}s")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Analítico — Situación 3: Portafolio Hipotecario"
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Ejecutar solo una fase específica (0-5). Default: todas.",
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Habilitar modo distribuido con Ray cluster.",
    )
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="Iniciar desde una fase específica (0-5). Default: 0.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  PIPELINE ANALÍTICO — SITUACIÓN 3")
    print("  Dataset: Freddie Mac Loan Performance")
    print(f"  Modo: {'DISTRIBUIDO (Ray)' if args.distributed else 'LOCAL'}")
    print("=" * 70)

    if args.distributed:
        print("\n  ℹ️ Modo distribuido: asegúrese de que Ray esté corriendo en ambos nodos.")
        print("    Worker: ray start --address='0.0.0.0:6379' --num-cpus=4 --block")
        print("    Master: ray start --head --port=6379 --num-cpus=8")

    # Seleccionar fases a ejecutar
    if args.phase is not None:
        phases = [p for p in PIPELINE_PHASES if p["phase"] == args.phase]
        if not phases:
            print(f"\n  ❌ Fase {args.phase} no existe. Fases disponibles: 0-5")
            sys.exit(1)
    else:
        phases = [p for p in PIPELINE_PHASES if p["phase"] >= args.start_from]

    # Ejecutar fases
    total_start = time.time()
    failed = []

    for phase_info in phases:
        success = run_phase(phase_info)
        if not success:
            failed.append(phase_info["phase"])
            print(f"\n  ⚠️ Fase {phase_info['phase']} falló. ¿Continuar con la siguiente? (s/n)")
            # En modo no interactivo, continuar
            break

    # Resumen final
    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETADO")
    print(f"  Tiempo total: {total_min}m {total_sec}s")
    print(f"  Fases ejecutadas: {len(phases)}")
    print(f"  Fases fallidas: {len(failed)} {failed if failed else ''}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
