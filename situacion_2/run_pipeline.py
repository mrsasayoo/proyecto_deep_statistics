#!/usr/bin/env python3
"""
SCRIPT MASTER - Ejecución Secuencial del Pipeline Completo
===========================================================

⚠️ IMPORTANTE: Este script ejecuta SECUENCIALMENTE todos los pasos del proyecto.
NO ejecutar scripts individuales mientras este script está corriendo.

Uso:
    python run_pipeline.py [--step STEP_NUMBER] [--skip-existing]

Opciones:
    --step STEP_NUMBER    Ejecutar desde un paso específico (1-7)
    --skip-existing       Saltar pasos ya completados (con checkpoint)
    --dry-run            Mostrar qué se ejecutaría sin ejecutar

Ejemplo:
    python run_pipeline.py                    # Ejecutar todo
    python run_pipeline.py --step 3           # Ejecutar desde paso 3
    python run_pipeline.py --skip-existing    # Continuar desde donde quedó
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from config import CHECKPOINTS, CHECKPOINTS_PATH, validate_checkpoint, create_checkpoint

# ============================================================================
# DEFINICIÓN DEL PIPELINE
# ============================================================================

PIPELINE_STEPS = [
    {
        'number': 1,
        'name': 'data_loading',
        'script': 'src/01_data_loading.py',
        'description': 'Carga y aplanamiento de imágenes multiespectrales',
        'estimated_time': '15-30 min'
    },
    {
        'number': 2,
        'name': 'normalization',
        'script': 'src/02_normalization.py',
        'description': 'Normalización de features con StandardScaler',
        'estimated_time': '5-10 min'
    },
    {
        'number': 3,
        'name': 'pca_reduction',
        'script': 'src/03_pca_reduction.py',
        'description': 'Reducción dimensional con PCA',
        'estimated_time': '10-20 min'
    },
    {
        'number': 4,
        'name': 'kmeans',
        'script': 'src/04_kmeans_clustering.py',
        'description': 'Modelado K-Means con optimización de k',
        'estimated_time': '20-40 min'
    },
    {
        'number': 5,
        'name': 'dbscan',
        'script': 'src/05_dbscan_clustering.py',
        'description': 'Modelado DBSCAN con búsqueda de hiperparámetros',
        'estimated_time': '15-30 min'
    },
    {
        'number': 6,
        'name': 'evaluation',
        'script': 'src/06_evaluation_validation.py',
        'description': 'Evaluación y validación (ARI, NMI, Silueta)',
        'estimated_time': '5-10 min'
    },
    {
        'number': 7,
        'name': 'visualization',
        'script': 'src/07_visualization_export.py',
        'description': 'Generación de todas las figuras',
        'estimated_time': '10-20 min'
    }
]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def print_header(text, char='='):
    """Imprime un encabezado formateado"""
    width = 80
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)
    print()


def print_step_info(step):
    """Imprime información de un paso"""
    print(f"📍 PASO {step['number']}/{len(PIPELINE_STEPS)}: {step['name'].upper()}")
    print(f"   Descripción: {step['description']}")
    print(f"   Script: {step['script']}")
    print(f"   Tiempo estimado: {step['estimated_time']}")
    print()


def is_checkpoint_complete(checkpoint_name):
    """Verifica si un checkpoint existe"""
    checkpoint_file = CHECKPOINTS_PATH / CHECKPOINTS[checkpoint_name]
    return checkpoint_file.exists()


def run_script(script_path):
    """
    Ejecuta un script de Python y captura su output.
    
    Returns:
        tuple: (success: bool, duration: float)
    """
    start_time = time.time()
    
    try:
        print(f"⏳ Ejecutando: {script_path}")
        print("-" * 80)
        
        # Ejecutar script
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,  # Mostrar output en tiempo real
            text=True
        )
        
        duration = time.time() - start_time
        print("-" * 80)
        print(f"✅ Completado exitosamente en {duration:.2f} segundos")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print("-" * 80)
        print(f"❌ ERROR después de {duration:.2f} segundos")
        print(f"Código de salida: {e.returncode}")
        return False, duration


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline secuencial del Proyecto Situación 2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--step', 
        type=int, 
        choices=range(1, 8),
        help='Ejecutar desde un paso específico (1-7)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Saltar pasos ya completados'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostrar qué se ejecutaría sin ejecutar'
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # Banner inicial
    # ========================================================================
    print_header("🚀 PIPELINE SECUENCIAL - PROYECTO SITUACIÓN 2")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de pasos: {len(PIPELINE_STEPS)}")
    
    if args.dry_run:
        print("\n⚠️  MODO DRY-RUN: No se ejecutará nada")
    
    print()
    
    # ========================================================================
    # Determinar pasos a ejecutar
    # ========================================================================
    start_step = args.step if args.step else 1
    steps_to_run = [s for s in PIPELINE_STEPS if s['number'] >= start_step]
    
    # Filtrar pasos ya completados si se solicitó
    if args.skip_existing:
        steps_to_run = [
            s for s in steps_to_run 
            if not is_checkpoint_complete(s['name'])
        ]
    
    if not steps_to_run:
        print("✅ Todos los pasos solicitados ya están completados.")
        print("   Usa --step N para reejecutar desde un paso específico.")
        return 0
    
    # ========================================================================
    # Mostrar resumen
    # ========================================================================
    print(f"📋 Se ejecutarán {len(steps_to_run)} pasos:")
    for step in steps_to_run:
        status = "✓ Completado" if is_checkpoint_complete(step['name']) else "⧗ Pendiente"
        print(f"   {step['number']}. {step['name']}: {status}")
    print()
    
    if args.dry_run:
        return 0
    
    # Confirmar ejecución
    if not args.skip_existing:
        response = input("¿Continuar con la ejecución? [S/n]: ").strip().lower()
        if response and response != 's':
            print("❌ Ejecución cancelada por el usuario.")
            return 1
    
    # ========================================================================
    # Ejecutar pipeline
    # ========================================================================
    total_start = time.time()
    completed_steps = 0
    failed_step = None
    
    for step in steps_to_run:
        print_header(f"PASO {step['number']}: {step['name'].upper()}", char='-')
        print_step_info(step)
        
        # Verificar que el script existe
        script_path = Path(step['script'])
        if not script_path.exists():
            print(f"❌ ERROR: Script no encontrado: {script_path}")
            failed_step = step
            break
        
        # Ejecutar script
        success, duration = run_script(script_path)
        
        if not success:
            print(f"\n❌ ERROR: El paso {step['number']} falló.")
            print("   Revisa los errores arriba y corrige el problema.")
            print(f"   Para reintentar: python run_pipeline.py --step {step['number']}")
            failed_step = step
            break
        
        completed_steps += 1
        print(f"\n✅ Paso {step['number']}/{len(PIPELINE_STEPS)} completado")
        
        # Pausar entre pasos para liberar recursos
        if step['number'] < len(PIPELINE_STEPS):
            print("\n⏸️  Pausando 5 segundos para liberar memoria...")
            time.sleep(5)
    
    # ========================================================================
    # Resumen final
    # ========================================================================
    total_duration = time.time() - total_start
    
    print_header("📊 RESUMEN DE EJECUCIÓN")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duración total: {total_duration/60:.2f} minutos")
    print(f"Pasos completados: {completed_steps}/{len(steps_to_run)}")
    
    if failed_step:
        print(f"\n❌ Pipeline interrumpido en el paso {failed_step['number']}: {failed_step['name']}")
        print(f"   Para continuar: python run_pipeline.py --step {failed_step['number']}")
        return 1
    else:
        print("\n✅ Pipeline completado exitosamente!")
        print("\n📦 Próximos pasos:")
        print("   1. Revisar figuras en: outputs/figures/")
        print("   2. Revisar métricas en: outputs/tables/")
        print("   3. Iniciar redacción del informe")
        return 0


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Ejecución interrumpida por el usuario (Ctrl+C)")
        print("   Puedes continuar después con: python run_pipeline.py --skip-existing")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
