#!/usr/bin/env python3
"""
01_data_loading.py - Carga y Aplanamiento de Imágenes Multiespectrales
======================================================================
FASE 1 - Paso 1.1, 1.2, 1.3

Lee las 27,000 imágenes .tif de EuroSAT (64x64x13 bandas Sentinel-2),
las aplana a vectores de 53,248 elementos y guarda la matriz resultante.

Estrategia de memoria:
- Procesa en lotes de BATCH_SIZE imágenes
- Usa float32 (no float64) para reducir uso de RAM a la mitad
- Guarda incrementalmente en disco con numpy memmap
- Libera memoria después de cada lote

Input:  dataset/EuroSAT_MS/[10 clases]/*.tif
Output: data_processed/features_raw_flattened.npy  (~5.3 GB en float32)
        data_processed/metadata_labels.csv
        
⚠️ Ejecutar SOLO este script. Esperar a que termine completamente.
⚠️ NO ejecutar otros scripts en paralelo.
"""

import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Agregar src/ al path para importar módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_PATH, DATA_PROCESSED_PATH, CHECKPOINTS_PATH,
    CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS, TOTAL_FEATURES,
    BATCH_SIZE, FILE_METADATA, FILE_FEATURES_RAW, FILE_PROCESSING_LOG,
    create_checkpoint
)
from utils.image_loader import get_all_image_paths, load_and_flatten_batch, extract_metadata
from utils.memory_utils import print_memory_status, clear_memory, ProcessingTimer, check_memory_threshold


def main():
    """Pipeline de carga y aplanamiento de imágenes."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 1: CARGA Y APLANAMIENTO DE IMÁGENES MULTIESPECTRALES")
    print("=" * 70)
    
    # ====================================================================
    # 1. Verificar recursos y dataset
    # ====================================================================
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=2.0)
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"❌ Dataset no encontrado en: {DATASET_PATH}\n"
            f"   Descarga y descomprime EuroSAT_MS primero."
        )
    
    # ====================================================================
    # 2. Recolectar rutas de imágenes y etiquetas
    # ====================================================================
    with ProcessingTimer("Recolección de rutas de imágenes", log_file):
        file_paths, labels = get_all_image_paths(DATASET_PATH, CLASS_NAMES)
        n_images = len(file_paths)
        print(f"\n📊 Total imágenes: {n_images}")
        print(f"📊 Clases: {len(set(labels))}")
        print(f"📊 Features por imagen: {TOTAL_FEATURES:,} (={IMG_HEIGHT}x{IMG_WIDTH}x{N_CHANNELS})")
    
    # ====================================================================
    # 3. Guardar metadatos (etiquetas reales para validación posterior)
    # ====================================================================
    with ProcessingTimer("Guardado de metadatos", log_file):
        metadata = extract_metadata(file_paths, labels)
        df_metadata = pd.DataFrame(metadata)
        metadata_path = DATA_PROCESSED_PATH / FILE_METADATA
        df_metadata.to_csv(metadata_path, index=False)
        print(f"💾 Metadatos guardados: {metadata_path}")
        print(f"   Dimensiones: {df_metadata.shape}")
    
    # ====================================================================
    # 4. Carga por lotes y aplanamiento
    # ====================================================================
    output_path = DATA_PROCESSED_PATH / FILE_FEATURES_RAW
    
    with ProcessingTimer("Carga y aplanamiento de imágenes (por lotes)", log_file):
        
        # Crear archivo memmap en disco para escritura incremental
        # Esto evita tener toda la matriz en RAM a la vez
        print(f"\n📦 Creando matriz memmap en disco: ({n_images}, {TOTAL_FEATURES})")
        print(f"   Tipo: float32 | Tamaño estimado: {n_images * TOTAL_FEATURES * 4 / (1024**3):.2f} GB")
        
        features_mmap = np.lib.format.open_memmap(
            str(output_path),
            mode='w+',
            dtype=np.float32,
            shape=(n_images, TOTAL_FEATURES)
        )
        
        # Procesar por lotes
        n_batches = (n_images + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n🔄 Procesando {n_batches} lotes de ~{BATCH_SIZE} imágenes cada uno...")
        
        for batch_idx in tqdm(range(n_batches), desc="Lotes"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_images)
            batch_paths = file_paths[start_idx:end_idx]
            
            # Cargar y aplanar este lote
            batch_data = load_and_flatten_batch(batch_paths)
            
            # Escribir en el memmap (directo a disco)
            features_mmap[start_idx:end_idx] = batch_data
            
            # Liberar memoria del lote
            del batch_data
            gc.collect()
        
        # Flush al disco
        del features_mmap
        gc.collect()
        
        print(f"\n💾 Matriz guardada: {output_path}")
        print(f"   Tamaño en disco: {output_path.stat().st_size / (1024**3):.2f} GB")
    
    # ====================================================================
    # 5. Verificación rápida de integridad
    # ====================================================================
    with ProcessingTimer("Verificación de integridad", log_file):
        # Cargar solo las primeras filas para verificar
        features_check = np.load(str(output_path), mmap_mode='r')
        print(f"✅ Shape verificado: {features_check.shape}")
        print(f"✅ Dtype: {features_check.dtype}")
        print(f"✅ Rango de valores: [{features_check[0].min():.2f}, {features_check[0].max():.2f}]")
        print(f"✅ ¿Contiene NaN? {np.isnan(features_check[0]).any()}")
        del features_check
        gc.collect()
    
    # ====================================================================
    # 6. Crear checkpoint y liberar memoria
    # ====================================================================
    print_memory_status("Fin del script")
    create_checkpoint('data_loading', 
                      f"n_images={n_images}, n_features={TOTAL_FEATURES}, dtype=float32")
    
    print("\n" + "=" * 70)
    print("  ✅ PASO 1 COMPLETADO - Datos crudos aplanados y guardados")
    print("  📌 Siguiente: Ejecutar 02_normalization.py")
    print("  ⚠️  Esperar a que este script termine antes de continuar")
    print("=" * 70)


if __name__ == "__main__":
    main()
