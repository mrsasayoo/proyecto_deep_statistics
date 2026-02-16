#!/usr/bin/env python3
"""
02_normalization.py - Normalización de Features con StandardScaler
==================================================================
FASE 1 - Paso 1.4

Normaliza la matriz aplanada usando StandardScaler (media=0, std=1).
Esto es CRÍTICO porque las bandas de Sentinel-2 tienen rangos de
reflectancia muy diferentes (visibles ~0-3000, infrarrojas ~0-10000).
Sin normalización, PCA estaría dominado por las bandas infrarrojas.

Estrategia de memoria:
- Carga la matriz con memmap (read-only, no carga toda en RAM)
- Calcula media y std parcialmente (partial_fit) por lotes
- Aplica la transformación por lotes y guarda en un nuevo .npy
- Persiste el scaler para aplicar en datos nuevos

Input:  data_processed/features_raw_flattened.npy
Output: data_processed/features_normalized.npy
        outputs/models/scaler_model.pkl
        
⚠️ Ejecutar SOLO después de que 01_data_loading.py haya terminado.
"""

import sys
import gc
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_PATH, MODELS_PATH,
    BATCH_SIZE, TOTAL_FEATURES,
    FILE_FEATURES_RAW, FILE_FEATURES_NORMALIZED,
    FILE_SCALER_MODEL, FILE_PROCESSING_LOG,
    validate_checkpoint, create_checkpoint
)
from utils.memory_utils import print_memory_status, ProcessingTimer, check_memory_threshold


def main():
    """Pipeline de normalización."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 2: NORMALIZACIÓN CON STANDARDSCALER")
    print("=" * 70)
    
    # ====================================================================
    # 1. Validar checkpoint anterior
    # ====================================================================
    validate_checkpoint('data_loading')
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=2.0)
    
    # ====================================================================
    # 2. Cargar datos crudos (memmap = lectura lazy desde disco)
    # ====================================================================
    raw_path = DATA_PROCESSED_PATH / FILE_FEATURES_RAW
    
    with ProcessingTimer("Carga de datos crudos (memmap)", log_file):
        features_raw = np.load(str(raw_path), mmap_mode='r')
        n_images = features_raw.shape[0]
        n_features = features_raw.shape[1]
        print(f"📊 Datos cargados (memmap): {features_raw.shape}")
        print(f"   Dtype: {features_raw.dtype}")
    
    # ====================================================================
    # 3. Calcular media y std por lotes (sin cargar todo en RAM)
    # ====================================================================
    from sklearn.preprocessing import StandardScaler
    
    with ProcessingTimer("Cálculo incremental de media y std", log_file):
        scaler = StandardScaler()
        
        n_batches = (n_images + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"🔄 Calculando estadísticas en {n_batches} lotes...")
        
        for batch_idx in tqdm(range(n_batches), desc="partial_fit"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_images)
            batch = np.array(features_raw[start_idx:end_idx], dtype=np.float32)
            scaler.partial_fit(batch)
            del batch
            gc.collect()
        
        print(f"✅ Media calculada: rango [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
        print(f"✅ Std calculada: rango [{scaler.scale_.min():.2f}, {scaler.scale_.max():.2f}]")
    
    # ====================================================================
    # 4. Aplicar transformación por lotes y guardar en nuevo .npy
    # ====================================================================
    output_path = DATA_PROCESSED_PATH / FILE_FEATURES_NORMALIZED
    
    with ProcessingTimer("Normalización por lotes y escritura a disco", log_file):
        
        features_norm = np.lib.format.open_memmap(
            str(output_path),
            mode='w+',
            dtype=np.float32,
            shape=(n_images, n_features)
        )
        
        for batch_idx in tqdm(range(n_batches), desc="transform"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_images)
            batch = np.array(features_raw[start_idx:end_idx], dtype=np.float32)
            
            # Normalizar este lote
            batch_normalized = scaler.transform(batch)
            features_norm[start_idx:end_idx] = batch_normalized.astype(np.float32)
            
            del batch, batch_normalized
            gc.collect()
        
        del features_norm, features_raw
        gc.collect()
        
        print(f"\n💾 Datos normalizados guardados: {output_path}")
        print(f"   Tamaño: {output_path.stat().st_size / (1024**3):.2f} GB")
    
    # ====================================================================
    # 5. Guardar el scaler para reproducibilidad
    # ====================================================================
    with ProcessingTimer("Persistencia del modelo Scaler", log_file):
        scaler_path = MODELS_PATH / FILE_SCALER_MODEL
        joblib.dump(scaler, scaler_path)
        print(f"💾 Scaler guardado: {scaler_path}")
        del scaler
        gc.collect()
    
    # ====================================================================
    # 6. Verificación rápida
    # ====================================================================
    with ProcessingTimer("Verificación de integridad", log_file):
        features_check = np.load(str(output_path), mmap_mode='r')
        # Verificar sobre una muestra
        sample = np.array(features_check[:100], dtype=np.float32)
        print(f"✅ Shape: {features_check.shape}")
        print(f"✅ Media de muestra (debe ser ~0): {sample.mean():.4f}")
        print(f"✅ Std de muestra (debe ser ~1): {sample.std():.4f}")
        del features_check, sample
        gc.collect()
    
    # ====================================================================
    # 7. Checkpoint
    # ====================================================================
    print_memory_status("Fin del script")
    create_checkpoint('normalization', f"n_images={n_images}, scaler=StandardScaler")
    
    print("\n" + "=" * 70)
    print("  ✅ PASO 2 COMPLETADO - Datos normalizados (media=0, std=1)")
    print("  📌 Siguiente: Ejecutar 03_pca_reduction.py")
    print("  ⚠️  Esperar a que este script termine antes de continuar")
    print("=" * 70)


if __name__ == "__main__":
    main()
