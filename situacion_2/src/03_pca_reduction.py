#!/usr/bin/env python3
"""
03_pca_reduction.py - Reducción Dimensional con PCA
====================================================
FASE 1 - Paso 1.5, 1.6

Reduce la matriz normalizada de 53,248 features a m componentes
que expliquen >90% de la varianza acumulada.

Estrategia: Usa IncrementalPCA (batch-wise) para manejar la alta
dimensionalidad sin cargar toda la matriz en RAM.

Input:  data_processed/features_normalized.npy
Output: data_processed/features_pca_reduced.csv  (27000 x m) - LIGERO
        data_processed/pca_variance_explained.csv
        outputs/models/pca_model.pkl
        outputs/figures/01_pca/variance_explained_cumulative.png
        outputs/figures/01_pca/scree_plot.png
        
⚠️ Ejecutar SOLO después de que 02_normalization.py haya terminado.
"""

import sys
import gc
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_PATH, MODELS_PATH, FIGURES_PATH,
    BATCH_SIZE, PCA_VARIANCE_THRESHOLD, PCA_MAX_COMPONENTS,
    FILE_FEATURES_NORMALIZED, FILE_FEATURES_PCA,
    FILE_PCA_VARIANCE, FILE_PCA_MODEL, FILE_METADATA, FILE_PROCESSING_LOG,
    FIGSIZE_MEDIUM, FIGURE_DPI,
    validate_checkpoint, create_checkpoint
)
from utils.memory_utils import print_memory_status, ProcessingTimer, check_memory_threshold
from utils.plotting_utils import configure_plot_style, save_figure


def main():
    """Pipeline de reducción dimensional con PCA."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 3: REDUCCIÓN DIMENSIONAL CON PCA")
    print("=" * 70)
    
    # ====================================================================
    # 1. Validar checkpoint anterior
    # ====================================================================
    validate_checkpoint('normalization')
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=2.0)
    
    # ====================================================================
    # 2. Cargar datos normalizados (memmap)
    # ====================================================================
    norm_path = DATA_PROCESSED_PATH / FILE_FEATURES_NORMALIZED
    
    with ProcessingTimer("Carga de datos normalizados (memmap)", log_file):
        features_norm = np.load(str(norm_path), mmap_mode='r')
        n_images, n_features = features_norm.shape
        print(f"📊 Datos: {features_norm.shape} ({features_norm.dtype})")
    
    # ====================================================================
    # 3. PCA Incremental (fit por lotes)
    # ====================================================================
    from sklearn.decomposition import IncrementalPCA
    
    # Limitar componentes al mínimo entre PCA_MAX_COMPONENTS y n_features
    n_components_fit = min(PCA_MAX_COMPONENTS, n_features, n_images)
    
    with ProcessingTimer(f"IncrementalPCA fit ({n_components_fit} componentes)", log_file):
        ipca = IncrementalPCA(n_components=n_components_fit)
        
        n_batches = (n_images + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"🔄 Ajustando PCA en {n_batches} lotes...")
        
        for batch_idx in tqdm(range(n_batches), desc="IPCA fit"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_images)
            batch = np.array(features_norm[start_idx:end_idx], dtype=np.float32)
            ipca.partial_fit(batch)
            del batch
            gc.collect()
    
    # ====================================================================
    # 4. Determinar número óptimo de componentes (>90% varianza)
    # ====================================================================
    with ProcessingTimer("Determinación de componentes óptimos", log_file):
        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        
        # Encontrar m donde se supera el umbral de varianza
        m_optimal = np.searchsorted(cumulative_variance, PCA_VARIANCE_THRESHOLD) + 1
        m_optimal = max(m_optimal, 2)  # Mínimo 2 componentes
        
        print(f"📊 Varianza explicada por los primeros componentes:")
        for i in [1, 2, 5, 10, 20, 50, min(100, n_components_fit)]:
            if i <= len(cumulative_variance):
                print(f"   {i} componentes: {cumulative_variance[i-1]*100:.2f}%")
        
        print(f"\n🎯 Componentes para >{PCA_VARIANCE_THRESHOLD*100:.0f}% varianza: m = {m_optimal}")
        print(f"   Varianza acumulada con m={m_optimal}: {cumulative_variance[m_optimal-1]*100:.2f}%")
        print(f"   Reducción: {n_features:,} → {m_optimal} ({(1-m_optimal/n_features)*100:.2f}%)")
    
    # ====================================================================
    # 5. Guardar tabla de varianza explicada
    # ====================================================================
    with ProcessingTimer("Guardado de varianza explicada", log_file):
        df_variance = pd.DataFrame({
            'component': range(1, n_components_fit + 1),
            'explained_variance_ratio': ipca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance
        })
        variance_path = DATA_PROCESSED_PATH / FILE_PCA_VARIANCE
        df_variance.to_csv(variance_path, index=False)
        print(f"💾 Varianza guardada: {variance_path}")
    
    # ====================================================================
    # 6. Generar gráficos de varianza
    # ====================================================================
    with ProcessingTimer("Generación de gráficos PCA", log_file):
        configure_plot_style()
        pca_fig_dir = FIGURES_PATH / "01_pca"
        pca_fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráfico 1: Varianza explicada acumulada
        fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
        ax.plot(range(1, n_components_fit + 1), cumulative_variance * 100, 
                'b-', linewidth=2, label='Varianza acumulada')
        ax.axhline(y=PCA_VARIANCE_THRESHOLD * 100, color='r', linestyle='--', 
                    linewidth=1.5, label=f'Umbral {PCA_VARIANCE_THRESHOLD*100:.0f}%')
        ax.axvline(x=m_optimal, color='g', linestyle='--', 
                    linewidth=1.5, label=f'm óptimo = {m_optimal}')
        ax.set_xlabel('Número de Componentes')
        ax.set_ylabel('Varianza Explicada Acumulada (%)')
        ax.set_title('PCA: Varianza Explicada Acumulada\nImágenes Multiespectrales EuroSAT (13 bandas Sentinel-2)')
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim(1, n_components_fit)
        ax.set_ylim(0, 102)
        save_figure(fig, pca_fig_dir / "variance_explained_cumulative.png")
        
        # Gráfico 2: Scree Plot (varianza individual por componente)
        fig2, ax2 = plt.subplots(figsize=FIGSIZE_MEDIUM)
        n_show = min(50, n_components_fit)  # Mostrar hasta 50 componentes
        ax2.bar(range(1, n_show + 1), 
                ipca.explained_variance_ratio_[:n_show] * 100, 
                color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
        ax2.set_xlabel('Componente Principal')
        ax2.set_ylabel('Varianza Explicada Individual (%)')
        ax2.set_title(f'Scree Plot - Primeros {n_show} Componentes\nEuroSAT Multiespectral')
        save_figure(fig2, pca_fig_dir / "scree_plot.png")
    
    # ====================================================================
    # 7. Transformar datos y guardar matriz reducida
    # ====================================================================
    with ProcessingTimer(f"Transformación PCA (→ {m_optimal} componentes)", log_file):
        
        # Transformar por lotes
        features_pca = np.empty((n_images, m_optimal), dtype=np.float32)
        
        for batch_idx in tqdm(range(n_batches), desc="IPCA transform"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_images)
            batch = np.array(features_norm[start_idx:end_idx], dtype=np.float32)
            
            # Transformar y recortar a m_optimal componentes
            batch_transformed = ipca.transform(batch)[:, :m_optimal]
            features_pca[start_idx:end_idx] = batch_transformed.astype(np.float32)
            
            del batch, batch_transformed
            gc.collect()
        
        # Liberar el memmap de datos normalizados
        del features_norm
        gc.collect()
        
        # Guardar como CSV (liviano, fácil de cargar para clustering)
        col_names = [f"PC{i+1}" for i in range(m_optimal)]
        df_pca = pd.DataFrame(features_pca, columns=col_names)
        
        # Agregar etiquetas reales desde metadatos
        metadata_path = DATA_PROCESSED_PATH / FILE_METADATA
        df_meta = pd.read_csv(metadata_path)
        df_pca.insert(0, 'true_label', df_meta['true_label'].values)
        
        pca_output_path = DATA_PROCESSED_PATH / FILE_FEATURES_PCA
        df_pca.to_csv(pca_output_path, index=False)
        
        print(f"\n💾 Matriz PCA guardada: {pca_output_path}")
        print(f"   Dimensiones: {df_pca.shape}")
        print(f"   Tamaño: {pca_output_path.stat().st_size / (1024**2):.1f} MB")
        
        del df_pca, df_meta
        gc.collect()
    
    # ====================================================================
    # 8. Generar proyección 2D
    # ====================================================================
    with ProcessingTimer("Generación de proyección PCA 2D", log_file):
        configure_plot_style()
        
        df_plot = pd.read_csv(pca_output_path, nrows=5000)  # Muestra para no saturar
        
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        labels_unique = sorted(df_plot['true_label'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels_unique)))
        
        for idx, label in enumerate(labels_unique):
            mask = df_plot['true_label'] == label
            ax3.scatter(df_plot.loc[mask, 'PC1'], df_plot.loc[mask, 'PC2'],
                       c=[colors[idx]], label=label, alpha=0.4, s=8, edgecolors='none')
        
        ax3.set_xlabel(f'PC1 ({ipca.explained_variance_ratio_[0]*100:.1f}% varianza)')
        ax3.set_ylabel(f'PC2 ({ipca.explained_variance_ratio_[1]*100:.1f}% varianza)')
        ax3.set_title('Proyección PCA 2D - Imágenes EuroSAT\n(Muestra de 5,000 imágenes)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, markerscale=3)
        save_figure(fig3, pca_fig_dir / "pca_2d_projection.png")
        
        del df_plot
        gc.collect()
    
    # ====================================================================
    # 9. Guardar modelo PCA
    # ====================================================================
    with ProcessingTimer("Persistencia del modelo PCA", log_file):
        pca_model_path = MODELS_PATH / FILE_PCA_MODEL
        joblib.dump(ipca, pca_model_path)
        print(f"💾 Modelo PCA guardado: {pca_model_path}")
        del ipca, features_pca
        gc.collect()
    
    # ====================================================================
    # 10. Checkpoint
    # ====================================================================
    print_memory_status("Fin del script")
    create_checkpoint('pca_reduction', 
                      f"n_components={m_optimal}, variance={cumulative_variance[m_optimal-1]*100:.2f}%")
    
    print("\n" + "=" * 70)
    print(f"  ✅ PASO 3 COMPLETADO - Reducción PCA: {n_features:,} → {m_optimal} componentes")
    print(f"  📊 Varianza explicada: {cumulative_variance[m_optimal-1]*100:.2f}%")
    print(f"  📌 Siguiente: Ejecutar 04_kmeans_clustering.py")
    print(f"  ⚠️  Esperar a que este script termine antes de continuar")
    print("=" * 70)


if __name__ == "__main__":
    main()
