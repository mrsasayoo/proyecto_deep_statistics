#!/usr/bin/env python3
"""
04_kmeans_clustering.py - Modelado K-Means con Optimización de k
=================================================================
FASE 2 - Pasos 2.1 a 2.6

Ejecuta K-Means sobre los datos reducidos de PCA para k=2..15.
Genera gráficos del Codo (SSE) y Coeficiente de Silueta para
determinar el k óptimo. Hipótesis: k entre 8-12.

Input:  data_processed/features_pca_reduced.csv
Output: outputs/models/kmeans_model.pkl
        outputs/models/kmeans_labels.npy
        outputs/figures/02_kmeans/elbow_plot_sse.png
        outputs/figures/02_kmeans/silhouette_scores.png
        outputs/figures/02_kmeans/clusters_pca_space.png
        outputs/tables/kmeans_optimization_results.csv
        
⚠️ Ejecutar SOLO después de que 03_pca_reduction.py haya terminado.
"""

import sys
import gc
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_PATH, MODELS_PATH, FIGURES_PATH, TABLES_PATH,
    KMEANS_K_RANGE, KMEANS_OPTIMAL_K_EXPECTED, KMEANS_RANDOM_STATE, KMEANS_N_INIT, KMEANS_MAX_ITER,
    FILE_FEATURES_PCA, FILE_KMEANS_MODEL, FILE_KMEANS_LABELS, FILE_PROCESSING_LOG,
    FIGSIZE_MEDIUM,
    validate_checkpoint, create_checkpoint
)
from utils.memory_utils import print_memory_status, ProcessingTimer, check_memory_threshold
from utils.plotting_utils import configure_plot_style, save_figure


def main():
    """Pipeline de clustering K-Means."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 4: MODELADO K-MEANS CON OPTIMIZACIÓN DE K")
    print("=" * 70)
    
    # ====================================================================
    # 1. Validar checkpoint anterior
    # ====================================================================
    validate_checkpoint('pca_reduction')
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=1.5)
    
    # ====================================================================
    # 2. Cargar datos PCA reducidos (ligeros)
    # ====================================================================
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    with ProcessingTimer("Carga de datos PCA reducidos", log_file):
        pca_path = DATA_PROCESSED_PATH / FILE_FEATURES_PCA
        df = pd.read_csv(pca_path)
        
        # Separar etiquetas reales (para validación posterior) y features
        true_labels = df['true_label'].values
        X = df.drop(columns=['true_label']).values.astype(np.float32)
        
        print(f"📊 Datos cargados: {X.shape}")
        print(f"📊 Componentes PCA: {X.shape[1]}")
        del df
        gc.collect()
    
    # ====================================================================
    # 3. Búsqueda del k óptimo (k = 2..15)
    # ====================================================================
    with ProcessingTimer("Búsqueda de k óptimo (SSE + Silueta)", log_file):
        
        k_values = list(KMEANS_K_RANGE)
        sse_scores = []
        silhouette_scores = []
        
        print(f"🔄 Probando k = {k_values[0]} a {k_values[-1]}...")
        print(f"   n_init={KMEANS_N_INIT}, max_iter={KMEANS_MAX_ITER}, random_state={KMEANS_RANDOM_STATE}")
        
        for k in k_values:
            print(f"\n   k={k}: ", end="", flush=True)
            
            km = KMeans(
                n_clusters=k,
                n_init=KMEANS_N_INIT,
                max_iter=KMEANS_MAX_ITER,
                random_state=KMEANS_RANDOM_STATE,
                algorithm='lloyd'
            )
            km.fit(X)
            
            # SSE (Sum of Squared Errors) = inertia
            sse = km.inertia_
            sse_scores.append(sse)
            
            # Coeficiente de Silueta (sobre muestra si dataset es grande)
            if X.shape[0] > 10000:
                # Muestra aleatoria para calcular silueta más rápido
                np.random.seed(KMEANS_RANDOM_STATE)
                sample_idx = np.random.choice(X.shape[0], 10000, replace=False)
                sil = silhouette_score(X[sample_idx], km.labels_[sample_idx])
            else:
                sil = silhouette_score(X, km.labels_)
            silhouette_scores.append(sil)
            
            print(f"SSE={sse:.0f}, Silueta={sil:.4f}", flush=True)
            
            del km
            gc.collect()
    
    # ====================================================================
    # 4. Determinar k óptimo
    # ====================================================================
    with ProcessingTimer("Determinación de k óptimo", log_file):
        # k con mayor silueta global
        best_k_silhouette_global = k_values[np.argmax(silhouette_scores)]
        best_silhouette_global = max(silhouette_scores)
        
        # Selección dentro del rango esperado (8-12) según hipótesis del proyecto
        # El dataset tiene 10 clases reales, por lo que el k óptimo debería
        # estar en ese rango. La silueta global decrece con k, lo cual es
        # esperado en datasets con múltiples clases similares entre sí.
        k_expected_range = list(KMEANS_OPTIMAL_K_EXPECTED)  # [8, 9, 10, 11, 12]
        
        # Buscar el mejor k dentro del rango esperado
        silhouette_in_range = {
            k: silhouette_scores[k_values.index(k)] 
            for k in k_expected_range if k in k_values
        }
        best_k_in_range = max(silhouette_in_range, key=silhouette_in_range.get)
        best_sil_in_range = silhouette_in_range[best_k_in_range]
        
        # Usar k del rango esperado (coincide con las 10 clases reales)
        k_optimal = best_k_in_range
        
        print(f"\n📊 Análisis de k óptimo:")
        print(f"   Mejor k global por Silueta: k={best_k_silhouette_global} (sil={best_silhouette_global:.4f})")
        print(f"   Mejor k en rango esperado [{k_expected_range[0]}-{k_expected_range[-1]}]: k={best_k_in_range} (sil={best_sil_in_range:.4f})")
        print(f"   ⚠️  La silueta global decrece con k, lo cual es esperado")
        print(f"       en un dataset con 10 clases de cobertura terrestre similar.")
        print(f"\n🎯 k seleccionado: k={k_optimal} (rango hipótesis del proyecto)")
        
        # Guardar resultados de optimización
        df_optim = pd.DataFrame({
            'k': k_values,
            'sse': sse_scores,
            'silhouette_score': silhouette_scores
        })
        optim_path = TABLES_PATH / "kmeans_optimization_results.csv"
        df_optim.to_csv(optim_path, index=False)
        print(f"💾 Resultados de optimización: {optim_path}")
    
    # ====================================================================
    # 5. Generar gráficos
    # ====================================================================
    with ProcessingTimer("Generación de gráficos K-Means", log_file):
        configure_plot_style()
        kmeans_fig_dir = FIGURES_PATH / "02_kmeans"
        kmeans_fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráfico 1: Método del Codo (SSE)
        fig1, ax1 = plt.subplots(figsize=FIGSIZE_MEDIUM)
        ax1.plot(k_values, sse_scores, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=k_optimal, color='r', linestyle='--', 
                     label=f'k seleccionado = {k_optimal}')
        ax1.set_xlabel('Número de Clústeres (k)')
        ax1.set_ylabel('SSE (Sum of Squared Errors / Inercia)')
        ax1.set_title('Método del Codo - K-Means\nEuroSAT Multiespectral')
        ax1.set_xticks(k_values)
        ax1.legend(fontsize=11)
        save_figure(fig1, kmeans_fig_dir / "elbow_plot_sse.png")
        
        # Gráfico 2: Coeficiente de Silueta
        fig2, ax2 = plt.subplots(figsize=FIGSIZE_MEDIUM)
        ax2.plot(k_values, silhouette_scores, 'gs-', linewidth=2, markersize=8)
        ax2.axvline(x=k_optimal, color='r', linestyle='--',
                     label=f'k seleccionado = {k_optimal} (sil={best_sil_in_range:.3f})')
        ax2.axvline(x=best_k_silhouette_global, color='orange', linestyle=':',
                     label=f'Mejor global = {best_k_silhouette_global} (sil={best_silhouette_global:.3f})')
        ax2.axvspan(k_expected_range[0], k_expected_range[-1], alpha=0.1, color='green',
                     label=f'Rango esperado [{k_expected_range[0]}-{k_expected_range[-1]}]')
        ax2.set_xlabel('Número de Clústeres (k)')
        ax2.set_ylabel('Coeficiente de Silueta Promedio')
        ax2.set_title('Coeficiente de Silueta vs. k - K-Means\nEuroSAT Multiespectral')
        ax2.set_xticks(k_values)
        ax2.legend(fontsize=10)
        save_figure(fig2, kmeans_fig_dir / "silhouette_scores.png")
    
    # ====================================================================
    # 6. Ajustar modelo final con k óptimo
    # ====================================================================
    with ProcessingTimer(f"Ajuste final K-Means con k={k_optimal}", log_file):
        
        kmeans_final = KMeans(
            n_clusters=k_optimal,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            random_state=KMEANS_RANDOM_STATE,
            algorithm='lloyd'
        )
        kmeans_final.fit(X)
        
        kmeans_labels = kmeans_final.labels_
        
        print(f"✅ K-Means final ajustado con k={k_optimal}")
        print(f"   Distribución de clústeres:")
        for c in range(k_optimal):
            count = np.sum(kmeans_labels == c)
            print(f"   Cluster {c}: {count} imágenes ({count/len(kmeans_labels)*100:.1f}%)")
    
    # ====================================================================
    # 7. Visualización de clústeres en espacio PCA 2D
    # ====================================================================
    with ProcessingTimer("Visualización de clústeres en PCA 2D", log_file):
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        
        # Muestra para visualización
        n_sample = min(5000, X.shape[0])
        np.random.seed(42)
        sample_idx = np.random.choice(X.shape[0], n_sample, replace=False)
        
        scatter = ax3.scatter(X[sample_idx, 0], X[sample_idx, 1],
                              c=kmeans_labels[sample_idx], cmap='tab10',
                              alpha=0.4, s=8, edgecolors='none')
        
        # Centroides
        centroids = kmeans_final.cluster_centers_
        ax3.scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='X', s=200, edgecolors='black',
                    linewidths=1.5, zorder=10, label='Centroides')
        
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title(f'K-Means Clustering (k={k_optimal}) en Espacio PCA 2D\n'
                       f'(Muestra de {n_sample:,} imágenes)')
        plt.colorbar(scatter, ax=ax3, label='Clúster')
        ax3.legend(fontsize=11)
        save_figure(fig3, kmeans_fig_dir / "clusters_pca_space.png")
    
    # ====================================================================
    # 8. Guardar modelo y etiquetas
    # ====================================================================
    with ProcessingTimer("Persistencia de modelo y etiquetas", log_file):
        joblib.dump(kmeans_final, MODELS_PATH / FILE_KMEANS_MODEL)
        np.save(MODELS_PATH / FILE_KMEANS_LABELS, kmeans_labels)
        print(f"💾 Modelo K-Means: {MODELS_PATH / FILE_KMEANS_MODEL}")
        print(f"💾 Etiquetas: {MODELS_PATH / FILE_KMEANS_LABELS}")
    
    # ====================================================================
    # 9. Checkpoint
    # ====================================================================
    del X, kmeans_final, kmeans_labels, true_labels
    gc.collect()
    print_memory_status("Fin del script")
    create_checkpoint('kmeans', f"k_optimal={k_optimal}, silhouette={best_sil_in_range:.4f}")
    
    print("\n" + "=" * 70)
    print(f"  ✅ PASO 4 COMPLETADO - K-Means con k={k_optimal}")
    print(f"  📊 Silueta: {best_sil_in_range:.4f}")
    print(f"  📌 Siguiente: Ejecutar 05_dbscan_clustering.py")
    print(f"  ⚠️  Esperar a que este script termine antes de continuar")
    print("=" * 70)


if __name__ == "__main__":
    main()
