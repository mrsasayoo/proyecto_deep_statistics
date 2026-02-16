#!/usr/bin/env python3
"""
06_evaluation_validation.py - Evaluación y Validación Cruzada
=============================================================
FASE 4 - Pasos 4.1 a 4.5

Evalúa los modelos K-Means y DBSCAN comparando sus clústeres con las
etiquetas reales de EuroSAT. Genera:
  - Matrices de confusión (raw y normalizadas)
  - Adjusted Rand Index (ARI) 
  - Normalized Mutual Information (NMI)
  - Silhouette Score comparativo
  - Tabla resumen de métricas

Input:  data_processed/features_pca_reduced.csv
        outputs/models/kmeans_labels.npy
        outputs/models/dbscan_labels.npy
Output: outputs/figures/04_evaluation/*.png
        outputs/tables/metrics_comparison.csv

⚠️ Ejecutar SOLO después de que 05_dbscan_clustering.py haya terminado.
"""

import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_PATH, MODELS_PATH, FIGURES_PATH, TABLES_PATH,
    FILE_FEATURES_PCA, FILE_KMEANS_LABELS, FILE_DBSCAN_LABELS, FILE_PROCESSING_LOG,
    CLASS_NAMES, FIGSIZE_LARGE,
    validate_checkpoint, create_checkpoint
)
from utils.memory_utils import print_memory_status, ProcessingTimer, check_memory_threshold
from utils.plotting_utils import configure_plot_style, save_figure


def compute_confusion_matrix_manual(true_labels, pred_labels, class_names):
    """
    Calcula la matriz de confusión entre etiquetas reales y predichas (clústeres).
    
    Para clustering, los clústeres no tienen orden natural, así que usamos
    la asignación mayoritaria: cada clúster se mapea a la clase real más 
    frecuente dentro de él.
    
    Soporta true_labels como strings (nombres de clase) o como enteros (índices).
    
    Returns:
        conf_matrix: np.ndarray, shape (n_true_classes, n_clusters)
        unique_clusters: list de cluster IDs
        cluster_to_class: dict, mapeo de clúster -> clase mayoritaria
    """
    
    unique_clusters = sorted(set(pred_labels))
    n_true = len(class_names)
    n_clusters = len(unique_clusters)
    
    # Convertir true_labels a índices numéricos si son strings
    if isinstance(true_labels[0], str) or (hasattr(true_labels, 'dtype') and true_labels.dtype.kind in ('U', 'S', 'O')):
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        true_indices = np.array([label_to_idx.get(str(l), -1) for l in true_labels])
    else:
        true_indices = true_labels.astype(int)
    
    # Construir la matriz de contingencia
    conf = np.zeros((n_true, n_clusters), dtype=int)
    for i in range(n_true):
        mask_true = true_indices == i
        for j, cluster_id in enumerate(unique_clusters):
            conf[i, j] = np.sum(pred_labels[mask_true] == cluster_id)
    
    # Mapeo por clase mayoritaria (simple y efectivo)
    cluster_to_class = {}
    for j, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:  # Ruido DBSCAN
            cluster_to_class[cluster_id] = "Ruido"
        else:
            majority_class_idx = np.argmax(conf[:, j])
            cluster_to_class[cluster_id] = class_names[majority_class_idx]
    
    return conf, unique_clusters, cluster_to_class


def plot_confusion_matrix(conf_matrix, true_names, cluster_labels, title, filepath, 
                          normalize=False):
    """Genera heatmap de la matriz de confusión."""
    configure_plot_style()
    
    if normalize:
        # Normalizar por fila (por clase real)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        conf_display = conf_matrix / row_sums
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        conf_display = conf_matrix
        fmt = 'd'
        vmin, vmax = None, None
    
    # Solo mostrar un subconjunto de clústeres si hay demasiados
    n_clusters_show = min(conf_matrix.shape[1], 20)
    
    fig, ax = plt.subplots(figsize=(max(10, n_clusters_show * 0.8), 8))
    sns.heatmap(conf_display[:, :n_clusters_show], annot=True, fmt=fmt, cmap='YlOrRd',
                xticklabels=[f"C{c}" for c in cluster_labels[:n_clusters_show]],
                yticklabels=true_names,
                ax=ax, vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor='white')
    ax.set_xlabel('Clúster Asignado')
    ax.set_ylabel('Clase Real')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, filepath)


def main():
    """Pipeline de evaluación y validación."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 6: EVALUACIÓN Y VALIDACIÓN DE MODELOS")
    print("=" * 70)
    
    # ====================================================================
    # 1. Validar checkpoint anterior
    # ====================================================================
    validate_checkpoint('dbscan')
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=1.0)
    
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        silhouette_score
    )
    
    # ====================================================================
    # 2. Cargar datos y etiquetas
    # ====================================================================
    with ProcessingTimer("Carga de datos y etiquetas", log_file):
        # Datos PCA
        df = pd.read_csv(DATA_PROCESSED_PATH / FILE_FEATURES_PCA)
        true_labels_raw = df['true_label'].values  # strings
        X = df.drop(columns=['true_label']).values.astype(np.float32)
        del df
        gc.collect()
        
        # Convertir labels de string a índice numérico
        label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        true_labels = np.array([label_to_idx.get(str(l), -1) for l in true_labels_raw])
        del true_labels_raw
        
        # Etiquetas K-Means
        kmeans_labels = np.load(MODELS_PATH / FILE_KMEANS_LABELS)
        
        # Etiquetas DBSCAN
        dbscan_labels = np.load(MODELS_PATH / FILE_DBSCAN_LABELS)
        
        print(f"📊 Datos: {X.shape}")
        print(f"   Clases reales: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES[:3])}...)")
        print(f"   K-Means labels: {len(set(kmeans_labels))} clústeres")
        print(f"   DBSCAN labels: {len(set(dbscan_labels))} clústeres "
              f"(incluyendo ruido={np.sum(dbscan_labels == -1)})")
    
    # ====================================================================
    # 3. Directorio de figuras
    # ====================================================================
    eval_fig_dir = FIGURES_PATH / "04_evaluation"
    eval_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ====================================================================
    # 4. Métricas K-Means
    # ====================================================================
    with ProcessingTimer("Evaluación K-Means", log_file):
        ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
        nmi_kmeans = normalized_mutual_info_score(true_labels, kmeans_labels)
        
        # Silhouette Score (usar muestra para eficiencia)
        n_sil = min(10000, X.shape[0])
        np.random.seed(42)
        sil_idx = np.random.choice(X.shape[0], n_sil, replace=False)
        sil_kmeans = silhouette_score(X[sil_idx], kmeans_labels[sil_idx])
        
        print(f"\n📊 K-Means:")
        print(f"   ARI  = {ari_kmeans:.4f}")
        print(f"   NMI  = {nmi_kmeans:.4f}")
        print(f"   Silueta = {sil_kmeans:.4f}")
    
    # ====================================================================
    # 5. Métricas DBSCAN
    # ====================================================================
    with ProcessingTimer("Evaluación DBSCAN", log_file):
        ari_dbscan = adjusted_rand_score(true_labels, dbscan_labels)
        nmi_dbscan = normalized_mutual_info_score(true_labels, dbscan_labels)
        
        # Silhouette solo para no-ruido
        non_noise = dbscan_labels != -1
        if np.sum(non_noise) > 100 and len(set(dbscan_labels[non_noise])) >= 2:
            n_sil_db = min(10000, np.sum(non_noise))
            np.random.seed(42)
            non_noise_idx = np.where(non_noise)[0]
            sil_db_idx = np.random.choice(non_noise_idx, n_sil_db, replace=False)
            sil_dbscan = silhouette_score(X[sil_db_idx], dbscan_labels[sil_db_idx])
        else:
            sil_dbscan = -1.0
        
        print(f"\n📊 DBSCAN:")
        print(f"   ARI  = {ari_dbscan:.4f}")
        print(f"   NMI  = {nmi_dbscan:.4f}")
        print(f"   Silueta (sin ruido) = {sil_dbscan:.4f}")
        print(f"   Ruido = {np.sum(~non_noise)} ({np.sum(~non_noise)/len(dbscan_labels)*100:.1f}%)")
    
    # ====================================================================
    # 6. Tabla comparativa
    # ====================================================================
    with ProcessingTimer("Tabla comparativa de métricas", log_file):
        comparison = pd.DataFrame({
            'Metrica': ['ARI', 'NMI', 'Silhouette Score', 'N_Clusters', 'N_Ruido'],
            'K-Means': [ari_kmeans, nmi_kmeans, sil_kmeans, 
                        len(set(kmeans_labels)), 0],
            'DBSCAN': [ari_dbscan, nmi_dbscan, sil_dbscan,
                       len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                       np.sum(dbscan_labels == -1)]
        })
        
        metrics_path = TABLES_PATH / "metrics_comparison.csv"
        comparison.to_csv(metrics_path, index=False)
        
        print(f"\n{'='*55}")
        print(f"  COMPARACIÓN: K-Means vs DBSCAN")
        print(f"{'='*55}")
        print(comparison.to_string(index=False))
        print(f"\n💾 Tabla guardada: {metrics_path}")
    
    # ====================================================================
    # 7. Matrices de confusión K-Means
    # ====================================================================
    with ProcessingTimer("Matrices de confusión K-Means", log_file):
        conf_km, clusters_km, map_km = compute_confusion_matrix_manual(
            true_labels, kmeans_labels, CLASS_NAMES
        )
        
        plot_confusion_matrix(
            conf_km, CLASS_NAMES, clusters_km,
            f"Matriz de Confusión K-Means\n(K={len(set(kmeans_labels))} vs {len(CLASS_NAMES)} clases reales)",
            eval_fig_dir / "confusion_matrix_kmeans_raw.png",
            normalize=False
        )
        
        plot_confusion_matrix(
            conf_km, CLASS_NAMES, clusters_km,
            f"Matriz de Confusión Normalizada K-Means\n(K={len(set(kmeans_labels))} vs {len(CLASS_NAMES)} clases reales)",
            eval_fig_dir / "confusion_matrix_kmeans_normalized.png",
            normalize=True
        )
        
        print(f"\n📊 Mapeo K-Means (clúster -> clase mayoritaria):")
        for k, v in sorted(map_km.items()):
            print(f"   Cluster {k} -> {v}")
    
    # ====================================================================
    # 8. Matrices de confusión DBSCAN
    # ====================================================================
    with ProcessingTimer("Matrices de confusión DBSCAN", log_file):
        conf_db, clusters_db, map_db = compute_confusion_matrix_manual(
            true_labels, dbscan_labels, CLASS_NAMES
        )
        
        plot_confusion_matrix(
            conf_db, CLASS_NAMES, clusters_db,
            f"Matriz de Confusión DBSCAN\n(eps seleccionado, {len(set(dbscan_labels))} clústeres)",
            eval_fig_dir / "confusion_matrix_dbscan_raw.png",
            normalize=False
        )
        
        plot_confusion_matrix(
            conf_db, CLASS_NAMES, clusters_db,
            f"Matriz de Confusión Normalizada DBSCAN\n(eps seleccionado, {len(set(dbscan_labels))} clústeres)",
            eval_fig_dir / "confusion_matrix_dbscan_normalized.png",
            normalize=True
        )
        
        print(f"\n📊 Mapeo DBSCAN (clúster -> clase mayoritaria):")
        for k, v in sorted(map_db.items()):
            label = f"Cluster {k}" if k >= 0 else "Ruido"
            print(f"   {label} -> {v}")
    
    # ====================================================================
    # 9. Gráfico comparativo de barras (ARI, NMI, Silhouette)
    # ====================================================================
    with ProcessingTimer("Visualización comparativa", log_file):
        configure_plot_style()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['ARI', 'NMI', 'Silhouette']
        km_vals = [ari_kmeans, nmi_kmeans, sil_kmeans]
        db_vals = [ari_dbscan, nmi_dbscan, sil_dbscan]
        colors = ['#2196F3', '#FF9800']
        
        for ax, metric, km_v, db_v in zip(axes, metrics, km_vals, db_vals):
            bars = ax.bar(['K-Means', 'DBSCAN'], [km_v, db_v], color=colors, 
                         edgecolor='black', linewidth=0.8)
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.set_ylim(-0.1, 1.1)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            # Valores sobre las barras
            for bar, val in zip(bars, [km_v, db_v]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        fig.suptitle('Comparación de Métricas: K-Means vs DBSCAN\nEuroSAT Multiespectral',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_figure(fig, eval_fig_dir / "metrics_comparison_barplot.png")
    
    # ====================================================================
    # 10. Checkpoint
    # ====================================================================
    del X, true_labels, kmeans_labels, dbscan_labels
    gc.collect()
    print_memory_status("Fin del script")
    
    create_checkpoint('evaluation', 
                      f"ARI(KM={ari_kmeans:.3f},DB={ari_dbscan:.3f}), "
                      f"NMI(KM={nmi_kmeans:.3f},DB={nmi_dbscan:.3f})")
    
    print("\n" + "=" * 70)
    print(f"  ✅ PASO 6 COMPLETADO - EVALUACIÓN DE MODELOS")
    print(f"  📊 ARI: K-Means={ari_kmeans:.4f}, DBSCAN={ari_dbscan:.4f}")
    print(f"  📊 NMI: K-Means={nmi_kmeans:.4f}, DBSCAN={nmi_dbscan:.4f}")
    print(f"  📌 Siguiente: Ejecutar 07_visualization_export.py")
    print(f"  ⚠️  Esperar a que este script termine antes de continuar")
    print("=" * 70)


if __name__ == "__main__":
    main()
