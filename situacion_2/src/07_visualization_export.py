#!/usr/bin/env python3
"""
07_visualization_export.py - Visualización Final y Exportación
==============================================================
FASE 5 - Pasos 5.1 a 5.5

Genera todas las visualizaciones finales e interpretativas:
  - Muestras de imágenes reales por clúster (K-Means y DBSCAN)
  - Heatmap de composición de clústeres
  - Distribución de clases por clúster
  - Visualización RGB de imágenes representativas
  - Exportación de tablas finales

Input:  data_processed/metadata_labels.csv
        data_processed/features_pca_reduced.csv
        outputs/models/kmeans_labels.npy
        outputs/models/dbscan_labels.npy
Output: outputs/figures/05_interpretation/*.png
        outputs/tables/cluster_composition_*.csv

⚠️ Ejecutar SOLO después de que 06_evaluation_validation.py haya terminado.
"""

import sys
import gc
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_PATH, MODELS_PATH, FIGURES_PATH, TABLES_PATH,
    DATASET_PATH, CLASS_NAMES, FIGSIZE_LARGE,
    FILE_FEATURES_PCA, FILE_KMEANS_LABELS, FILE_DBSCAN_LABELS,
    FILE_METADATA, FILE_PROCESSING_LOG,
    validate_checkpoint, create_checkpoint
)
from utils.memory_utils import print_memory_status, ProcessingTimer, check_memory_threshold
from utils.plotting_utils import configure_plot_style, save_figure


def load_tif_as_rgb(filepath):
    """
    Carga una imagen .tif multiespectral y extrae bandas RGB (B4, B3, B2)
    para visualización. Las bandas Sentinel-2 en EuroSAT MS están en orden:
      B01..B12, B8A → 13 bandas
    Bandas RGB Sentinel-2: B04=rojo [idx 3], B03=verde [idx 2], B02=azul [idx 1]
    
    Returns:
        rgb: np.ndarray, shape (64, 64, 3), normalizado a [0, 1] float32
    """
    img = tifffile.imread(filepath)
    
    # Manejar forma (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    
    # Extraer bandas RGB (B4=rojo, B3=verde, B2=azul)
    # Índices: B02=1, B03=2, B04=3
    rgb = img[:, :, [3, 2, 1]].astype(np.float32)
    
    # Normalizar por percentiles para mejorar contraste
    for c in range(3):
        band = rgb[:, :, c]
        p2, p98 = np.percentile(band, (2, 98))
        if p98 > p2:
            rgb[:, :, c] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            rgb[:, :, c] = 0
    
    return rgb


def plot_sample_images_per_cluster(metadata, labels, model_name, output_dir, 
                                    n_samples=5, n_clusters_show=None):
    """
    Muestra ejemplos de imágenes RGB para cada clúster.
    
    Args:
        metadata: DataFrame con columnas ['filepath', 'class_name']
        labels: np.ndarray de etiquetas de clúster
        model_name: str ('K-Means' o 'DBSCAN')
        output_dir: Path de salida
        n_samples: int, imágenes por clúster
        n_clusters_show: int o None, máx clústeres a mostrar
    """
    unique_clusters = sorted(set(labels))
    # Excluir ruido para la visualización de grilla
    clusters_to_show = [c for c in unique_clusters if c >= 0]
    
    if n_clusters_show is not None:
        clusters_to_show = clusters_to_show[:n_clusters_show]
    
    n_rows = len(clusters_to_show)
    if n_rows == 0:
        print(f"   ⚠️ No hay clústeres válidos para {model_name}")
        return
    
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_samples == 1:
        axes = axes[:, np.newaxis]
    
    for row, cluster_id in enumerate(clusters_to_show):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
        # Seleccionar muestras aleatorias del clúster
        np.random.seed(42 + cluster_id)
        n_avail = min(n_samples, len(cluster_indices))
        sample_indices = np.random.choice(cluster_indices, n_avail, replace=False)
        
        for col in range(n_samples):
            ax = axes[row, col]
            if col < n_avail:
                idx = sample_indices[col]
                filepath = metadata.iloc[idx]['filepath']
                try:
                    rgb = load_tif_as_rgb(filepath)
                    ax.imshow(rgb)
                    class_name = metadata.iloc[idx]['class_name']
                    ax.set_title(class_name, fontsize=7, pad=2)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=8)
            ax.axis('off')
        
        axes[row, 0].set_ylabel(f'C{cluster_id}', fontsize=10, fontweight='bold',
                                 rotation=0, labelpad=30)
    
    fig.suptitle(f'Muestras por Clúster - {model_name}\n(Bandas RGB: B4, B3, B2)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_figure(fig, output_dir / f"sample_images_{model_name.lower().replace('-','')}.png")


def plot_cluster_composition_heatmap(true_labels, pred_labels, class_names, 
                                      model_name, output_dir):
    """
    Heatmap mostrando la composición de cada clúster (% de cada clase real).
    """
    unique_clusters = sorted([c for c in set(pred_labels) if c >= 0])
    n_clusters = len(unique_clusters)
    
    # Construir tabla de composición
    composition = np.zeros((n_clusters, len(class_names)))
    for i, cid in enumerate(unique_clusters):
        mask = pred_labels == cid
        for j, cls in enumerate(class_names):
            composition[i, j] = np.sum(true_labels[mask] == j)
    
    # Normalizar por fila (porcentaje dentro de cada clúster)
    row_sums = composition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    composition_pct = composition / row_sums * 100
    
    # Guardar tabla
    df_comp = pd.DataFrame(composition_pct, 
                            index=[f'Cluster_{c}' for c in unique_clusters],
                            columns=class_names)
    comp_path = TABLES_PATH / f"cluster_composition_{model_name.lower().replace('-','')}.csv"
    df_comp.to_csv(comp_path)
    
    # Heatmap
    configure_plot_style()
    n_show = min(n_clusters, 20)  # Limitar para legibilidad
    
    fig, ax = plt.subplots(figsize=(12, max(6, n_show * 0.5)))
    sns.heatmap(composition_pct[:n_show], annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=class_names,
                yticklabels=[f'C{c}' for c in unique_clusters[:n_show]],
                ax=ax, vmin=0, vmax=100,
                linewidths=0.5, linecolor='white')
    ax.set_xlabel('Clase Real (EuroSAT)')
    ax.set_ylabel('Clúster')
    ax.set_title(f'Composición de Clústeres - {model_name}\n(% de cada clase real por clúster)',
                 fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, output_dir / f"composition_heatmap_{model_name.lower().replace('-','')}.png")
    
    return df_comp


def plot_class_distribution_per_cluster(true_labels, pred_labels, class_names,
                                         model_name, output_dir):
    """
    Gráfico de barras apiladas: distribución de clases para cada clúster.
    """
    unique_clusters = sorted([c for c in set(pred_labels) if c >= 0])
    
    data = []
    for cid in unique_clusters:
        mask = pred_labels == cid
        for j, cls in enumerate(class_names):
            data.append({
                'Cluster': f'C{cid}',
                'Clase': cls,
                'Count': np.sum(true_labels[mask] == j)
            })
    
    df = pd.DataFrame(data)
    pivot = df.pivot(index='Cluster', columns='Clase', values='Count').fillna(0)
    
    configure_plot_style()
    n_show = min(len(unique_clusters), 15)
    
    fig, ax = plt.subplots(figsize=(max(10, n_show), 7))
    pivot.iloc[:n_show].plot(kind='bar', stacked=True, ax=ax, 
                              colormap='tab10', edgecolor='white', linewidth=0.3)
    ax.set_ylabel('Número de Imágenes')
    ax.set_xlabel('Clúster')
    ax.set_title(f'Distribución de Clases por Clúster - {model_name}',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Clase', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, output_dir / f"class_distribution_{model_name.lower().replace('-','')}.png")


def main():
    """Pipeline de visualización e interpretación."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 7: VISUALIZACIÓN FINAL E INTERPRETACIÓN")
    print("=" * 70)
    
    # ====================================================================
    # 1. Validar checkpoint anterior
    # ====================================================================
    validate_checkpoint('evaluation')
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=1.0)
    
    # ====================================================================
    # 2. Cargar datos
    # ====================================================================
    with ProcessingTimer("Carga de datos", log_file):
        # PCA data (para las etiquetas verdaderas)
        df_pca = pd.read_csv(DATA_PROCESSED_PATH / FILE_FEATURES_PCA)
        true_labels_raw = df_pca['true_label'].values  # strings
        del df_pca
        gc.collect()
        
        # Convertir labels de string a índice numérico
        label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        true_labels = np.array([label_to_idx.get(str(l), -1) for l in true_labels_raw])
        del true_labels_raw
        
        # Metadata (para rutas de archivos)
        metadata = pd.read_csv(DATA_PROCESSED_PATH / FILE_METADATA)
        # Renombrar columnas para compatibilidad interna
        metadata = metadata.rename(columns={'file_path': 'filepath', 'true_label': 'class_name'})
        
        # Labels de modelos
        kmeans_labels = np.load(MODELS_PATH / FILE_KMEANS_LABELS)
        dbscan_labels = np.load(MODELS_PATH / FILE_DBSCAN_LABELS)
        
        print(f"📊 Imágenes: {len(metadata)}")
        print(f"   K-Means: {len(set(kmeans_labels))} clústeres")
        print(f"   DBSCAN: {len(set(dbscan_labels))} clústeres "
              f"(ruido={np.sum(dbscan_labels == -1)})")
    
    interp_dir = FIGURES_PATH / "05_interpretation"
    interp_dir.mkdir(parents=True, exist_ok=True)
    
    # ====================================================================
    # 3. Muestras de imágenes por clúster - K-Means
    # ====================================================================
    with ProcessingTimer("Muestras de imágenes por clúster (K-Means)", log_file):
        n_km_clusters = len(set(kmeans_labels))
        plot_sample_images_per_cluster(
            metadata, kmeans_labels, "K-Means", interp_dir,
            n_samples=5, n_clusters_show=min(n_km_clusters, 15)
        )
    
    gc.collect()
    
    # ====================================================================
    # 4. Muestras de imágenes por clúster - DBSCAN
    # ====================================================================
    with ProcessingTimer("Muestras de imágenes por clúster (DBSCAN)", log_file):
        n_db_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        plot_sample_images_per_cluster(
            metadata, dbscan_labels, "DBSCAN", interp_dir,
            n_samples=5, n_clusters_show=min(n_db_clusters, 15)
        )
    
    gc.collect()
    
    # ====================================================================
    # 5. Heatmaps de composición de clústeres
    # ====================================================================
    with ProcessingTimer("Heatmaps de composición (K-Means)", log_file):
        comp_km = plot_cluster_composition_heatmap(
            true_labels, kmeans_labels, CLASS_NAMES, "K-Means", interp_dir
        )
        print(f"\n   K-Means composición:\n{comp_km.to_string()}")
    
    gc.collect()
    
    with ProcessingTimer("Heatmaps de composición (DBSCAN)", log_file):
        comp_db = plot_cluster_composition_heatmap(
            true_labels, dbscan_labels, CLASS_NAMES, "DBSCAN", interp_dir
        )
    
    gc.collect()
    
    # ====================================================================
    # 6. Distribución de clases por clúster (barras apiladas)
    # ====================================================================
    with ProcessingTimer("Distribución de clases por clúster (K-Means)", log_file):
        plot_class_distribution_per_cluster(
            true_labels, kmeans_labels, CLASS_NAMES, "K-Means", interp_dir
        )
    
    gc.collect()
    
    with ProcessingTimer("Distribución de clases por clúster (DBSCAN)", log_file):
        plot_class_distribution_per_cluster(
            true_labels, dbscan_labels, CLASS_NAMES, "DBSCAN", interp_dir
        )
    
    gc.collect()
    
    # ====================================================================
    # 7. Resumen final
    # ====================================================================
    with ProcessingTimer("Generación de resumen final", log_file):
        # Cargar métricas
        metrics = pd.read_csv(TABLES_PATH / "metrics_comparison.csv")
        
        print("\n" + "=" * 55)
        print("  RESUMEN FINAL DE MÉTRICAS")
        print("=" * 55)
        print(metrics.to_string(index=False))
        
        # Tabla de interpretación agroecológica
        print(f"\n📊 Interpretación por uso de suelo:")
        for cid in sorted(set(kmeans_labels)):
            mask = kmeans_labels == cid
            true_in_cluster = true_labels[mask]
            # Filtrar posibles -1 de labels no mapeados
            valid = true_in_cluster[true_in_cluster >= 0]
            if len(valid) > 0:
                class_counts = np.bincount(valid, minlength=len(CLASS_NAMES))
                dominant_class = CLASS_NAMES[np.argmax(class_counts)]
                purity = class_counts.max() / class_counts.sum() * 100
                print(f"   K-Means C{cid}: {dominant_class} ({purity:.1f}% pureza)")
            else:
                print(f"   K-Means C{cid}: Sin datos válidos")
    
    # ====================================================================
    # 8. Checkpoint
    # ====================================================================
    del true_labels, kmeans_labels, dbscan_labels, metadata
    gc.collect()
    print_memory_status("Fin del script")
    
    create_checkpoint('visualization', "Todas las figuras e interpretaciones generadas")
    
    print("\n" + "=" * 70)
    print("  ✅ PASO 7 COMPLETADO - VISUALIZACIÓN E INTERPRETACIÓN")
    print("  📁 Figuras en: outputs/figures/05_interpretation/")
    print("  📁 Tablas en: outputs/tables/")
    print("")
    print("  🎉 PIPELINE COMPLETO - TODOS LOS PASOS FINALIZADOS")
    print("=" * 70)


if __name__ == "__main__":
    main()
