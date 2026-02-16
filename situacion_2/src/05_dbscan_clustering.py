#!/usr/bin/env python3
"""
05_dbscan_clustering.py - Modelado DBSCAN
==========================================
FASE 3 - Pasos 3.1 a 3.6

Aplica DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
sobre los datos reducidos de PCA. DBSCAN detecta clústeres de forma arbitra-
ria y marca outliers como ruido (-1), útil para anomalías agroecológicas.

Estrategia para elegir epsilon:
- Usar k-distance graph (distancia al k-ésimo vecino más cercano)
- El "codo" del gráfico indica un buen epsilon.

Input:  data_processed/features_pca_reduced.csv
Output: outputs/models/dbscan_model.pkl
        outputs/models/dbscan_labels.npy
        outputs/figures/03_dbscan/k_distance_graph.png
        outputs/figures/03_dbscan/clusters_dbscan.png
        outputs/tables/dbscan_hyperparameters.csv
        
⚠️ Ejecutar SOLO después de que 04_kmeans_clustering.py haya terminado.
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
    DBSCAN_N_NEIGHBORS_FOR_EPSILON, DBSCAN_MIN_SAMPLES_RANGE,
    FILE_FEATURES_PCA, FILE_DBSCAN_MODEL, FILE_DBSCAN_LABELS, FILE_PROCESSING_LOG,
    FIGSIZE_MEDIUM,
    validate_checkpoint, create_checkpoint
)
from utils.memory_utils import print_memory_status, ProcessingTimer, check_memory_threshold
from utils.plotting_utils import configure_plot_style, save_figure


def main():
    """Pipeline de clustering DBSCAN."""
    
    log_file = DATA_PROCESSED_PATH / FILE_PROCESSING_LOG
    
    print("=" * 70)
    print("  PASO 5: MODELADO DBSCAN (DENSITY-BASED CLUSTERING)")
    print("=" * 70)
    
    # ====================================================================
    # 1. Validar checkpoint anterior
    # ====================================================================
    validate_checkpoint('kmeans')
    print_memory_status("Inicio del script")
    check_memory_threshold(min_available_gb=1.5)
    
    # ====================================================================
    # 2. Cargar datos PCA reducidos
    # ====================================================================
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    
    with ProcessingTimer("Carga de datos PCA", log_file):
        pca_path = DATA_PROCESSED_PATH / FILE_FEATURES_PCA
        df = pd.read_csv(pca_path)
        
        true_labels = df['true_label'].values
        X = df.drop(columns=['true_label']).values.astype(np.float32)
        
        print(f"📊 Datos: {X.shape}")
        del df
        gc.collect()
    
    # ====================================================================
    # 3. Determinar epsilon con k-distance graph
    # ====================================================================
    with ProcessingTimer("Cálculo de k-distance graph para epsilon", log_file):
        
        k_neighbors = DBSCAN_N_NEIGHBORS_FOR_EPSILON
        
        # Usar muestra si el dataset es muy grande (NearestNeighbors es O(n²))
        if X.shape[0] > 10000:
            print(f"   Usando muestra de 10,000 para k-distance (dataset grande)")
            np.random.seed(42)
            sample_idx = np.random.choice(X.shape[0], 10000, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')
        nn.fit(X_sample)
        distances, _ = nn.kneighbors(X_sample)
        
        # Distancia al k-ésimo vecino (ordenar de menor a mayor)
        k_distances = np.sort(distances[:, k_neighbors])
        
        # Generar múltiples valores de epsilon para explorar
        # En datos de alta dimensionalidad, probar percentiles variados
        epsilon_candidates = {
            'P50': np.percentile(k_distances, 50),
            'P60': np.percentile(k_distances, 60),
            'P70': np.percentile(k_distances, 70),
            'P80': np.percentile(k_distances, 80),
            'P90': np.percentile(k_distances, 90),
        }
        
        print(f"📊 k-distances (k={k_neighbors}):")
        print(f"   Min: {k_distances.min():.4f}")
        print(f"   Mediana: {np.median(k_distances):.4f}")
        print(f"   P70: {np.percentile(k_distances, 70):.4f}")
        print(f"   P80: {np.percentile(k_distances, 80):.4f}")
        print(f"   P90: {np.percentile(k_distances, 90):.4f}")
        print(f"   P95: {np.percentile(k_distances, 95):.4f}")
        print(f"   Max: {k_distances.max():.4f}")
        print(f"\n🎯 Epsilon candidatos: {', '.join(f'{k}={v:.2f}' for k,v in epsilon_candidates.items())}")
        
        del X_sample, nn, distances
        gc.collect()
    
    # ====================================================================
    # 4. Generar k-distance graph
    # ====================================================================
    with ProcessingTimer("Generación del k-distance graph", log_file):
        configure_plot_style()
        dbscan_fig_dir = FIGURES_PATH / "03_dbscan"
        dbscan_fig_dir.mkdir(parents=True, exist_ok=True)
        
        fig1, ax1 = plt.subplots(figsize=FIGSIZE_MEDIUM)
        ax1.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
        colors_eps = ['green', 'orange', 'red', 'purple', 'brown']
        for (name, eps_val), color in zip(epsilon_candidates.items(), colors_eps):
            ax1.axhline(y=eps_val, color=color, linestyle='--', linewidth=1.5,
                         label=f'ε {name} = {eps_val:.1f}')
        ax1.set_xlabel('Puntos (ordenados)')
        ax1.set_ylabel(f'Distancia al {k_neighbors}° vecino más cercano')
        ax1.set_title(f'K-Distance Graph (k={k_neighbors}) para Determinar ε\n'
                       f'EuroSAT PCA - Múltiples candidatos de ε')
        ax1.legend(fontsize=10)
        save_figure(fig1, dbscan_fig_dir / "k_distance_graph.png")
        
        del k_distances
        gc.collect()
    
    # ====================================================================
    # 5. Ejecutar DBSCAN con múltiples eps y min_samples
    # ====================================================================
    with ProcessingTimer("Búsqueda de hiperparámetros DBSCAN", log_file):
        
        results = []
        best_silhouette = -1
        best_config = None
        best_labels = None
        best_model = None
        best_score = -999  # composite score para selección
        
        for eps_name, eps_val in epsilon_candidates.items():
            for min_samples in DBSCAN_MIN_SAMPLES_RANGE:
                print(f"\n   eps={eps_val:.2f} ({eps_name}), min_samples={min_samples}: ", end="", flush=True)
                
                dbscan = DBSCAN(
                    eps=eps_val,
                    min_samples=min_samples,
                    metric='euclidean',
                    n_jobs=-1
                )
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = np.sum(labels == -1)
                noise_pct = n_noise / len(labels) * 100
                
                # Calcular silueta (solo si hay >1 cluster y no todo es ruido)
                if n_clusters >= 2:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > n_clusters:
                        sil = silhouette_score(X[non_noise_mask], labels[non_noise_mask], 
                                               sample_size=min(10000, np.sum(non_noise_mask)))
                    else:
                        sil = -1
                else:
                    sil = -1
                
                print(f"Clusters={n_clusters}, Ruido={n_noise} ({noise_pct:.1f}%), Silueta={sil:.4f}")
                
                results.append({
                    'epsilon': eps_val,
                    'epsilon_percentile': eps_name,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percent': noise_pct,
                    'silhouette_score': sil
                })
                
                # Seleccionar mejor configuración para comparación con K-Means:
                # Priorizar: n_clusters razonable (5-15), ruido < 45%, luego mejor silueta
                # Si no hay >=5 clusters, buscar >=3 con buen balance
                is_candidate = False
                priority = 0  # mayor = mejor
                
                if n_clusters >= 5 and noise_pct < 45:
                    priority = 2  # Preferida: clusters en rango razonable
                    is_candidate = True
                elif n_clusters >= 3 and noise_pct < 45:
                    priority = 1  # Aceptable
                    is_candidate = True
                
                if is_candidate:
                    # Composite score: prioridad + silueta normalizada
                    score = priority * 10 + sil
                    if score > best_score:
                        best_score = score
                        best_silhouette = sil
                        best_config = {'eps': eps_val, 'min_samples': min_samples, 'eps_name': eps_name}
                        best_labels = labels.copy()
                        best_model = dbscan
                
                del dbscan, labels
                gc.collect()
        
        # Si no se encontró configuración con >= 3 clusters, tomar el mejor global
        if best_config is None:
            print("\n   ⚠️ No se encontró config con >=3 clusters y <50% ruido.")
            print("   Seleccionando la mejor por silueta global...")
            best_sil_global = -1
            for eps_name, eps_val in epsilon_candidates.items():
                for min_samples in DBSCAN_MIN_SAMPLES_RANGE:
                    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples, metric='euclidean', n_jobs=-1)
                    labels = dbscan.fit_predict(X)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = np.sum(labels == -1)
                    if n_clusters >= 2:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > n_clusters:
                            sil = silhouette_score(X[non_noise_mask], labels[non_noise_mask],
                                                   sample_size=min(10000, np.sum(non_noise_mask)))
                            if sil > best_sil_global:
                                best_sil_global = sil
                                best_silhouette = sil
                                best_config = {'eps': eps_val, 'min_samples': min_samples, 'eps_name': eps_name}
                                best_labels = labels.copy()
                                best_model = dbscan
                    del dbscan, labels
                    gc.collect()
        
        # Guardar resultados de búsqueda
        df_results = pd.DataFrame(results)
        param_path = TABLES_PATH / "dbscan_hyperparameters.csv"
        df_results.to_csv(param_path, index=False)
        print(f"\n💾 Resultados de búsqueda: {param_path}")
        print(f"\n🎯 Mejor configuración: eps={best_config['eps']:.4f} ({best_config['eps_name']}), "
              f"min_samples={best_config['min_samples']}")
    
    # ====================================================================
    # 6. Análisis del mejor modelo DBSCAN
    # ====================================================================
    with ProcessingTimer("Análisis del modelo DBSCAN final", log_file):
        n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise_final = np.sum(best_labels == -1)
        
        print(f"\n📊 DBSCAN Final:")
        print(f"   Clústeres encontrados: {n_clusters_final}")
        print(f"   Imágenes como ruido: {n_noise_final} ({n_noise_final/len(best_labels)*100:.1f}%)")
        print(f"   Silueta (sin ruido): {best_silhouette:.4f}")
        
        print(f"\n   Distribución de clústeres:")
        for cluster_id in sorted(set(best_labels)):
            count = np.sum(best_labels == cluster_id)
            label_name = f"Cluster {cluster_id}" if cluster_id >= 0 else "Ruido (-1)"
            print(f"   {label_name}: {count} imágenes ({count/len(best_labels)*100:.1f}%)")
    
    # ====================================================================
    # 7. Visualización de clústeres DBSCAN
    # ====================================================================
    with ProcessingTimer("Visualización DBSCAN en PCA 2D", log_file):
        
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        
        # Muestra para visualización
        n_sample = min(5000, X.shape[0])
        np.random.seed(42)
        sample_idx = np.random.choice(X.shape[0], n_sample, replace=False)
        
        # Puntos no-ruido
        non_noise = best_labels[sample_idx] != -1
        scatter = ax2.scatter(X[sample_idx][non_noise, 0], X[sample_idx][non_noise, 1],
                              c=best_labels[sample_idx][non_noise], cmap='tab10',
                              alpha=0.4, s=10, edgecolors='none', label='Clústeres')
        
        # Ruido en gris
        noise = best_labels[sample_idx] == -1
        if noise.any():
            ax2.scatter(X[sample_idx][noise, 0], X[sample_idx][noise, 1],
                       c='gray', alpha=0.3, s=5, marker='x', label='Ruido')
        
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title(f"DBSCAN Clustering\n"
                       f"ε={best_config['eps']:.2f} ({best_config.get('eps_name','')}), "
                       f"min_samples={best_config['min_samples']}, "
                       f"clusters={n_clusters_final}, ruido={n_noise_final}")
        ax2.legend(fontsize=11)
        plt.colorbar(scatter, ax=ax2, label='Clúster ID')
        save_figure(fig2, dbscan_fig_dir / "clusters_dbscan.png")
    
    # ====================================================================
    # 8. Guardar modelo y etiquetas
    # ====================================================================
    with ProcessingTimer("Persistencia de modelo y etiquetas DBSCAN", log_file):
        joblib.dump(best_model, MODELS_PATH / FILE_DBSCAN_MODEL)
        np.save(MODELS_PATH / FILE_DBSCAN_LABELS, best_labels)
        print(f"💾 Modelo DBSCAN: {MODELS_PATH / FILE_DBSCAN_MODEL}")
        print(f"💾 Etiquetas: {MODELS_PATH / FILE_DBSCAN_LABELS}")
    
    # ====================================================================
    # 9. Checkpoint
    # ====================================================================
    del X, best_labels, best_model, true_labels
    gc.collect()
    print_memory_status("Fin del script")
    create_checkpoint('dbscan', 
                      f"eps={best_config['eps']:.4f}, min_samples={best_config['min_samples']}, "
                      f"clusters={n_clusters_final}, noise={n_noise_final}")
    
    print("\n" + "=" * 70)
    print(f"  ✅ PASO 5 COMPLETADO - DBSCAN: {n_clusters_final} clústeres, {n_noise_final} ruido")
    print(f"  📊 Silueta: {best_silhouette:.4f}")
    print(f"  📌 Siguiente: Ejecutar 06_evaluation_validation.py")
    print(f"  ⚠️  Esperar a que este script termine antes de continuar")
    print("=" * 70)


if __name__ == "__main__":
    main()
