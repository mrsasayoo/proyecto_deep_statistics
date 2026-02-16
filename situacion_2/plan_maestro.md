# Proyecto 1 - Situación 2: Clustering para Evaluación Agroecológica

## 🎯 Contexto del Problema

Una agencia de gestión agrícola necesita realizar una **evaluación rápida del estado de la cobertura terrestre** en una región vulnerable después de un **evento climático extremo** (sequía o inundación). 

**Desafío:** No poseen etiquetas actualizadas (ground truth) de la zona, pero tienen acceso a miles de imágenes satelitales de alta resolución.

**Objetivo:** Utilizar **clustering no supervisado** para agrupar imágenes en categorías que representen diferentes tipologías de cobertura terrestre (cultivo saludable, bosque denso, suelo desnudo, zona inundada). 

**Hipótesis:** Los clústeres que representen "suelo desnudo", "zona industrial" o "agua estancada" son **indicadores de zonas agroecológicas dañadas o en estrés**.

---

## 📊 Fuente de Datos

**Dataset:** EuroSAT - Land Use and Land Cover Classification with Sentinel-2

- **Total imágenes:** 27,000 imágenes satelitales multiespectrales
- **Resolución:** 64×64 píxeles
- **Bandas espectrales:** 13 bandas del satélite Sentinel-2
- **Tamaño:** ~2.1 GB (versión Multiespectral)
- **10 Clases (solo para validación final):**
  - AnnualCrop (Cultivo Anual)
  - PermanentCrop (Cultivo Permanente)
  - Pasture (Pasto)
  - Forest (Bosque)
  - HerbaceousVegetation (Vegetación Herbácea)
  - Industrial (Zona Industrial)
  - Residential (Zona Residencial)
  - SeaLake (Mar/Lago)
  - River (Río)
  - Highway (Carretera)

**⚠️ Importante:** Las etiquetas reales solo se utilizarán al final para **validación externa**, manteniendo el enfoque no supervisado del clustering.

---

## 🗂️ Organización del Proyecto

Dado que vas a manejar el dataset **Multiespectral (MS)** de 2.1 GB, la organización es clave para no saturar la memoria RAM y para que tu flujo de trabajo sea profesional.

Aquí tienes la propuesta de estructura de directorios y el plan granular paso a paso.

---

### 1. Estructura de Directorios Completa y Predicha

Esta estructura muestra TODOS los archivos y carpetas que se generarán durante el proyecto:

```text
situacion_2/
├── dataset/
│   ├── EuroSAT_MS/                    # Datos crudos (.tif) - 2.1 GB
│   │   ├── AnnualCrop/                # ~3,000 imágenes .tif
│   │   ├── Forest/
│   │   ├── HerbaceousVegetation/
│   │   ├── Highway/
│   │   ├── Industrial/
│   │   ├── Pasture/
│   │   ├── PermanentCrop/
│   │   ├── Residential/
│   │   ├── River/
│   │   └── SeaLake/
│   └── EuroSAT_MS.zip                 # Archivo original (backup)
│
├── data_processed/                    # ⚡ Datos procesados listos para usar
│   ├── .checkpoints/                  # 🔒 Control de ejecución secuencial
│   │   ├── 01_data_loading.done
│   │   ├── 02_normalization.done
│   │   ├── 03_pca_reduction.done
│   │   ├── 04_kmeans_clustering.done
│   │   ├── 05_dbscan_clustering.done
│   │   └── 06_evaluation_validation.done
│   ├── metadata_labels.csv            # [image_id, true_label, file_path]
│   ├── features_raw_flattened.npy     # Matriz (27000 x 53248) - ~10 GB [OPCIONAL]
│   ├── features_normalized.npy        # Matriz normalizada (27000 x 53248)
│   ├── features_pca_reduced.csv       # ⭐ Matriz reducida (27000 x m) - LIGERA
│   ├── pca_variance_explained.csv     # Varianza por componente
│   └── processing_log.txt             # Log de tiempos de procesamiento
│
├── src/                               # 📜 Scripts de procesamiento secuencial
│   ├── config.py                      # Configuración global (paths, constantes)
│   ├── 01_data_loading.py             # Carga y aplanamiento de imágenes
│   ├── 02_normalization.py            # Normalización con StandardScaler
│   ├── 03_pca_reduction.py            # Reducción dimensional con PCA
│   ├── 04_kmeans_clustering.py        # Modelado K-Means con optimización k
│   ├── 05_dbscan_clustering.py        # Modelado DBSCAN con GridSearch
│   ├── 06_evaluation_validation.py    # Métricas ARI, NMI, Silueta
│   ├── 07_visualization_export.py     # Generación de todas las figuras
│   └── utils/
│       ├── __init__.py
│       ├── image_loader.py            # Funciones de carga eficiente
│       ├── memory_utils.py            # Gestión de memoria y liberación
│       └── plotting_utils.py          # Funciones de visualización
│
├── notebooks/
│   ├── situacion_2.ipynb              # 📓 Notebook principal integrado
│   ├── 01_exploratory_analysis.ipynb  # EDA de imágenes (opcional)
│   └── 02_debug_testing.ipynb         # Testing con subset pequeño
│
├── outputs/                           # 📊 Resultados finales
│   ├── models/                        # Modelos entrenados persistidos
│   │   ├── pca_model.pkl              # Modelo PCA (para reproducibilidad)
│   │   ├── scaler_model.pkl           # StandardScaler (para nuevos datos)
│   │   ├── kmeans_model.pkl           # Modelo K-Means final
│   │   ├── dbscan_model.pkl           # Modelo DBSCAN final
│   │   ├── kmeans_labels.npy          # Etiquetas de clúster K-Means
│   │   └── dbscan_labels.npy          # Etiquetas de clúster DBSCAN
│   │
│   ├── figures/                       # 🎨 Gráficos para el informe
│   │   ├── 01_pca/
│   │   │   ├── variance_explained_cumulative.png
│   │   │   ├── scree_plot.png
│   │   │   └── pca_2d_projection.png  # Proyección primeros 2 componentes
│   │   ├── 02_kmeans/
│   │   │   ├── elbow_plot_sse.png     # Gráfico del Codo
│   │   │   ├── silhouette_scores.png  # Coeficientes de Silueta por k
│   │   │   ├── clusters_pca_space.png # Visualización en espacio PCA
│   │   │   └── sample_images_per_cluster/ # Imágenes representativas
│   │   │       ├── cluster_0_samples.png
│   │   │       ├── cluster_1_samples.png
│   │   │       └── ... (hasta cluster_k)
│   │   ├── 03_dbscan/
│   │   │   ├── k_distance_graph.png   # Gráfico para determinar epsilon
│   │   │   ├── clusters_dbscan.png    # Clústeres encontrados
│   │   │   ├── outliers_analysis.png  # Visualización de anomalías
│   │   │   └── noise_sample_images.png # Imágenes clasificadas como ruido
│   │   ├── 04_evaluation/
│   │   │   ├── confusion_matrix_kmeans.png
│   │   │   ├── confusion_matrix_dbscan.png
│   │   │   ├── confusion_matrix_kmeans_normalized.png
│   │   │   ├── confusion_matrix_dbscan_normalized.png
│   │   │   └── metrics_comparison_table.png # ARI, NMI, Silueta
│   │   └── 05_interpretation/
│   │       ├── cluster_composition_heatmap.png
│   │       ├── stress_zones_distribution.png
│   │       └── crops_vs_forests_separation.png
│   │
│   ├── tables/                        # 📋 Tablas cuantitativas
│   │   ├── pca_components_variance.csv
│   │   ├── kmeans_optimization_results.csv
│   │   ├── dbscan_hyperparameters.csv
│   │   ├── metrics_comparison.csv     # ARI, NMI, Silueta
│   │   └── cluster_characterization.csv # Composición de cada clúster
│   │
│   └── reports/                       # 📄 Documentos finales
│       ├── informe_situacion_2.pdf    # Informe final (<25 páginas)
│       ├── presentacion.pptx          # Presentación opcional
│       └── README_RESULTADOS.md       # Resumen ejecutivo
│
├── docs/                              # 📚 Documentación del proyecto
│   ├── metodologia.md                 # Justificación técnica
│   └── referencias.bib                # Referencias bibliográficas
│
├── plan_maestro.md                    # 🎯 Este archivo - Guía completa
├── README.md                          # Descripción general del proyecto
└── requirements.txt                   # Dependencias Python
```

---

### 🔒 Sistema de Control de Ejecución Secuencial

**Carpeta crítica:** `data_processed/.checkpoints/`

Cada script de procesamiento crea un archivo `.done` al finalizar exitosamente. Los scripts subsecuentes validan la existencia de estos archivos antes de ejecutarse.

**Flujo de validación:**
```
01_data_loading.py → crea 01_data_loading.done
02_normalization.py → valida 01_data_loading.done → crea 02_normalization.done
03_pca_reduction.py → valida 02_normalization.done → crea 03_pca_reduction.done
... y así sucesivamente
```

**⚠️ REGLA CRÍTICA:** Nunca ejecutar dos scripts de procesamiento simultáneamente para evitar saturación de RAM.

---

### 📊 Tamaños Estimados de Archivos

| Archivo | Tamaño Aproximado | Descripción |
|---------|-------------------|-------------|
| `dataset/EuroSAT_MS/` | ~2.1 GB | Datos crudos originales |
| `features_raw_flattened.npy` | ~10 GB | Matriz completa aplanada (opcional) |
| `features_normalized.npy` | ~10 GB | Matriz normalizada |
| `features_pca_reduced.csv` | ~50-200 MB | ⭐ Matriz reducida (ligera) |
| Modelos `.pkl` | ~10-100 MB cada uno | Modelos entrenados |
| Figuras `.png` | ~500 KB - 2 MB cada una | Gráficos de alta resolución |

**Total espacio requerido:** ~25-30 GB (incluyendo datos temporales)

---

### 2. Plan Granular Paso a Paso

He dividido el plan en **5 fases** que siguen la lógica del taller:

## ⚡ DIAGRAMA DE FLUJO DE EJECUCIÓN SECUENCIAL

```
┌─────────────────────────────────────────────────────────────────┐
│  🔒 PIPELINE SECUENCIAL - EJECUCIÓN OBLIGATORIA EN ORDEN        │
│  ⚠️  NO ejecutar pasos en paralelo (saturación de RAM)          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 1: 01_data_loading.py                    │
    │ Carga 27,000 imágenes .tif (13 bandas)        │
    │ Output: features_raw_flattened.npy (10 GB)    │
    │ Tiempo: ~15-30 min                             │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 01_data_loading.done
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 2: 02_normalization.py                   │
    │ Normaliza con StandardScaler                  │
    │ Output: features_normalized.npy               │
    │ Tiempo: ~5-10 min                              │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 02_normalization.done
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 3: 03_pca_reduction.py                   │
    │ PCA: 53,248 → ~50-200 componentes             │
    │ Output: features_pca_reduced.csv (ligero)     │
    │ Tiempo: ~10-20 min                             │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 03_pca_reduction.done
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 4: 04_kmeans_clustering.py               │
    │ K-Means con k=2..15, Codo + Silueta           │
    │ Output: kmeans_model.pkl + labels             │
    │ Tiempo: ~20-40 min                             │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 04_kmeans_clustering.done
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 5: 05_dbscan_clustering.py               │
    │ DBSCAN con optimización de epsilon            │
    │ Output: dbscan_model.pkl + labels             │
    │ Tiempo: ~15-30 min                             │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 05_dbscan_clustering.done
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 6: 06_evaluation_validation.py           │
    │ Matrices confusión, ARI, NMI, Silueta         │
    │ Output: tablas de métricas                    │
    │ Tiempo: ~5-10 min                              │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 06_evaluation_validation.done
                            ▼
    ┌───────────────────────────────────────────────┐
    │ PASO 7: 07_visualization_export.py            │
    │ Genera todas las figuras para informe         │
    │ Output: ~20 gráficos en outputs/figures/      │
    │ Tiempo: ~10-20 min                             │
    └───────────────────────────────────────────────┘
                            │
                            │ ✓ checkpoint: 07_visualization_export.done
                            ▼
            ┌───────────────────────────────┐
            │ ✅ PIPELINE COMPLETADO         │
            │ Total: ~80-160 minutos         │
            │ Listo para redactar informe   │
            └───────────────────────────────┘
```

### 🎯 Ejecución Automática del Pipeline

**Opción 1: Script Master (Recomendado)**
```bash
python run_pipeline.py
```

Este script:
- ✅ Ejecuta todos los pasos EN ORDEN automáticamente
- ✅ Valida checkpoints antes de cada paso
- ✅ Pausa 5 segundos entre pasos para liberar RAM
- ✅ Muestra tiempo estimado y progreso
- ✅ Permite continuar desde donde quedó si falla

**Opción 2: Ejecución Manual Paso a Paso**
```bash
python src/01_data_loading.py
# Esperar a que termine completamente
python src/02_normalization.py
# Esperar a que termine completamente
python src/03_pca_reduction.py
# ... y así sucesivamente
```

⚠️ **NUNCA ejecutar dos scripts simultáneamente** (ej: abrir dos terminales).

---

### 2.1. Detalles de Cada Fase

#### Fase 1: Ingeniería de Datos y Reducción Dimensional ($n \times p \to n \times m$)

**Objetivo:** Transformar 27,000 imágenes multiespectrales de alta dimensionalidad en una matriz compacta lista para clustering.

*   **Paso 1.1:** **Lectura eficiente de imágenes `.tif`:**
    *   Usar `tifffile` o `rasterio` para leer las 13 bandas espectrales.
    *   Implementar procesamiento por lotes para no saturar la RAM (~1,000 imágenes por lote).
    *   Verificar dimensiones: cada imagen debe ser $(64 \times 64 \times 13)$.
    
*   **Paso 1.2:** **Extracción de metadatos:**
    *   Extraer etiquetas reales desde los nombres de carpetas (AnnualCrop, Forest, etc.).
    *   **Importante:** Estas etiquetas NO se usan para el clustering, solo para validación final.
    *   Guardar en un DataFrame: `[image_id, true_label, file_path]`.
    
*   **Paso 1.3:** **Aplanamiento (Flattening):**
    *   Aplanar cada imagen de $(64 \times 64 \times 13)$ a un vector de $p = 53,248$ elementos.
    *   Resultado: Matriz de características $(n \times p)$ donde $n \approx 27,000$ y $p = 53,248$.
    
*   **Paso 1.4:** **Normalización:**
    *   Aplicar `StandardScaler` de scikit-learn sobre la matriz completa.
    *   Razón: Los valores de reflectancia satelital varían significativamente entre bandas (visibles vs. infrarrojas).
    *   La normalización es **crítica** para que PCA funcione correctamente.
    
*   **Paso 1.5:** **Reducción de Dimensionalidad (PCA):**
    *   Aplicar PCA para reducir de $p = 53,248$ a $m$ componentes.
    *   Criterio: Retener componentes que expliquen >90% de la varianza acumulada.
    *   Meta esperada: $m \approx 50-200$ componentes (reducción drástica de dimensionalidad).
    *   Generar gráfico de varianza explicada acumulada.
    
*   **Paso 1.6:** **Persistencia de datos procesados:**
    *   Guardar matriz reducida $(n \times m)$ en `data_processed/features_pca.csv`.
    *   Guardar también el modelo PCA en `outputs/models/pca_model.pkl` para reproducibilidad.
    *   **Beneficio:** A partir de aquí, trabajas con datos ligeros y el procesamiento es instantáneo.

#### Fase 2: Modelado de Clustering - K-Means
*   **Paso 2.1:** **Optimización de K:** Ejecutar K-means para $k$ de 2 a 15.
*   **Paso 2.2:** Generar **Gráfico del Codo (SSE)** - Suma de Errores Cuadráticos dentro de cada clúster.
*   **Paso 2.3:** Generar **Gráfico de Coeficiente de Silueta** - Métrica clave para determinar la calidad de la separación entre clústeres.
*   **Paso 2.4:** **Elegir el $k$ óptimo** basado en los gráficos. **Hipótesis del proyecto:** $k$ óptimo entre 8 a 12 clústeres (correspondiente aproximadamente al número de clases reales).
*   **Paso 2.5:** Ajustar el modelo K-Means final con el $k$ óptimo y asignar etiquetas de clúster a cada imagen.
*   **Paso 2.6:** Guardar el modelo entrenado en `outputs/models/kmeans_model.pkl`.

#### Fase 3: Modelado de Clustering - DBSCAN (Density-Based Spatial Clustering)
*   **Paso 3.1:** **Búsqueda de hiperparámetros óptimos:**
    *   `epsilon` ($\epsilon$): Distancia máxima entre dos puntos para ser considerados vecinos. *Pista: usar el método de la rodilla (k-distance graph) con vecinos cercanos para determinar $\epsilon$.*
    *   `min_samples`: Número mínimo de puntos en un vecindario para formar un clúster denso.
*   **Paso 3.2:** **Justificación de hiperparámetros:** Documentar cómo se eligieron $\epsilon$ y `min_samples` basándose en la naturaleza de los datos espectrales.
*   **Paso 3.3:** Ejecutar DBSCAN sobre los datos reducidos de PCA.
*   **Paso 3.4:** **Análisis de Resultados:** 
    *   ¿Cuántos clústeres detecta DBSCAN?
    *   ¿Identifica muchas imágenes como `-1` (ruido/anomalías)?
*   **Paso 3.5:** **Análisis de Ruido (Outliers):** Identificar y visualizar imágenes marcadas como anomalías. Pregunta clave: ¿Son nubes, errores de sensor, o cobertura terrestre anómala?
*   **Paso 3.6:** Guardar el modelo y las asignaciones de clústeres en `outputs/models/dbscan_model.pkl`.

#### Fase 4: Evaluación y Validación Externa
*   **Paso 4.1:** **Validación Interna:** 
    *   Calcular y comparar el **Coeficiente de Silueta** promedio entre K-Means y DBSCAN.
    *   El modelo con mayor Coeficiente de Silueta tiene mejor separación interna entre clústeres.
*   **Paso 4.2:** **Validación Externa (Usando las Etiquetas Ocultas):**
    *   Ahora, y **solo para validar**, utilice las etiquetas reales del dataset EuroSAT (AnnualCrop, Industrial, Forest, etc.).
    *   **Matriz de Confusión:** Cruzar los $k$ clústeres encontrados (Cluster 0, Cluster 1...) con las 10 clases reales.
    *   Generar matrices de confusión separadas para K-Means y DBSCAN.
*   **Paso 4.3:** **Métricas de Concordancia:**
    *   **ARI (Adjusted Rand Index):** Mide la similitud entre las agrupaciones encontradas y las etiquetas reales, ajustado por el azar. Rango: [-1, 1], donde 1 = concordancia perfecta.
    *   **NMI (Normalized Mutual Information):** Cuantifica la información compartida entre clústeres y clases reales. Rango: [0, 1], donde 1 = concordancia perfecta.
    *   Reportar ARI y NMI para ambos algoritmos.
*   **Paso 4.4:** **Selección del Mejor Modelo:** Comparar K-Means vs DBSCAN usando:
    *   Coeficiente de Silueta (validación interna)
    *   ARI y NMI (validación externa)
    *   Capacidad de detectar anomalías (ventaja de DBSCAN)

#### Fase 5: Perfilado e Interpretación Agroecológica (Conclusión del Proyecto)
*   **Paso 5.1:** **Análisis de Composición de Clústeres:**
    *   Para cada clúster, analizar qué clases reales predominan (usando la matriz de confusión).
    *   Crear tabla resumen: Clúster → Clases dominantes → Interpretación.
*   **Paso 5.2:** **Identificación de Zonas en Estrés Agroecológico:**
    *   Responder: ¿Qué clústeres representan **"suelo desnudo"**, **"zona industrial"** o **"agua estancada"**?
    *   Estos clústeres son **indicadores de zonas agroecológicas dañadas o en estrés**.
    *   ¿Cuántas imágenes de la región están en estos clústeres de alto riesgo?
*   **Paso 5.3:** **Separación Cultivos vs. Bosques:**
    *   Pregunta clave del PDF: ¿Se separaron claramente los **Cultivos** (AnnualCrop, PermanentCrop) de los **Bosques** (Forest)?
    *   Gracias a las **13 bandas espectrales** de Sentinel-2, esta separación debería ser muy clara en los clústeres.
*   **Paso 5.4:** **Ventaja del Enfoque Multiespectral:**
    *   Discutir cómo las 13 bandas (incluyendo infrarrojo cercano e infrarrojo de onda corta) permitieron detectar diferencias sutiles que serían invisibles en imágenes RGB convencionales.
*   **Paso 5.5:** **Conclusión Agroecológica Final:**
    *   Sintetizar hallazgos para la agencia: 
        - ¿Qué porcentaje del territorio está en estado saludable vs. en estrés?
        - ¿Qué regiones requieren intervención inmediata post-evento climático?
        - ¿El clustering no supervisado fue efectivo para este diagnóstico rápido sin ground truth actualizado?

---

### 3. Requisitos Técnicos

**Librerías Necesarias:**
```bash
pip install tifffile scikit-learn pandas numpy matplotlib seaborn
pip install rasterio  # Alternativa robusta para leer GeoTIFF
```

**Consideraciones de Rendimiento:**
- **Memoria RAM:** Los 2.1 GB de imágenes no se cargan todos a la vez. Procesamiento por lotes (batch processing).
- **Normalización obligatoria:** Los valores de reflectancia satelital varían mucho entre bandas → usar `StandardScaler` antes de PCA.
- **Dimensionalidad inicial:** Cada imagen es $(64 \times 64 \times 13) = 53,248$ características por imagen.
- **Meta de PCA:** Reducir a ~50-200 componentes que capturen >90% de varianza.

---

### 4. Resultados Esperados del Proyecto

El proyecto debe culminar con un **Jupyter Notebook** (o informe en PDF) que incluya:

1. **Visualizaciones de calidad:**
   - Gráfico del Codo y Coeficiente de Silueta para K-Means
   - K-distance graph para DBSCAN
   - Matrices de Confusión (clústeres vs. clases reales)
   - Montajes de imágenes representativas de cada clúster

2. **Métricas cuantitativas:**
   - Coeficiente de Silueta (K-Means y DBSCAN)
   - ARI y NMI (validación externa)
   - Porcentaje de varianza explicada por PCA

3. **Interpretación Agroecológica:**
   - Identificación clara de clústeres que representan zonas en estrés
   - Análisis sobre la separabilidad de cultivos vs. bosques
   - Recomendaciones para la agencia de gestión agrícola

---

### 5. Próximos Pasos para Comenzar

**Paso Inmediato:** Comenzar con la **Fase 1 - Ingeniería de Datos**

**¿Necesitas ayuda con:** 
- ¿Código para cargar las 13 bandas de las imágenes `.tif` de forma eficiente (Paso 1.1 y 1.2)?
- ¿Script para aplanar, normalizar y aplicar PCA sin saturar la RAM (Pasos 1.3 a 1.6)?
- ¿Estructura del código para procesar las 27,000 imágenes por lotes?

**Recomendación:** Trabaja primero con un subconjunto de ~1,000 imágenes para validar el pipeline completo antes de procesar las 27,000 imágenes.

---

### 6. Estructura del Informe Final

**Formato de Entrega:**
- **Informe en PDF:** Máximo 25 páginas (incluyendo imágenes y tablas)
- **Código:** Jupyter Notebook (.ipynb) o scripts de Python (.py)
- **Empaquetado:** Archivo ZIP conteniendo el informe PDF y todos los códigos ejecutables

**Secciones Recomendadas del Informe:**

1. **Introducción (1-2 páginas)**
   - Contexto del problema agroecológico
   - Descripción del dataset EuroSAT
   - Objetivos del proyecto

2. **Metodología (3-4 páginas)**
   - Preprocesamiento y reducción dimensional (PCA)
   - Algoritmos de clustering utilizados (K-Means y DBSCAN)
   - Justificación de hiperparámetros

3. **Resultados (10-12 páginas)**
   - **Fase 1:** Varianza explicada por PCA
   - **Fase 2:** Gráficos del Codo y Silueta para K-Means
   - **Fase 3:** Resultados de DBSCAN y análisis de outliers
   - **Fase 4:** Matrices de confusión, ARI, NMI
   - Incluir código relevante y bien comentado

4. **Interpretación Agroecológica (4-6 páginas)**
   - Caracterización de cada clúster
   - Identificación de zonas en estrés
   - Análisis cultivos vs. bosques
   - Implicaciones para la agencia de gestión agrícola

5. **Conclusiones y Recomendaciones (2-3 páginas)**
   - Efectividad del clustering no supervisado
   - Ventajas de usar datos multiespectrales (13 bandas)
   - Recomendaciones para intervenciones post-desastre

6. **Anexos**
   - Código completo (si no está integrado en las secciones)
   - Tablas complementarias
   - Imágenes adicionales de clústeres representativos

**⚠️ Recordatorio:** Siempre interpretar los resultados en el contexto agroecológico, no solo reportar números.

---

### 7. Preguntas Clave a Responder en el Informe

Estas son las preguntas centrales derivadas del PDF que deben responderse:

1. **Sobre la reducción dimensional:**
   - ¿Cuántos componentes de PCA se necesitaron para explicar >90% de la varianza?
   - ¿Qué porcentaje de reducción dimensional se logró? ($53,248 \to m$)

2. **Sobre K-Means:**
   - ¿Cuál es el número óptimo de clústeres $k$ según el método del codo y la Silueta?
   - ¿El $k$ óptimo está en el rango esperado de 8-12?
   - ¿Qué clústeres de K-Means representan zonas en estrés agroecológico?

3. **Sobre DBSCAN:**
   - ¿Cómo se determinaron $\epsilon$ y `min_samples`?
   - ¿Cuántos clústeres encontró DBSCAN?
   - ¿Qué porcentaje de imágenes fueron clasificadas como ruido/anomalías (`-1`)?
   - ¿Estas anomalías corresponden a nubes, errores de sensor o coberturas anómalas?

4. **Sobre la validación:**
   - ¿Qué algoritmo tuvo mejor Coeficiente de Silueta?
   - ¿Qué algoritmo tuvo mayor ARI y NMI al comparar con las etiquetas reales?
   - ¿La validación externa confirma la calidad del clustering no supervisado?

5. **Sobre la interpretación agroecológica:**
   - ¿Qué clústeres corresponden a "suelo desnudo", "zona industrial" y "agua estancada"?
   - ¿Se separaron claramente los **Cultivos** de los **Bosques**?
   - ¿Qué proporción del territorio está en estado saludable vs. en estrés?
   - ¿El enfoque multiespectral (13 bandas) fue ventajoso vs. imágenes RGB tradicionales?

---

### 8. Referencias y Recursos Adicionales

**Dataset:**
- EuroSAT: [https://github.com/phelber/eurosat](https://github.com/phelber/eurosat)
- Paper: *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification* (Helber et al., 2019)

**Documentación Técnica:**
- Sentinel-2 Bands: [ESA Sentinel-2 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- Scikit-learn Clustering: [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)

**Métricas de Evaluación:**
- Adjusted Rand Index (ARI): Mide similitud entre particiones ajustando por azar
- Normalized Mutual Information (NMI): Cuantifica información compartida entre agrupaciones
- Silhouette Coefficient: Evalúa cohesión intra-cluster y separación inter-cluster

**Conceptos Clave:**
- **Clustering No Supervisado:** Agrupación sin etiquetas previas, ideal para escenarios post-desastre
- **PCA para datos espectrales:** Reducción dimensional extrayendo patrones dominantes de reflectancia
- **13 Bandas de Sentinel-2:** Incluyen infrarrojo cercano (NIR) e infrarrojo de onda corta (SWIR), críticos para diferenciar vegetación saludable de suelo desnudo

---

## ✅ Checklist de Completitud del Proyecto

Usa esta lista para verificar que has completado todos los elementos requeridos:

- [ ] **Fase 1 completada:** Datos reducidos con PCA guardados en `data_processed/`
- [ ] **Fase 2 completada:** K-Means ejecutado con gráficos del Codo y Silueta
- [ ] **Fase 3 completada:** DBSCAN con justificación de hiperparámetros
- [ ] **Fase 4 completada:** Matrices de confusión, ARI y NMI calculados
- [ ] **Fase 5 completada:** Interpretación agroecológica documentada
- [ ] **Código limpio y comentado:** Jupyter Notebook funcional
- [ ] **Visualizaciones de calidad:** Todas las figuras exportadas en `outputs/figures/`
- [ ] **Informe PDF:** Máximo 25 páginas con interpretaciones contextualizadas
- [ ] **Empaquetado:** Archivo ZIP con informe + código listo para entrega
- [ ] **Revisión final:** Todas las preguntas clave respondidas

---

## 📋 Criterios de Evaluación

**Situación 2 vale 0.75 puntos del Proyecto 1. Aspectos a evaluar:**

### 1. Metodología Técnica (30%)
- **Reducción dimensional apropiada:** PCA correctamente aplicado con normalización previa
- **Selección de algoritmos:** Justificación de uso de K-Means y DBSCAN
- **Optimización de hiperparámetros:** Proceso documentado para elegir $k$, $\epsilon$ y `min_samples`

### 2. Implementación y Código (25%)
- **Código funcional:** Jupyter Notebook ejecutable sin errores
- **Eficiencia:** Manejo adecuado de grandes volúmenes de datos (2.1 GB)
- **Reproducibilidad:** Datos procesados guardados, modelos persistidos
- **Documentación:** Código bien comentado y organizado

### 3. Análisis y Visualizaciones (25%)
- **Gráficos obligatorios:** Codo, Silueta, Matriz de Confusión
- **Calidad visual:** Gráficos profesionales con títulos, ejes etiquetados, leyendas
- **Métricas cuantitativas:** ARI, NMI, Coeficiente de Silueta reportados correctamente

### 4. Interpretación Contextual (20%)
- **Enfoque agroecológico:** Interpretación de clústeres en términos de cobertura terrestre
- **Identificación de zonas en estrés:** Respuesta clara a la pregunta del proyecto
- **Conclusiones fundamentadas:** Hallazgos respaldados por evidencia cuantitativa

---

## 💡 Mejores Prácticas y Recomendaciones

### Para la Fase 1 (Procesamiento):
- ✅ **Trabajar primero con un subset pequeño** (~1,000 imágenes) para validar el pipeline
- ✅ Verificar que todas las imágenes tienen las mismas dimensiones (64×64×13)
- ✅ Documentar el tiempo de procesamiento de cada fase
- ❌ No cargar las 27,000 imágenes en RAM simultáneamente

### Para las Fases 2-3 (Clustering):
- ✅ Probar múltiples valores de $k$ (no solo uno)
- ✅ Visualizar clústeres en el espacio PCA (primeros 2-3 componentes)
- ✅ Guardar los modelos entrenados para reproducibilidad
- ❌ No aplicar clustering sobre datos sin normalizar

### Para la Fase 4 (Evaluación):
- ✅ Generar matrices de confusión normalizadas (porcentajes)
- ✅ Calcular todas las métricas (Silueta, ARI, NMI) para ambos modelos
- ✅ Comparar resultados de forma objetiva con tabla resumen
- ❌ No confundir validación interna (Silueta) con externa (ARI, NMI)

### Para el Informe:
- ✅ Incluir código relevante directamente en el informe (no solo en anexo)
- ✅ Cada gráfico debe tener un análisis textual asociado
- ✅ Usar terminología técnica correcta (clúster, dimensionalidad, reflectancia espectral)
- ❌ No exceder las 25 páginas
- ❌ No solo reportar números sin interpretación contextual

---

## 🚀 ¿Listo para Comenzar?

**Siguiente acción inmediata:**  
Comenzar con la **Fase 1, Paso 1.1** - Carga eficiente de las imágenes `.tif`

**¿Necesitas soporte de código?** Pregunta por ayuda específica en cualquier paso del plan.