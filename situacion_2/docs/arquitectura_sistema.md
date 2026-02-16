# Arquitectura del Proyecto - Situación 2

## 📐 Diagrama de Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROYECTO SITUACIÓN 2                             │
│              Clustering Multiespectral para Análisis Agroecológico     │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │         DATOS DE ENTRADA (2.1 GB)             │
        │   27,000 imágenes .tif (64×64×13 bandas)      │
        │   dataset/EuroSAT_MS/[10 clases]/            │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     CAPA DE PROCESAMIENTO                              │
│                   (Ejecución Secuencial Estricta)                      │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MÓDULO 1: Ingeniería de Datos                               │    │
│  │  • 01_data_loading.py    → Carga + Aplanamiento              │    │
│  │  • 02_normalization.py   → StandardScaler                    │    │
│  │  Output: features_normalized.npy (10 GB)                     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                            │                                           │
│                            ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MÓDULO 2: Reducción Dimensional                             │    │
│  │  • 03_pca_reduction.py   → PCA (53,248 → ~100)               │    │
│  │  Output: features_pca_reduced.csv (50-200 MB) ⭐              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                            │                                           │
│                            ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MÓDULO 3: Modelado de Clustering                            │    │
│  │  • 04_kmeans_clustering.py   → K-Means (k=2..15)             │    │
│  │  • 05_dbscan_clustering.py   → DBSCAN + GridSearch           │    │
│  │  Output: modelos .pkl + labels .npy                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                            │                                           │
│                            ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MÓDULO 4: Evaluación y Validación                           │    │
│  │  • 06_evaluation_validation.py                               │    │
│  │    - Silueta (validación interna)                            │    │
│  │    - ARI, NMI (validación externa)                           │    │
│  │    - Matrices de confusión                                   │    │
│  │  Output: tablas .csv con métricas                            │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                            │                                           │
│                            ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MÓDULO 5: Visualización y Exportación                       │    │
│  │  • 07_visualization_export.py                                │    │
│  │    - ~20 gráficos de alta resolución                         │    │
│  │    - Figuras para informe académico                          │    │
│  │  Output: figuras .png (300 DPI)                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │         SALIDAS FINALES (outputs/)            │
        ├───────────────────────────────────────────────┤
        │  • models/     → 6 modelos .pkl               │
        │  • figures/    → ~20 gráficos                 │
        │  • tables/     → Métricas cuantitativas       │
        │  • reports/    → Informe PDF (<25 págs)       │
        └───────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE CONTROL SECUENCIAL                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  data_processed/.checkpoints/                                           │
│  ├── 01_data_loading.done         ← Validado por script 02             │
│  ├── 02_normalization.done        ← Validado por script 03             │
│  ├── 03_pca_reduction.done        ← Validado por script 04             │
│  ├── 04_kmeans_clustering.done    ← Validado por script 05             │
│  ├── 05_dbscan_clustering.done    ← Validado por script 06             │
│  ├── 06_evaluation_validation.done← Validado por script 07             │
│  └── 07_visualization_export.done ← Pipeline completo                  │
│                                                                          │
│  🔒 Cada script verifica el checkpoint anterior antes de ejecutar       │
│  ⚠️  Si falta un checkpoint → ERROR: "Ejecutar primero script XX"       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo de Datos Detallado

### Transformaciones Dimensionales

```
Entrada Original:
27,000 imágenes × (64 × 64 × 13)
= Matriz de 27,000 × 53,248 elementos
≈ 10 GB en memoria


        │  Paso 1: Aplanamiento
        ▼

Matriz Raw Flattened:
numpy.ndarray (27000, 53248)
dtype: float32
Tamaño: ~10 GB


        │  Paso 2: Normalización
        ▼

Matriz Normalizada:
numpy.ndarray (27000, 53248)
Media = 0, Std = 1 por columna
Tamaño: ~10 GB


        │  Paso 3: PCA (Reducción crítica)
        ▼

Matriz PCA Reducida:
pandas.DataFrame (27000, ~100)
Varianza explicada: >90%
Tamaño: ~50 MB ⭐ (200x más ligera)


        │  Paso 4-5: Clustering
        ▼

Asignaciones de Clústeres:
numpy.ndarray (27000,)
dtype: int32
Etiquetas: 0, 1, 2, ..., k
Tamaño: ~100 KB
```

---

## 💾 Gestión de Memoria por Paso

| Paso | RAM Pico | Duración | Archivos Generados | Liberación |
|------|----------|----------|-------------------|------------|
| 01 - Carga | ~12 GB | 15-30 min | features_raw.npy (10 GB) | ✅ Al finalizar |
| 02 - Normalización | ~15 GB | 5-10 min | features_normalized.npy (10 GB) | ✅ Al finalizar |
| 03 - PCA | ~12 GB | 10-20 min | features_pca.csv (50 MB) | ✅ Al finalizar |
| 04 - K-Means | ~2 GB | 20-40 min | kmeans_model.pkl | ✅ Entre iteraciones |
| 05 - DBSCAN | ~3 GB | 15-30 min | dbscan_model.pkl | ✅ Al finalizar |
| 06 - Evaluación | ~500 MB | 5-10 min | tablas .csv | - |
| 07 - Visualización | ~1 GB | 10-20 min | figuras .png | - |

**RAM mínima recomendada:** 16 GB  
**RAM óptima:** 32 GB  

⚠️ Si tienes menos RAM: Reduce `BATCH_SIZE` en `src/config.py`

---

## 🔐 Mecanismos de Seguridad

### 1. Validación de Checkpoints

Cada script comienza con:
```python
from config import validate_checkpoint

# Validar paso previo
try:
    validate_checkpoint('nombre_paso_anterior')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)
```

### 2. Liberación Explícita de Memoria

Cada script termina con:
```python
import gc

# Liberar variables pesadas
del features_matrix
del image_data
gc.collect()

# Crear checkpoint
create_checkpoint('nombre_paso_actual')
```

### 3. Procesamiento por Lotes

```python
BATCH_SIZE = 1000  # Configurable

for i in range(0, n_images, BATCH_SIZE):
    batch = load_batch(i, i+BATCH_SIZE)
    process_batch(batch)
    del batch  # Liberar memoria del lote
```

---

## 🎯 Puntos Críticos de Control

1. **Antes de ejecutar cualquier script:**
   - ✅ Verificar RAM disponible (`htop` o `free -h`)
   - ✅ Cerrar programas innecesarios
   - ✅ Confirmar que el script anterior finalizó

2. **Durante la ejecución:**
   - 👀 Monitorear uso de RAM
   - 🚫 NO abrir otros programas pesados
   - 🚫 NO ejecutar otro script en paralelo

3. **Después de cada script:**
   - ✅ Verificar que el checkpoint `.done` fue creado
   - ✅ Verificar que los archivos de salida existen
   - ✅ Esperar 5-10 segundos antes del siguiente paso

---

## 📊 Dashboard de Monitoreo (Opcional)

Para monitorear el progreso en tiempo real:

```bash
# Terminal 1: Ejecutar pipeline
python run_pipeline.py

# Terminal 2: Monitorear memoria
watch -n 2 'free -h'

# Terminal 3: Ver logs
tail -f data_processed/processing_log.txt
```

---

## ✅ Checklist Pre-Ejecución

Antes de iniciar el pipeline completo:

- [ ] Dataset descargado y descomprimido en `dataset/EuroSAT_MS/`
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] RAM disponible: ≥ 16 GB libres
- [ ] Espacio en disco: ≥ 30 GB libres
- [ ] Todos los programas pesados cerrados
- [ ] Script `run_pipeline.py` con permisos de ejecución
- [ ] `config.py` revisado y configurado correctamente
