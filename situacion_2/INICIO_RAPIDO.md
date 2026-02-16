# 🚀 INICIO RÁPIDO - Proyecto Situación 2

## ⚡ Ejecución en 3 Pasos

### 1️⃣ Preparar Entorno (una sola vez)
```bash
cd tareas/proyecto/situacion_2
pip install -r requirements.txt
```

### 2️⃣ Descargar Dataset (si no lo tienes)
- Descargar EuroSAT Multiespectral (2.1 GB)
- Descomprimir en `dataset/EuroSAT_MS/`
- Verificar: deben existir 10 carpetas (AnnualCrop, Forest, etc.)

### 3️⃣ Ejecutar Pipeline Completo
```bash
python run_pipeline.py
```

**Tiempo total:** ~80-160 minutos (depende del hardware)

---

## 🎯 Lo Más Importante

### ⚠️ REGLA DE ORO: EJECUCIÓN SECUENCIAL

```
❌ NUNCA HACER:
   - Ejecutar dos scripts al mismo tiempo
   - Ejecutar script N sin haber terminado script N-1
   - Ejecutar en múltiples terminales en paralelo

✅ SIEMPRE HACER:
   - Ejecutar UN script a la vez
   - Esperar a que termine completamente
   - Verificar que el checkpoint .done existe
   - Pausar entre scripts para liberar RAM
```

---

## 📂 Estructura Predicha del Proyecto

```
situacion_2/
├── dataset/EuroSAT_MS/          # 2.1 GB - Imágenes originales
├── data_processed/              # ~20 GB - Datos intermedios
│   ├── .checkpoints/            # 🔒 Control secuencial (7 archivos .done)
│   ├── features_pca_reduced.csv # ⭐ Archivo clave (50-200 MB)
│   └── [otros .npy temporales]
├── src/                         # 📜 7 scripts secuenciales + utils
├── outputs/
│   ├── models/                  # 6 modelos .pkl
│   ├── figures/                 # ~20 gráficos PNG (300 DPI)
│   │   ├── 01_pca/
│   │   ├── 02_kmeans/
│   │   ├── 03_dbscan/
│   │   ├── 04_evaluation/
│   │   └── 05_interpretation/
│   ├── tables/                  # 5 tablas CSV con métricas
│   └── reports/                 # Informe final PDF
└── docs/                        # Documentación adicional
```

---

## 🔢 Scripts del Pipeline (Orden de Ejecución)

| # | Script | Descripción | Tiempo | Output Principal |
|---|--------|-------------|--------|------------------|
| 1 | `01_data_loading.py` | Carga 27k imágenes | 15-30 min | `features_raw_flattened.npy` |
| 2 | `02_normalization.py` | Normaliza datos | 5-10 min | `features_normalized.npy` |
| 3 | `03_pca_reduction.py` | Reduce dimensiones | 10-20 min | `features_pca_reduced.csv` ⭐ |
| 4 | `04_kmeans_clustering.py` | K-Means clustering | 20-40 min | `kmeans_model.pkl` |
| 5 | `05_dbscan_clustering.py` | DBSCAN clustering | 15-30 min | `dbscan_model.pkl` |
| 6 | `06_evaluation_validation.py` | Métricas ARI/NMI | 5-10 min | `metrics_comparison.csv` |
| 7 | `07_visualization_export.py` | Genera gráficos | 10-20 min | `~20 figuras .png` |

**Total:** ~80-160 minutos

---

## 📊 Gráficos Que Se Generarán

### Para el Informe (outputs/figures/):

**PCA (01_pca/)**
- ✅ Varianza explicada acumulada
- ✅ Scree plot
- ✅ Proyección 2D de imágenes

**K-Means (02_kmeans/)**
- ✅ Gráfico del Codo (SSE)
- ✅ Coeficiente de Silueta por k
- ✅ Clústeres en espacio PCA
- ✅ Imágenes representativas de cada clúster

**DBSCAN (03_dbscan/)**
- ✅ K-distance graph (para epsilon)
- ✅ Clústeres encontrados
- ✅ Análisis de outliers/ruido

**Evaluación (04_evaluation/)**
- ✅ Matriz de confusión K-Means (cruda y normalizada)
- ✅ Matriz de confusión DBSCAN (cruda y normalizada)
- ✅ Tabla comparativa de métricas

**Interpretación (05_interpretation/)**
- ✅ Heatmap de composición de clústeres
- ✅ Distribución de zonas en estrés
- ✅ Separación cultivos vs. bosques

**Total:** ~20 figuras de alta resolución (300 DPI)

---

## 📋 Métricas Cuantitativas (outputs/tables/)

| Archivo | Contenido |
|---------|-----------|
| `pca_components_variance.csv` | Varianza por componente, acumulada |
| `kmeans_optimization_results.csv` | SSE y Silueta para k=2..15 |
| `dbscan_hyperparameters.csv` | Epsilon y min_samples probados |
| `metrics_comparison.csv` | **ARI, NMI, Silueta** (K-Means vs DBSCAN) |
| `cluster_characterization.csv` | ¿Qué clases predominan en cada clúster? |

---

## 💾 Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|------------|---------|-------------|
| **RAM** | 16 GB | 32 GB |
| **Disco** | 30 GB libres | 50 GB libres |
| **CPU** | 4 cores | 8+ cores |
| **Python** | 3.9+ | 3.10+ |

---

## 🛠️ Comandos Útiles

### Ver progreso del pipeline
```bash
python run_pipeline.py --skip-existing  # Continuar desde donde quedó
```

### Ejecutar desde un paso específico
```bash
python run_pipeline.py --step 4  # Ejecutar desde K-Means
```

### Ver qué se ejecutaría sin ejecutar
```bash
python run_pipeline.py --dry-run
```

### Monitorear memoria en tiempo real
```bash
watch -n 2 'free -h'
```

### Ver checkpoints completados
```bash
ls -lh data_processed/.checkpoints/
```

---

## 🚨 Solución de Problemas Comunes

### ❌ "Error: Checkpoint no encontrado"
**Causa:** Script anterior no finalizó correctamente  
**Solución:** Ejecuta el script previo en el orden

### ❌ "MemoryError" o sistema se congela
**Causa:** RAM insuficiente  
**Solución:**
1. Cierra todos los programas
2. Edita `src/config.py`: cambia `BATCH_SIZE = 1000` a `BATCH_SIZE = 500`
3. Reinicia el paso que falló

### ❌ "FileNotFoundError: Dataset not found"
**Causa:** Dataset no está en la ubicación correcta  
**Solución:** Verifica que existe `dataset/EuroSAT_MS/` con 10 subcarpetas

### ❌ Pipeline interrumpido (Ctrl+C)
**Causa:** Interrupción manual  
**Solución:**
```bash
python run_pipeline.py --skip-existing
```

---

## 📚 Documentación Disponible

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Descripción general del proyecto |
| `plan_maestro.md` | 🎯 **Guía completa metodológica** (LEER PRIMERO) |
| `docs/arquitectura_sistema.md` | Diagramas de arquitectura y flujo de datos |
| `src/config.py` | Configuración global (paths, constantes) |

---

## ✅ Checklist de Entrega Final

Antes de empaquetar el ZIP:

- [ ] Pipeline completo ejecutado (7 checkpoints .done)
- [ ] Todas las figuras generadas (~20 archivos .png)
- [ ] Todas las tablas generadas (5 archivos .csv)
- [ ] Informe PDF redactado (<25 páginas)
- [ ] Código limpio y comentado
- [ ] `requirements.txt` actualizado
- [ ] README.md con instrucciones de ejecución

**Estructura del ZIP:**
```
apellido_nombre_situacion2.zip
├── informe_situacion_2.pdf
├── src/
├── notebooks/
├── outputs/figures/
├── outputs/tables/
└── README.md
```

---

## 🎓 Preguntas Clave del Informe

**Fase PCA:**
- [ ] ¿Cuántos componentes para >90% varianza?
- [ ] ¿Qué reducción dimensional se logró?

**Fase K-Means:**
- [ ] ¿Cuál es el k óptimo (método Codo + Silueta)?
- [ ] ¿Está en el rango esperado (8-12)?

**Fase DBSCAN:**
- [ ] ¿Cómo se determinaron epsilon y min_samples?
- [ ] ¿Cuántos clústeres encontró?
- [ ] ¿Qué % de imágenes son ruido?

**Interpretación Agroecológica:**
- [ ] ¿Qué clústeres representan zonas en estrés?
- [ ] ¿Se separaron claramente cultivos de bosques?
- [ ] ¿Qué % del territorio está en estado saludable?

**Validación:**
- [ ] ¿Qué modelo tiene mejor Silueta?
- [ ] ¿Qué modelo tiene mejor ARI y NMI?

---

## 🚀 ¡Listo para Comenzar!

```bash
python run_pipeline.py
```

⏱️ Ve por un café... el pipeline tardará ~2 horas.

**Siguiente paso:** Mientras se ejecuta, puedes ir leyendo [`plan_maestro.md`](plan_maestro.md) para preparar el informe.
