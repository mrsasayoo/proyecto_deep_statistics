# Proyecto 1 - Situación 2: Clustering para Evaluación Agroecológica

## 📋 Descripción

Análisis de clustering no supervisado sobre imágenes satelitales multiespectrales (EuroSAT) para identificar zonas agroecológicas en estrés tras eventos climáticos extremos.

**Dataset:** EuroSAT - 27,000 imágenes de Sentinel-2 (13 bandas espectrales)  
**Objetivo:** Segmentar coberturas terrestres usando K-Means y DBSCAN  
**Contexto:** Evaluación rápida sin ground truth actualizado  

---

## 🗂️ Estructura del Proyecto

```
situacion_2/
├── dataset/              # Datos crudos (2.1 GB)
├── data_processed/       # Datos procesados listos para usar
│   └── .checkpoints/     # Control de ejecución secuencial
├── src/                  # Scripts de procesamiento (ejecutar en orden)
│   └── utils/            # Funciones auxiliares
├── notebooks/            # Jupyter notebooks
├── outputs/              # Resultados finales
│   ├── models/           # Modelos entrenados
│   ├── figures/          # Gráficos para informe
│   ├── tables/           # Tablas cuantitativas
│   └── reports/          # Informe final PDF
└── docs/                 # Documentación adicional
```

---

## 🚀 Pipeline de Ejecución Secuencial

⚠️ **IMPORTANTE:** Los scripts deben ejecutarse **EN ORDEN** para evitar saturación de RAM.

### Paso 1: Instalación de dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Verificar dataset
Asegúrate de que el dataset esté en:
```
dataset/EuroSAT_MS/
├── AnnualCrop/
├── Forest/
├── ... (10 clases)
```

### Paso 3: Ejecución del pipeline

```bash
# Ejecutar cada script en orden, esperando que termine antes del siguiente
python src/01_data_loading.py
python src/02_normalization.py
python src/03_pca_reduction.py
python src/04_kmeans_clustering.py
python src/05_dbscan_clustering.py
python src/06_evaluation_validation.py
python src/07_visualization_export.py
```

Cada script creará un checkpoint en `data_processed/.checkpoints/` al finalizar.

---

## 📊 Outputs Esperados

### Datos Procesados
- `features_pca_reduced.csv` - Matriz reducida lista para clustering  
- `metadata_labels.csv` - Metadatos de imágenes  
- `pca_variance_explained.csv` - Varianza por componente  

### Modelos
- `pca_model.pkl`, `scaler_model.pkl`  
- `kmeans_model.pkl`, `dbscan_model.pkl`  

### Figuras (ver `outputs/figures/`)
- Varianza explicada acumulada (PCA)  
- Gráfico del Codo y Silueta (K-Means)  
- K-distance graph (DBSCAN)  
- Matrices de Confusión (K-Means vs DBSCAN)  
- Análisis de composición de clústeres  

### Tablas (ver `outputs/tables/`)
- Métricas de evaluación (ARI, NMI, Silueta)  
- Caracterización de clústeres  

---

## 📖 Documentación Completa

Ver: [`plan_maestro.md`](plan_maestro.md) - Guía completa con metodología paso a paso

---

## ✅ Checklist de Ejecución

- [ ] Dependencias instaladas (`requirements.txt`)
- [ ] Dataset descargado y descomprimido
- [ ] Script 01: Carga de datos completado
- [ ] Script 02: Normalización completada
- [ ] Script 03: PCA completado
- [ ] Script 04: K-Means completado
- [ ] Script 05: DBSCAN completado
- [ ] Script 06: Evaluación completada
- [ ] Script 07: Visualizaciones exportadas
- [ ] Informe PDF generado (<25 páginas)
- [ ] Código documentado y limpio
- [ ] Todo empaquetado en ZIP para entrega

---

## 🔧 Troubleshooting

### Problema: "Error: Checkpoint no encontrado"
**Solución:** Ejecuta primero el script anterior en la secuencia.

### Problema: "MemoryError" o RAM saturada
**Solución:** Reduce `BATCH_SIZE` en `src/config.py` (ej: de 1000 a 500).

### Problema: "FileNotFoundError: Dataset not found"
**Solución:** Verifica que el dataset esté en `dataset/EuroSAT_MS/` con las 10 subcarpetas de clases.

---

## 📚 Referencias

- **Dataset:** [EuroSAT GitHub](https://github.com/phelber/eurosat)
- **Paper:** Helber et al., 2019 - EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification
- **Sentinel-2:** [ESA User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)

---

## 👥 Autores

**Grupo:** [Tu nombre/grupo aquí]  
**Curso:** Analítica de Datos I  
**Fecha:** Febrero 2026  
