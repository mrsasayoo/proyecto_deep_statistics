# Plan Maestro — Situación 3: Análisis de Portafolio Hipotecario a Gran Escala

**Proyecto:** Analítica de Datos I — Proyecto 1, Situación 3  
**Dataset:** Freddie Mac Single-Family Fixed-Rate Loan Performance Data  
**Autor:** Nicolás Zapata Obando  
**Fecha:** Febrero 2026

---

## 1. Resumen del Problema

Se requiere procesar y analizar el dataset **Freddie Mac Single-Family Fixed-Rate Loan Performance**, compuesto por **101 archivos CSV** dentro de un único archivo comprimido (`Performance.zip`, 55.31 GB comprimido, **~820 GB descomprimido**). Cada archivo corresponde a un trimestre (2000Q1 → 2025Q1) y contiene datos mensuales de desempeño de préstamos hipotecarios. Los archivos individuales varían entre 0.11 GB y 37.20 GB descomprimidos, con 110 columnas separadas por `|` y **sin fila de encabezados**.

### Restricciones Técnicas Críticas

| Restricción | Detalle |
|:---|:---|
| **RAM combinada** | ~30 GB efectivos (Portátil 19 GB + Servidor 14 GB - overhead SO) |
| **Archivo más grande** | 2003Q3.csv → 37.20 GB descomprimido (no cabe en RAM de ninguna máquina) |
| **Total descomprimido** | ~820 GB (imposible mantener todo en memoria) |
| **Sin GPU CUDA** | 100% procesamiento por CPU → se necesita paralelización |
| **Datos en disco remoto** | Dataset alojado en HDD 1TB del servidor, acceso vía NFS sobre Wi-Fi |
| **Latencia de red** | Wi-Fi introduce delay; se requieren barras de progreso y buffering |

---

## 2. Arquitectura del Clúster Distribuido

### 2.1 Topología: Master-Worker sobre LAN

```
┌──────────────────────────────────┐       LAN (Wi-Fi / Ethernet)       ┌──────────────────────────────────┐
│     PORTÁTIL (Master Node)       │◄═══════════════════════════════════►│    PC DE MESA (Worker Node)      │
│  Ubuntu 24.04 Desktop            │                                     │  Ubuntu 24.04 Server             │
│  Intel i5-1155G7 (4C/8T)         │           Ray Cluster               │  AMD Athlon 3000G (2C/4T)        │
│  19 GB RAM                       │◄───────────────────────────────────►│  14 GB RAM                       │
│  IP: 192.168.1.17                │                                     │  IP: 192.168.1.15                │
│  NVMe 238 GB                     │         NFS (puerto 2049)           │  HDD 1TB (Dataset 55 GB ZIP)     │
│                                  │◄───────────────────────────────────►│  SSD 256 GB (SO)                 │
│  → Coordina tareas Ray           │         SSH (puerto 22)             │  → Servidor de datos NFS         │
│  → Procesa archivos GRANDES      │◄───────────────────────────────────►│  → Procesa archivos PEQUEÑOS     │
│  → Agrega resultados finales     │                                     │  → Libera memoria al acabar      │
└──────────────────────────────────┘                                     └──────────────────────────────────┘
                    │                                                                    │
                    └──────────────────── Recursos Combinados ──────────────────────────┘
                                    12 hilos CPU | ~30 GB RAM efectiva
```

### 2.2 Roles Fijos (Sin Coordinación Dinámica)

| Propiedad | Portátil (Master) | Servidor (Worker) |
|:---|:---|:---|
| **CPU** | 8 hilos (i5-1155G7) | 4 hilos (Athlon 3000G) |
| **RAM total** | 19 GB | 14 GB |
| **RAM para datos** | ~4.5 GB por chunk (25% de 19 GB) | ~3 GB por chunk (25% de 13 GB) |
| **Chunk size** | 750,000 filas | 500,000 filas |
| **Chunk size (archivos >20 GB)** | 375,000 filas | 250,000 filas |
| **Archivos asignados** | ~37 archivos GRANDES (>10 GB) | ~64 archivos PEQUEÑOS y MEDIANOS |
| **Rol primario** | Procesador principal + agregación | Procesador secundario |
| **Escritura Parquet** | A disco compartido NFS | A disco compartido NFS |

La lista de 101 archivos se divide **estáticamente al inicio**: el Master toma los archivos pesados, el Worker toma los más ligeros. No hay coordinación dinámica entre máquinas — cada una trabaja su lista de manera independiente para evitar complejidad.

### 2.3 Framework de Distribución: Ray

Se utiliza **Ray** como framework de computación distribuida:

1. **Distribución automática de memoria:** Objetos en Object Store compartido sin duplicar datos
2. **Scheduling inteligente:** Asigna tareas al nodo con más recursos disponibles
3. **Tolerancia a fallos:** Reintento automático de tareas fallidas por red
4. **API Pythónica:** Se integra con pandas, numpy y scikit-learn

```bash
# Iniciar clúster (usar scripts/ray_start.sh):
# En el Servidor (Worker):
ray start --address='192.168.1.17:6379' --num-cpus=4

# En el Portátil (Master):
ray start --head --port=6379 --num-cpus=8
```

### 2.4 Coordinación entre Máquinas

El esquema de coordinación es un **archivo de estado compartido en JSON** en el directorio NFS:

```
data_processed/.checkpoints/processing_state.json
```

Cada máquina escribe el estado de cada archivo:
- `PENDIENTE` → no iniciado
- `EN_PROCESO` → una máquina lo tiene
- `COMPLETADO` → Parquet escrito y validado
- `ERROR` → falló, requiere reprocesamiento

Antes de empezar un archivo, la máquina verifica que no esté `EN_PROCESO` en la otra.

---

## 3. Estructura de Directorios

```
situacion_3/
│
├── plan_maestro.md                    ← Este archivo
├── README.md                          ← Inicio rápido
├── requirements.txt                   ← Dependencias
├── run_pipeline.py                    ← Ejecutor principal del pipeline
├── .gitignore                         ← Excluir datos pesados
│
├── scripts/                           ← Scripts de ejecución rápida
│   ├── info_cluster.sh               ← Info CPU/RAM de ambas máquinas
│   ├── monitor_cluster.sh            ← Monitor en tiempo real
│   ├── ray_start.sh                  ← Iniciar/detener clúster Ray
│   └── test_parallelization.sh       ← Verificar que Ray funciona
│
├── docs/                              ← Documentación técnica
│   ├── red/                           ← Scripts de red y SSH
│   │   ├── .config_red.sh            ← Credenciales SSH (no subir a git)
│   │   ├── ssh_rapido.sh             ← SSH al servidor sin contraseña
│   │   ├── verificar_red.sh          ← Diagnóstico de red completo
│   │   └── ...
│   └── paralelizacion/               ← Doc de la instalación del clúster
│       └── documentacion_paralelizacion.md
│
├── dataset/                           ← Datos crudos (NO subir a git)
│   └── Performance.zip                ← 55.31 GB comprimido (101 CSVs)
│
├── data_processed/                    ← Datos intermedios procesados
│   ├── panel_analitico/               ← Archivos Parquet convertidos
│   │   ├── 2000Q1.parquet
│   │   ├── 2000Q2.parquet
│   │   └── ... (101 archivos)
│   ├── perfiles/                      ← Perfiles estadísticos JSON
│   │   ├── perfil_2000Q1.json
│   │   └── ...
│   ├── perfil_global.json             ← Perfil consolidado de 110 columnas
│   ├── features_latentes/             ← Scores factoriales / embeddings
│   ├── .checkpoints/                  ← Estado del procesamiento
│   │   └── processing_state.json
│   └── processing_log.txt
│
├── src/                               ← Código fuente del pipeline
│   ├── config.py                      ← Variables globales, rutas, parámetros
│   ├── 00_test_headers.py             ← Fase 0.0: Verificar estructura
│   ├── 01_construccion_panel.py       ← Fase 0: EDA + Conversión a Parquet
│   ├── 02_analisis_latente.py         ← Fase 1: AFE/AFC + reducción dimensional
│   ├── 03_deep_learning.py            ← Fase 2: VAE para embeddings
│   ├── 04_clustering.py               ← Fase 3: K-Means, GMM, jerárquico
│   ├── 05_perfilado_riesgo.py         ← Fase 4: Perfiles de riesgo
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py             ← Carga distribuida desde ZIP
│       ├── memory_utils.py            ← Monitoreo y liberación de RAM
│       └── plotting_utils.py          ← Gráficos académicos (300 DPI)
│
├── outputs/
│   ├── figures/
│   │   ├── 00_exploratorio/           ← Histogramas, nulidad, distribuciones
│   │   ├── 01_panel/
│   │   ├── 02_latente/
│   │   ├── 03_deep_learning/
│   │   ├── 04_clustering/
│   │   └── 05_perfiles/
│   ├── models/                        ← Modelos (.pkl, .pt)
│   ├── tables/                        ← Tablas CSV
│   └── reports/
│
└── notebooks/
    └── situacion_3.ipynb
```

---

## 4. Especificación del Dataset

### 4.1 Inventario del ZIP

| Propiedad | Valor |
|:---|:---|
| **Archivo** | `Performance.zip` |
| **Comprimido** | 55.31 GB |
| **Descomprimido** | ~820 GB |
| **Archivos internos** | 101 (2000Q1 a 2025Q1) |
| **Formato** | CSV delimitado por `\|` (pipe), SIN encabezados |
| **Columnas** | 110 (primera vacía por `\|` inicial) |
| **Más pequeño** | `2025Q1.csv` → 0.11 GB |
| **Más grande** | `2003Q3.csv` → 37.20 GB |

### 4.2 Clasificación de Archivos por Tamaño

| Categoría | Rango | Archivos aprox. | Chunk size | Máquina asignada |
|:---|:---|:---|:---|:---|
| **Pequeño** | < 2 GB | ~5 | Carga directa | Worker |
| **Mediano** | 2 – 10 GB | ~55 | 500K filas | Worker |
| **Grande** | > 10 GB | ~41 | 250K filas | Master |

### 4.3 Encabezados

Los 110 nombres de columna se definen en `config.py → PERFORMANCE_COLUMNS`. Incluyen:

| Rango | Descripción |
|:---|:---|
| 1-5 | Identificación del préstamo y servicer |
| 6-9 | Tasas de interés y saldos |
| 10-18 | Plazos, fechas, madurez |
| 19-24 | LTV, DTI, FICO (variables clave de riesgo) |
| 25-35 | Tipo de propiedad, propósito, ocupación |
| 36-60 | Modificaciones, costos de disposición, ejecución |
| 61-109 | Campos extendidos Freddie Mac 2024 |

---

## 5. ¿Por qué convertir a Parquet? — Comparación real

| Métrica | CSV descomprimido | Parquet (Snappy) | Parquet (Zstd) |
|:---|:---|:---|:---|
| **Tamaño estimado** | 819 GB | 90–130 GB | 70–100 GB |
| **Leer 10 columnas de 110** | Lee 819 GB | Lee ~75 GB | Lee ~64 GB |
| **Filtrar por año** | Recorre todo | Salta particiones | Salta particiones |
| **Schema / tipos** | Inferido cada vez | Guardado | Guardado |
| **Velocidad de lectura** | 1x | 8–15x | 10–20x |
| **Compresión vs ZIP** | 1.46x más grande | Similar al ZIP | Más pequeño que ZIP |

**Punto clave:** Con lectura columnar, si una consulta necesita 15 de las 110 columnas, Parquet lee físicamente solo esas 15. El CSV lee las 819 GB completas. Con RAM limitada (19 y 13 GB), esto es la diferencia entre que el script funcione o que congele la máquina.

---

## 6. Fase 0: Construcción del Panel Analítico — EDA Masivo desde ZIP

Esta fase es la más crítica del pipeline. Hace **dos cosas simultáneamente** para no leer los 819 GB dos veces:
1. **Convierte los 101 CSVs a Parquet** (comprimido, columnar)
2. **Acumula estadísticas del EDA** por cada archivo y columna

### 6.1 Estrategia de Extracción desde el ZIP — Sin Descomprimir

La librería `zipfile` de Python permite leer archivos individuales del ZIP sin extraer el resto. **Nunca se extraen los 819 GB al disco.**

```
Performance.zip (55.31 GB en disco)
    │
    ├── Leer metadatos del ZIP (tabla de contenidos, tamaños, CRC32)
    │
    ├── Por cada archivo en la lista asignada a esta máquina:
    │   ├── Abrir stream del CSV dentro del ZIP (sin extraer)
    │   ├── Leer en chunks de N filas
    │   ├── Procesar chunk → acumular estadísticas
    │   ├── Escribir chunk a Parquet (append)
    │   ├── Liberar memoria del chunk explícitamente
    │   └── Al terminar → cerrar stream → gc.collect()
    │
    └── Archivo Parquet resultante queda en disco
```

### 6.2 Criterio para el Tamaño del Chunk

Con 19 GB de RAM en el Master, el chunk debe ocupar máximo el **25% de la RAM disponible** para dejar espacio al proceso de escritura y a las estadísticas acumuladas.

Para las 110 columnas de tipo mixto, una fila pesa aproximadamente **500-800 bytes** en memoria como DataFrame:

| Máquina | RAM disponible para chunk | Bytes por fila | Chunk size |
|:---|:---|:---|:---|
| **Master (19 GB)** | 4.5 GB (25%) | ~600 bytes | ~750,000 filas |
| **Worker (14 GB)** | 3 GB (25%) | ~600 bytes | ~500,000 filas |

Para archivos marcados como **grandes** (>20 GB descomprimidos, como 2003Q3 o 2020-2021), el chunk se reduce a la mitad automáticamente: ~375,000 filas en Master.

---

### 6.3 Sub-Fase 0.1 — Inventario del ZIP

> **Script:** `00_test_headers.py` (ya ejecutado ✅)  
> **Memoria:** < 50 MB  
> **Tiempo:** ~30 segundos

Lee la tabla de contenidos del ZIP sin extraer nada:
- Nombre de cada archivo dentro del ZIP
- Tamaño comprimido y descomprimido
- CRC32 de cada archivo (para verificar integridad después)
- Verificación de las 110 columnas en cada archivo

**Output:** `file_inventory.csv`, `column_consistency_check.csv`, `file_sizes_inventory.png`

---

### 6.4 Sub-Fase 0.2 — Conversión a Parquet + Perfilado Simultáneo

> **Script:** `01_construccion_panel.py`  
> **Tiempo estimado:** 9-15 horas (paralelo, no supervisado)

Este es el script central. Procesa cada archivo CSV en esta secuencia:

#### Bloque 1 — Inicialización por archivo

1. Abrir stream del CSV dentro del ZIP
2. Leer primeras 5 filas para detectar el schema real
3. Inicializar el acumulador de estadísticas (un dict por columna)
4. Inicializar el writer de Parquet en modo append

#### Bloque 2 — Loop de chunks

Por cada chunk leído:

**Paso 1 — Limpieza de valores centinela:**
Reemplazar valores centinela de Freddie Mac (`9`, `99`, `999`, `9999` según el campo) por `NaN` real. Sin esta limpieza, la media de FICO con los 9999 de missing sale disparada.

**Paso 2 — Inferencia de tipos por columna:**
Freddie Mac reporta columnas numéricas como string porque mezclan valores como `"XX"` para missing. Se detecta qué columnas son numéricas, categóricas y fechas. El mapeo se guarda una sola vez en el primer chunk.

**Paso 3 — Acumulación de estadísticas:**
- **Numéricas:** n válidos, n nulos, suma, suma de cuadrados, mínimo, máximo, histograma de 200 bins (bins fijados en primer chunk)
- **Categóricas:** Counter de frecuencias (top-500 categorías más frecuentes)

**Paso 4 — Escritura a Parquet:**
El chunk con tipos correctos se escribe al Parquet con compresión **Zstd nivel 3** (balance compresión/velocidad). Si el archivo descomprimido supera 5 GB, se divide en múltiples Parquet de máximo 2 GB.

**Paso 5 — Liberación de memoria:**
Se eliminan todas las variables del chunk, se llama `gc.collect()`, y se verifica con `psutil` que la memoria volvió al nivel base. Si detecta leak, reduce el chunk size automáticamente.

#### Bloque 3 — Cierre del archivo

1. Cerrar writer de Parquet
2. Calcular estadísticas finales: media = suma/n, varianza = (suma²/n) - media²
3. Guardar perfil como JSON: `perfil_2003Q1.json`
4. Registrar en log: nombre, tiempo, filas totales, columnas con >5% nulos
5. `gc.collect()` final

---

### 6.5 Sub-Fase 0.3 — Validación de Integridad Post-Conversión

Después de convertir cada archivo, un script de validación verifica:

1. **Row count:** Parquet vs log de conversión (deben ser idénticos)
2. **Rango de valores:** Ninguna columna numérica fuera del rango observado
3. **Schema:** El Parquet coincide con el schema target
4. **CRC:** Registro de integridad futura

Si falla → marca como `REQUIERE_REPROCESAMIENTO` y reintenta solo ese archivo.

---

### 6.6 Sub-Fase 0.4 — Consolidación del Perfil Estadístico Global

Una vez que ambas máquinas terminan, se consolidan los 101 perfiles JSON individuales:

**Outputs:**

1. **Tabla maestra de perfil de columnas:** 110 filas × ~30 columnas de estadísticas. Input del EDA visual y decisiones de ingeniería de features.

2. **Reporte de evolución temporal:** Para las 20 columnas clave (FICO, LTV, DTI, tasa, monto, estado de pago), una tabla donde cada fila es un trimestre y las columnas son estadísticas. Revela cambios estructurales sin visualización.

3. **Mapa de nulidad:** Matriz de 101 trimestres × 110 columnas con el % de nulos en cada celda. Visible si hay columnas que se vuelven nulas en ciertos períodos.

4. **Ranking de columnas por informatividad:** Ordenadas por entropía (categóricas) o coeficiente de variación (numéricas). Columnas con entropía ~0 o variación <1% → candidatas a eliminación antes del AFE.

---

### 6.7 Estimación de Tiempos

| Etapa | Master (Portátil) | Worker (Servidor) |
|:---|:---|:---|
| Inventario del ZIP | 30s ✅ | — |
| Conversión + perfilado | 8-14 horas | 5-9 horas |
| Validación post-conversión | 30-60 min | 20-40 min |
| Consolidación global | 10-20 min | — |

**Tiempo total en paralelo:** ~9-15 horas de procesamiento no supervisado.

Por eso el **checkpoint por archivo** es crítico: si la máquina se congela a las 7 horas, el script retoma desde el último archivo completado, no desde cero.

### 6.8 Output Final de la Fase 0

Al terminar:
- 101 archivos Parquet (estimado 70-100 GB total)
- Perfil estadístico completo de las 110 columnas
- Log de integridad de cada archivo
- Matriz de nulidad temporal
- Ranking de informatividad de columnas

Con esto, las fases posteriores trabajan sobre Parquet en **minutos** en lugar de horas, y nunca más tocan el ZIP original salvo para re-validar.

---

## 7. Fases del Pipeline Analítico

### Fase 1: Extracción de Componentes Latentes
**Script:** `02_analisis_latente.py`
- AFE (Análisis Factorial Exploratorio) sobre variables numéricas del panel
- AFC (Análisis Factorial Confirmatorio) para validar hipótesis de riesgo
- Reducción dimensional con PCA/Factor Analysis para datos mixtos
- Generar matriz de puntuaciones factoriales

### Fase 2: Deep Learning — Autoencoders
**Script:** `03_deep_learning.py`
- VAE (Variational Autoencoder) para representación no lineal
- Entrenamiento distribuido entre Master y Worker con Ray
- Generar embeddings de riesgo de baja dimensión

### Fase 3: Segmentación (Clustering)
**Script:** `04_clustering.py`
- K-Means sobre scores/embeddings latentes
- Gaussian Mixture Models (GMM)
- Clustering jerárquico (Ward)
- Comparación de métricas (Silhouette, Davies-Bouldin, Calinski-Harabasz)

### Fase 4: Caracterización de Perfiles de Riesgo
**Script:** `05_perfilado_riesgo.py`
- Centroides por cluster
- Perfilado financiero: FICO, LTV, DTI, tasa de morosidad
- Visualización de perfiles con gráficos de radar
- Generación de informe final

---

## 8. Parámetros de Procesamiento

### 8.1 Parámetros de Memoria

| Parámetro | Valor | Descripción |
|:---|:---|:---|
| `CHUNK_THRESHOLD_GB` | 2.0 | Si archivo > 2 GB → modo chunked |
| `CHUNK_SIZE_ROWS` | 500,000 | Filas por chunk (medianos) |
| `CHUNK_SIZE_ROWS_LARGE` | 250,000 | Filas por chunk (>10 GB) |
| `MIN_AVAILABLE_RAM_GB` | 3.0 | Pausar si RAM disponible < 3 GB |
| `MEMORY_WARNING_THRESHOLD` | 80% | Advertencia |
| `MEMORY_CRITICAL_THRESHOLD` | 90% | Pausa forzada |

### 8.2 Parámetros de Red

| Parámetro | Valor | Descripción |
|:---|:---|:---|
| `NFS_MOUNT_POINT` | `/mnt/datasets/` | Punto de montaje NFS |
| `NETWORK_TIMEOUT_SECONDS` | 120 | Timeout para lectura |
| `NETWORK_RETRY_ATTEMPTS` | 3 | Reintentos ante fallo |
| `NETWORK_RETRY_DELAY` | 5 | Segundos entre reintentos |

### 8.3 Parámetros de Ray

| Parámetro | Valor | Descripción |
|:---|:---|:---|
| `RAY_MASTER_IP` | `192.168.1.17` | IP del portátil |
| `RAY_WORKER_IP` | `192.168.1.15` | IP del servidor |
| `RAY_HEAD_PORT` | 6379 | Puerto Ray |
| `RAY_MASTER_CPUS` | 8 | CPUs Master |
| `RAY_WORKER_CPUS` | 4 | CPUs Worker |
| `RAY_OBJECT_STORE_MB` | 4000 | Object Store (~4 GB) |

### 8.4 Parámetros de Parquet

| Parámetro | Valor | Descripción |
|:---|:---|:---|
| `PARQUET_COMPRESSION` | `zstd` | Compresión (mejor ratio que snappy) |
| `PARQUET_COMPRESSION_LEVEL` | 3 | Nivel Zstd (balance velocidad/ratio) |
| `PARQUET_MAX_FILE_SIZE_GB` | 2.0 | Dividir si Parquet > 2 GB |
| `PARQUET_ROW_GROUP_SIZE` | 500,000 | Filas por row group |

### 8.5 Parámetros de Modelos

| Parámetro | Valor | Descripción |
|:---|:---|:---|
| `PCA_VARIANCE_THRESHOLD` | 0.90 | Retener 90% varianza |
| `N_FACTORS_RANGE` | (5, 30) | Factores para AFE |
| `KMEANS_K_RANGE` | range(3, 15) | Valores de K |
| `GMM_N_COMPONENTS_RANGE` | range(3, 15) | Componentes GMM |
| `VAE_LATENT_DIM` | 32 | Dimensión espacio latente |
| `VAE_EPOCHS` | 50 | Épocas de entrenamiento |
| `VAE_BATCH_SIZE` | 1024 | Batch size |
| `RANDOM_STATE` | 42 | Semilla reproducible |

---

## 9. Valores Centinela de Freddie Mac

Freddie Mac usa valores específicos para indicar datos faltantes. Deben reemplazarse por `NaN` antes de calcular estadísticas:

| Campo | Valor centinela | Significado |
|:---|:---|:---|
| `borrower_credit_score` | 9999 | FICO no disponible |
| `co_borrower_credit_score` | 9999 | FICO co-prestatario no disponible |
| `original_dti` | 999 | DTI no disponible |
| `original_ltv` | 999 | LTV no disponible |
| `original_cltv` | 999 | CLTV no disponible |
| `mortgage_insurance_pct` | 999 | MI% no disponible |
| `number_of_borrowers` | 99 | No disponible |
| `current_loan_delinquency_status` | `"XX"` | No reportado |
| Campos genéricos string | `""`, `" "` | Vacío |

---

## 10. Gestión de la Latencia de Red (Wi-Fi / NFS)

### 10.1 Velocidad estimada

| Escenario | Velocidad | Tiempo para 37 GB |
|:---|:---|:---|
| Wi-Fi 5 (mejor caso) | 100 MB/s | ~6 min |
| Wi-Fi (caso típico) | 40 MB/s | ~15 min |
| Wi-Fi (peor caso) | 20 MB/s | ~31 min |

### 10.2 Barras de progreso

```
📁 Leyendo 2003Q3.csv (37.20 GB descomprimido)...
[████████████████████░░░░░░░░░░░] 65% | Chunk 10/74 | 42 MB/s | ETA: 5:12
💾 RAM: 12.3/19.0 GB (65%) | Disponible: 6.7 GB
```

### 10.3 Reintentos automáticos

3 intentos con 5 segundos entre cada uno ante fallos de `IOError` o `TimeoutError`.

---

## 11. Convenciones de Código

1. **Cada script es autosuficiente** — ejecutable individualmente o como pipeline
2. **Importaciones desde config** — nunca hardcodear rutas o constantes
3. **Liberación de memoria** — `del variable` + `gc.collect()` después de usar
4. **Logging** — timestamps, uso de RAM y progreso al iniciar/finalizar
5. **Barras de progreso** — `tqdm` en toda operación >5 segundos
6. **Docstrings en español** — cada función documentada
7. **Type hints** — en todas las funciones
8. **Formato Parquet** — datos intermedios en Parquet (no CSV)
9. **Figuras a 300 DPI** — vía `plotting_utils.save_figure()`
10. **Checkpoint por archivo** — poder retomar si falla

---

## 12. Scripts de Ejecución Rápida

| Script | Función | Uso |
|:---|:---|:---|
| `scripts/info_cluster.sh` | Info CPU/RAM/Ray de ambas máquinas | `bash scripts/info_cluster.sh` |
| `scripts/monitor_cluster.sh` | Monitor en tiempo real (cada 3s) | `bash scripts/monitor_cluster.sh [intervalo]` |
| `scripts/ray_start.sh` | Iniciar/detener clúster Ray | `bash scripts/ray_start.sh {all\|stop\|status}` |
| `scripts/test_parallelization.sh` | Test funcional de Ray | `bash scripts/test_parallelization.sh` |
| `docs/red/ssh_rapido.sh` | SSH al servidor sin contraseña | `bash docs/red/ssh_rapido.sh [comando]` |
| `docs/red/verificar_red.sh` | Diagnóstico de red completo | `bash docs/red/verificar_red.sh` |

---

## 13. Ejecución

### Verificar estructura (ya completado ✅):
```bash
python src/00_test_headers.py
```

### Fase 0 — Construcción del Panel:
```bash
# Iniciar el clúster Ray:
bash scripts/ray_start.sh all

# Monitor en otra terminal:
bash scripts/monitor_cluster.sh 5

# Ejecutar la construcción del panel:
python src/01_construccion_panel.py
```

### Pipeline completo (desde Fase 0):
```bash
python run_pipeline.py --distributed
```
