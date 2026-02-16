# Situación 3 — Portafolio Hipotecario a Gran Escala

## Inicio Rápido

```bash
# 1. Instalar dependencias
cd situacion_3/
pip install -r requirements.txt

# 2. Verificar estructura del dataset (leer encabezados de los 101 archivos)
python src/00_test_headers.py

# 3. Ejecutar pipeline completo
python run_pipeline.py
```

## Dataset

- **Fuente:** Freddie Mac Single-Family Fixed-Rate Loan Performance Data
- **Archivo:** `dataset/Performance.zip` (55.31 GB comprimido, ~820 GB descomprimido)
- **Contenido:** 101 archivos CSV (2000Q1 → 2025Q1), delimitados por `|`, sin encabezados, 110 columnas

## Estructura

```
src/
  config.py              # Variables globales, rutas, parámetros
  00_test_headers.py     # Verificar estructura de los 101 archivos
  01-05_*.py             # Fases del pipeline analítico
  utils/
    data_loader.py       # Carga desde ZIP con chunks + progreso
    memory_utils.py      # Monitoreo y liberación de RAM
    plotting_utils.py    # Gráficos académicos (300 DPI)
```

## Documentación

- [plan_maestro.md](plan_maestro.md) — Plan completo del proyecto con arquitectura de paralelización
