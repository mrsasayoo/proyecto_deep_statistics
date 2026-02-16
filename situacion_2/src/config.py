"""
Configuración Global del Proyecto - Situación 2
================================================
Define rutas, constantes y parámetros globales para mantener consistencia
en todos los scripts del pipeline secuencial.
"""

import os
from pathlib import Path

# ============================================================================
# 1. RUTAS DE DIRECTORIOS
# ============================================================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "EuroSAT_MS"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data_processed"
CHECKPOINTS_PATH = DATA_PROCESSED_PATH / ".checkpoints"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
MODELS_PATH = OUTPUTS_PATH / "models"
FIGURES_PATH = OUTPUTS_PATH / "figures"
TABLES_PATH = OUTPUTS_PATH / "tables"

# Crear directorios si no existen
for path in [DATA_PROCESSED_PATH, CHECKPOINTS_PATH, OUTPUTS_PATH, 
             MODELS_PATH, FIGURES_PATH, TABLES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. CONSTANTES DE DATOS
# ============================================================================

# Dimensiones de las imágenes
IMG_HEIGHT = 64
IMG_WIDTH = 64
N_CHANNELS = 13  # Bandas espectrales de Sentinel-2
TOTAL_FEATURES = IMG_HEIGHT * IMG_WIDTH * N_CHANNELS  # 53,248

# Clases del dataset (10 categorías de cobertura terrestre)
CLASS_NAMES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

# ============================================================================
# 3. PARÁMETROS DE PROCESAMIENTO
# ============================================================================

# Procesamiento por lotes (evitar saturación de RAM)
BATCH_SIZE = 1000  # Cargar 1000 imágenes a la vez

# PCA
PCA_VARIANCE_THRESHOLD = 0.90  # Retener componentes que expliquen >90% varianza
PCA_MAX_COMPONENTS = 200  # Límite máximo de componentes

# K-Means
KMEANS_K_RANGE = range(2, 16)  # Probar k de 2 a 15
KMEANS_OPTIMAL_K_EXPECTED = range(8, 13)  # Rango esperado según hipótesis
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# DBSCAN
DBSCAN_N_NEIGHBORS_FOR_EPSILON = 4  # Para k-distance graph
DBSCAN_MIN_SAMPLES_RANGE = [3, 5, 10]  # Valores a probar
DBSCAN_EPSILON_PERCENTILE = 95  # Percentil para determinar epsilon

# ============================================================================
# 4. PARÁMETROS DE VISUALIZACIÓN
# ============================================================================

# Estilo profesional para gráficos académicos
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300  # Alta resolución para el informe
FIGURE_FORMAT = 'png'

# Tamaños de figura estándar
FIGSIZE_SMALL = (8, 6)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_LARGE = (16, 10)

# Colores consistentes
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'
COLOR_ACCENT = '#F18F01'

# ============================================================================
# 5. NOMBRES DE ARCHIVOS DE SALIDA
# ============================================================================

# Data processed
FILE_METADATA = "metadata_labels.csv"
FILE_FEATURES_RAW = "features_raw_flattened.npy"
FILE_FEATURES_NORMALIZED = "features_normalized.npy"
FILE_FEATURES_PCA = "features_pca_reduced.csv"
FILE_PCA_VARIANCE = "pca_variance_explained.csv"
FILE_PROCESSING_LOG = "processing_log.txt"

# Modelos
FILE_SCALER_MODEL = "scaler_model.pkl"
FILE_PCA_MODEL = "pca_model.pkl"
FILE_KMEANS_MODEL = "kmeans_model.pkl"
FILE_DBSCAN_MODEL = "dbscan_model.pkl"
FILE_KMEANS_LABELS = "kmeans_labels.npy"
FILE_DBSCAN_LABELS = "dbscan_labels.npy"

# ============================================================================
# 6. CONTROL DE EJECUCIÓN SECUENCIAL
# ============================================================================

# Nombres de archivos checkpoint para validación secuencial
CHECKPOINTS = {
    'data_loading': '01_data_loading.done',
    'normalization': '02_normalization.done',
    'pca_reduction': '03_pca_reduction.done',
    'kmeans': '04_kmeans_clustering.done',
    'dbscan': '05_dbscan_clustering.done',
    'evaluation': '06_evaluation_validation.done',
    'visualization': '07_visualization_export.done'
}

# ============================================================================
# 7. FUNCIONES AUXILIARES
# ============================================================================

def validate_checkpoint(checkpoint_name: str) -> bool:
    """
    Valida que un checkpoint previo haya sido completado.
    
    Args:
        checkpoint_name: Nombre del checkpoint (clave en CHECKPOINTS)
        
    Returns:
        True si el checkpoint existe, False de lo contrario
        
    Raises:
        FileNotFoundError: Si el checkpoint no existe
    """
    checkpoint_file = CHECKPOINTS_PATH / CHECKPOINTS[checkpoint_name]
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"ERROR: Checkpoint '{checkpoint_name}' no encontrado.\n"
            f"Debes ejecutar primero el script correspondiente."
        )
    return True


def create_checkpoint(checkpoint_name: str, metadata: str = None) -> None:
    """
    Crea un archivo checkpoint al finalizar exitosamente un script.
    
    Args:
        checkpoint_name: Nombre del checkpoint (clave en CHECKPOINTS)
        metadata: Información adicional a guardar en el checkpoint
    """
    checkpoint_file = CHECKPOINTS_PATH / CHECKPOINTS[checkpoint_name]
    with open(checkpoint_file, 'w') as f:
        f.write(f"Completed successfully\n")
        if metadata:
            f.write(f"Metadata: {metadata}\n")
    print(f"✅ Checkpoint creado: {checkpoint_file.name}")


def get_path(path_type: str) -> Path:
    """
    Obtiene una ruta configurada.
    
    Args:
        path_type: Tipo de ruta ('dataset', 'processed', 'models', etc.)
        
    Returns:
        Path object
    """
    paths = {
        'dataset': DATASET_PATH,
        'processed': DATA_PROCESSED_PATH,
        'checkpoints': CHECKPOINTS_PATH,
        'models': MODELS_PATH,
        'figures': FIGURES_PATH,
        'tables': TABLES_PATH,
        'outputs': OUTPUTS_PATH
    }
    return paths.get(path_type, PROJECT_ROOT)


# ============================================================================
# 8. INFORMACIÓN DE CONFIGURACIÓN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURACIÓN DEL PROYECTO - SITUACIÓN 2")
    print("=" * 70)
    print(f"\n📁 Directorio raíz: {PROJECT_ROOT}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"💾 Datos procesados: {DATA_PROCESSED_PATH}")
    print(f"🎯 Modelos: {MODELS_PATH}")
    print(f"📈 Figuras: {FIGURES_PATH}")
    print(f"\n🔢 Dimensiones de imagen: {IMG_HEIGHT}x{IMG_WIDTH}x{N_CHANNELS}")
    print(f"🔢 Total features por imagen: {TOTAL_FEATURES:,}")
    print(f"📦 Tamaño de lote: {BATCH_SIZE}")
    print(f"🎯 Varianza PCA deseada: {PCA_VARIANCE_THRESHOLD*100}%")
    print(f"🔍 Rango K-Means: {list(KMEANS_K_RANGE)}")
    print("=" * 70)
