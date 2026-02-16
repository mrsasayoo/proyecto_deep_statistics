"""
image_loader.py - Carga eficiente de imágenes multiespectrales
==============================================================
Funciones para leer imágenes .tif de Sentinel-2 (13 bandas) por lotes
sin saturar la RAM. Usa tifffile para lectura directa de GeoTIFF.

Input: Carpetas con imágenes .tif (64x64x13)
Output: numpy arrays en float32 para eficiencia de memoria
"""

import os
import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, List


def get_all_image_paths(dataset_path: Path, class_names: list) -> Tuple[List[str], List[str]]:
    """
    Recorre las carpetas del dataset y recolecta rutas de imágenes + etiquetas.
    
    Args:
        dataset_path: Ruta raíz del dataset EuroSAT_MS/
        class_names: Lista de nombres de clases (carpetas)
        
    Returns:
        file_paths: Lista de rutas absolutas a cada .tif
        labels: Lista de etiquetas correspondientes (nombre de carpeta)
    """
    file_paths = []
    labels = []
    
    for class_name in sorted(class_names):
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"⚠️  Carpeta no encontrada: {class_dir}")
            continue
        
        # Recolectar todos los .tif de esta clase
        tif_files = sorted([f for f in class_dir.iterdir() if f.suffix == '.tif'])
        
        for tif_file in tif_files:
            file_paths.append(str(tif_file))
            labels.append(class_name)
    
    print(f"📊 Total imágenes encontradas: {len(file_paths)}")
    print(f"📊 Distribución por clase:")
    for class_name in sorted(class_names):
        count = labels.count(class_name)
        if count > 0:
            print(f"   {class_name}: {count}")
    
    return file_paths, labels


def load_single_image(file_path: str) -> np.ndarray:
    """
    Carga una imagen .tif multiespectral y la devuelve como array.
    
    Args:
        file_path: Ruta al archivo .tif
        
    Returns:
        Imagen como numpy array de shape (64, 64, 13) en float32
    """
    # tifffile lee en formato (channels, height, width) para Sentinel-2
    img = tifffile.imread(file_path)
    
    # Asegurar formato (H, W, C) - si viene como (C, H, W), transponer
    if img.ndim == 3 and img.shape[0] == 13:
        img = np.transpose(img, (1, 2, 0))
    
    return img.astype(np.float32)


def load_and_flatten_batch(file_paths: List[str], 
                           expected_shape: tuple = (64, 64, 13)) -> np.ndarray:
    """
    Carga un lote de imágenes, las aplana y devuelve como matriz 2D.
    
    Cada imagen (64x64x13) se convierte en un vector de 53,248 elementos.
    Usa float32 para eficiencia de memoria (vs float64 que duplicaría el uso).
    
    Args:
        file_paths: Lista de rutas a archivos .tif
        expected_shape: Tupla con dimensiones esperadas (H, W, C)
        
    Returns:
        Matriz numpy (n_images, 53248) en float32
    """
    n_features = expected_shape[0] * expected_shape[1] * expected_shape[2]
    batch_matrix = np.empty((len(file_paths), n_features), dtype=np.float32)
    
    skipped = 0
    for i, fpath in enumerate(file_paths):
        try:
            img = load_single_image(fpath)
            
            # Validar dimensiones
            if img.shape != expected_shape:
                print(f"⚠️  Dimensiones inesperadas en {fpath}: {img.shape}")
                skipped += 1
                batch_matrix[i] = 0  # Rellenar con ceros
                continue
            
            # Aplanar: (64, 64, 13) → (53248,)
            batch_matrix[i] = img.flatten()
            
        except Exception as e:
            print(f"❌ Error leyendo {fpath}: {e}")
            skipped += 1
            batch_matrix[i] = 0
    
    if skipped > 0:
        print(f"⚠️  {skipped} imágenes con problemas en este lote")
    
    return batch_matrix


def extract_metadata(file_paths: List[str], labels: List[str]) -> dict:
    """
    Crea un diccionario de metadatos para guardar como CSV.
    
    Args:
        file_paths: Lista de rutas a imágenes
        labels: Lista de etiquetas de clase
        
    Returns:
        dict con columnas: image_id, true_label, file_path
    """
    image_ids = [Path(fp).stem for fp in file_paths]
    
    return {
        'image_id': image_ids,
        'true_label': labels,
        'file_path': file_paths
    }
