"""
Configuración Global del Proyecto — Situación 3
=================================================
Dataset: Freddie Mac Single-Family Fixed-Rate Loan Performance Data
Archivos: 101 CSVs comprimidos (55.31 GB ZIP, ~820 GB descomprimido)

Define rutas, constantes, encabezados de tablas, parámetros de procesamiento,
umbrales de memoria y nombres de modelos para mantener consistencia
en todos los scripts del pipeline.
"""

import os
from pathlib import Path

# ============================================================================
# 1. RUTAS DE DIRECTORIOS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
ZIP_FILE_PATH = DATASET_PATH / "Performance.zip"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data_processed"
PANEL_PATH = DATA_PROCESSED_PATH / "panel_analitico"
PERFILES_PATH = DATA_PROCESSED_PATH / "perfiles"
FEATURES_PATH = DATA_PROCESSED_PATH / "features_latentes"
CHECKPOINTS_PATH = DATA_PROCESSED_PATH / ".checkpoints"
PROCESSING_STATE_FILE = CHECKPOINTS_PATH / "processing_state.json"

OUTPUTS_PATH = PROJECT_ROOT / "outputs"
MODELS_PATH = OUTPUTS_PATH / "models"
FIGURES_PATH = OUTPUTS_PATH / "figures"
TABLES_PATH = OUTPUTS_PATH / "tables"
REPORTS_PATH = OUTPUTS_PATH / "reports"
LOGS_PATH = PROJECT_ROOT / "logs"

# Subdirectorios de figuras (uno por fase del pipeline)
FIGURES_SUBDIRS = {
    "exploratorio": FIGURES_PATH / "00_exploratorio",
    "panel": FIGURES_PATH / "01_panel",
    "latente": FIGURES_PATH / "02_latente",
    "deep_learning": FIGURES_PATH / "03_deep_learning",
    "clustering": FIGURES_PATH / "04_clustering",
    "perfiles": FIGURES_PATH / "05_perfiles",
}

# Crear todos los directorios al importar config
for path in [
    DATASET_PATH, DATA_PROCESSED_PATH, PANEL_PATH, PERFILES_PATH,
    FEATURES_PATH, CHECKPOINTS_PATH, OUTPUTS_PATH, MODELS_PATH,
    FIGURES_PATH, TABLES_PATH, REPORTS_PATH, LOGS_PATH,
]:
    path.mkdir(parents=True, exist_ok=True)

for subdir_path in FIGURES_SUBDIRS.values():
    subdir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. CONSTANTES DEL DATASET
# ============================================================================

# El ZIP contiene 101 archivos CSV (2000Q1 a 2025Q1)
TOTAL_CSV_FILES = 101
CSV_DELIMITER = "|"
CSV_ENCODING = "utf-8"
CSV_HAS_HEADER = False  # Los archivos NO tienen fila de encabezados

# Rango temporal del dataset
YEAR_START = 2000
YEAR_END = 2025
QUARTER_RANGE = ["Q1", "Q2", "Q3", "Q4"]

# ============================================================================
# 3. ENCABEZADOS DE LAS COLUMNAS (Fannie Mae CAS Performance File)
# ============================================================================
# Los archivos NO contienen encabezados. Esta lista define el nombre de cada
# columna según el diccionario de datos de Fannie Mae (110 posiciones,
# basado en columnas_110_fanniemae.csv).
#
# NOTA IMPORTANTE SOBRE EL MAPEO DE POSICIONES:
# - Cada línea CSV inicia y termina con '|', generando 110 campos (0-109).
# - Campo 0 es vacío (artefacto del delimitador inicial) → _empty
# - Campos 1-109 contienen datos, mapeados a las posiciones 2-110 del CSV
#   de Fannie Mae (la posición 1 = Reference Pool ID no está en este dataset).
# - Campo 109 puede ser vacío (artefacto del delimitador final) o interest_bearing_upb.

PERFORMANCE_COLUMNS = [
    "_empty",                                           #  0 - Vacía (artefacto del delimitador inicial '|')
    # === IDENTIFICACIÓN ===
    "loan_identifier",                                  #  1 - Identificador único del préstamo (CSV pos 2)
    # === TEMPORAL ===
    "monthly_reporting_period",                         #  2 - Período de reporte MMYYYY (CSV pos 3)
    # === ENTIDADES ===
    "channel",                                          #  3 - Canal: R=Retail, C=Correspondent, B=Broker (CSV pos 4)
    "seller_name",                                      #  4 - Entidad que entregó el préstamo (CSV pos 5)
    "servicer_name",                                    #  5 - Servicer primario (CSV pos 6)
    "master_servicer",                                  #  6 - Servicer maestro (CSV pos 7)
    # === ORIGINACIÓN ===
    "original_interest_rate",                           #  7 - Tasa de interés original (CSV pos 8)
    # === PERFORMANCE MENSUAL ===
    "current_interest_rate",                            #  8 - Tasa de interés vigente (CSV pos 9)
    # === ORIGINACIÓN ===
    "original_upb",                                     #  9 - Monto original del préstamo USD (CSV pos 10)
    # === IDENTIFICACIÓN ===
    "upb_at_issuance",                                  # 10 - UPB a la fecha de corte del pool CRT (CSV pos 11)
    # === PERFORMANCE MENSUAL ===
    "current_actual_upb",                               # 11 - Saldo insoluto actual real (CSV pos 12)
    # === ORIGINACIÓN ===
    "original_loan_term",                               # 12 - Plazo original en meses (CSV pos 13)
    # === TEMPORAL ===
    "origination_date",                                 # 13 - Fecha de emisión del pagaré MMYYYY (CSV pos 14)
    "first_payment_date",                               # 14 - Fecha del primer pago MMYYYY (CSV pos 15)
    # === PERFORMANCE MENSUAL ===
    "loan_age",                                         # 15 - Edad del préstamo en meses (CSV pos 16)
    "remaining_months_to_legal_maturity",               # 16 - Meses hasta vencimiento legal (CSV pos 17)
    "remaining_months_to_maturity",                     # 17 - Meses restantes considerando prepagos (CSV pos 18)
    # === TEMPORAL ===
    "maturity_date",                                    # 18 - Fecha de vencimiento MMYYYY (CSV pos 19)
    # === ORIGINACIÓN ===
    "original_loan_to_value_ratio_ltv",                 # 19 - LTV original % (CSV pos 20)
    "original_combined_loan_to_value_ratio_cltv",       # 20 - CLTV original % (CSV pos 21)
    "number_of_borrowers",                              # 21 - Número de prestatarios (CSV pos 22)
    "debt_to_income_dti",                               # 22 - Ratio deuda/ingreso % (CSV pos 23)
    "borrower_credit_score_at_origination",             # 23 - FICO prestatario al originar (CSV pos 24)
    "co_borrower_credit_score_at_origination",          # 24 - FICO co-prestatario al originar (CSV pos 25)
    "first_time_home_buyer_indicator",                  # 25 - Primer comprador Y/N (CSV pos 26)
    "loan_purpose",                                     # 26 - C=Cash-Out/R=Refi/P=Purchase (CSV pos 27)
    # === COLATERAL ===
    "property_type",                                    # 27 - SF/CO/CP/MH/PU (CSV pos 28)
    "number_of_units",                                  # 28 - Unidades de la propiedad 1-4 (CSV pos 29)
    "occupancy_status",                                 # 29 - P=Principal/S=Second/I=Investor (CSV pos 30)
    "property_state",                                   # 30 - Estado de la propiedad (2 letras) (CSV pos 31)
    "metropolitan_statistical_area_msa",                # 31 - Código MSA de la propiedad (CSV pos 32)
    "zip_code_short",                                   # 32 - Primeros 3 dígitos del ZIP (CSV pos 33)
    # === ORIGINACIÓN ===
    "mortgage_insurance_percentage",                    # 33 - % cobertura seguro hipotecario (CSV pos 34)
    "amortization_type",                                # 34 - FRM=fija / ARM=variable (CSV pos 35)
    "prepayment_penalty_indicator",                     # 35 - Penalidad por prepago Y/N (CSV pos 36)
    "interest_only_loan_indicator",                     # 36 - Solo interés Y/N (CSV pos 37)
    # === TEMPORAL ===
    "interest_only_first_principal_and_interest_payment_date",  # 37 - Fecha 1er pago P&I MMYYYY (CSV pos 38)
    # === ORIGINACIÓN ===
    "months_to_amortization",                           # 38 - Meses hasta amortización (CSV pos 39)
    # === PERFORMANCE MENSUAL ===
    "current_loan_delinquency_status",                  # 39 - Meses de mora (XX=desconocido) (CSV pos 40)
    "loan_payment_history",                             # 40 - Historial 24 meses codificado (CSV pos 41)
    # === MODIFICACIÓN ===
    "modification_flag",                                # 41 - Préstamo modificado Y/N (CSV pos 42)
    # === PERFORMANCE MENSUAL ===
    "mortgage_insurance_cancellation_indicator",        # 42 - Seguro MI cancelado/vigente (CSV pos 43)
    # === LIQUIDACIÓN ===
    "zero_balance_code",                                # 43 - Razón saldo cero 01-16 (CSV pos 44)
    # === TEMPORAL ===
    "zero_balance_effective_date",                      # 44 - Fecha saldo cero MMYYYY (CSV pos 45)
    # === LIQUIDACIÓN ===
    "upb_at_the_time_of_removal",                       # 45 - UPB al remover del pool (CSV pos 46)
    # === TEMPORAL ===
    "repurchase_date",                                  # 46 - Fecha de recompra MMYYYY (CSV pos 47)
    # === PERFORMANCE MENSUAL ===
    "scheduled_principal_current",                      # 47 - Pago capital programado (CSV pos 48)
    "total_principal_current",                          # 48 - Cambio total en UPB (CSV pos 49)
    "unscheduled_principal_current",                    # 49 - Prepago parcial del período (CSV pos 50)
    # === TEMPORAL ===
    "last_paid_installment_date",                       # 50 - Fecha última cuota pagada MMYYYY (CSV pos 51)
    "foreclosure_date",                                 # 51 - Fecha foreclosure MMYYYY (CSV pos 52)
    "disposition_date",                                 # 52 - Fecha disposición MMYYYY (CSV pos 53)
    # === LIQUIDACIÓN ===
    "foreclosure_costs",                                # 53 - Gastos de foreclosure USD (CSV pos 54)
    "property_preservation_and_repair_costs",           # 54 - Gastos mantenimiento/reparación (CSV pos 55)
    "asset_recovery_costs",                             # 55 - Gastos recuperación activos (CSV pos 56)
    "miscellaneous_holding_expenses_and_credits",       # 56 - Gastos varios tenencia (CSV pos 57)
    "associated_taxes_for_holding_property",            # 57 - Impuestos tenencia (CSV pos 58)
    "net_sales_proceeds",                               # 58 - Ingresos netos venta (CSV pos 59)
    "credit_enhancement_proceeds",                      # 59 - Ingresos mejora crediticia (CSV pos 60)
    "repurchase_make_whole_proceeds",                   # 60 - Ingresos recompra (CSV pos 61)
    "other_foreclosure_proceeds",                       # 61 - Otros ingresos foreclosure (CSV pos 62)
    # === MODIFICACIÓN ===
    "modification_related_non_interest_bearing_upb",    # 62 - UPB sin interés por modificación (CSV pos 63)
    # === LIQUIDACIÓN ===
    "principal_forgiveness_amount",                     # 63 - Condonación de principal USD (CSV pos 64)
    # === PRECIO PROPIEDAD ===
    "original_list_start_date",                         # 64 - Fecha inicio listado original MMYYYY (CSV pos 65)
    "original_list_price",                              # 65 - Precio original de venta USD (CSV pos 66)
    "current_list_start_date",                          # 66 - Fecha inicio listado actual MMYYYY (CSV pos 67)
    "current_list_price",                               # 67 - Precio actual de venta USD (CSV pos 68)
    # === CRÉDITO DINÁMICO ===
    "borrower_credit_score_at_issuance",                # 68 - FICO prestatario a emisión CRT (CSV pos 69)
    "co_borrower_credit_score_at_issuance",             # 69 - FICO co-prestatario a emisión CRT (CSV pos 70)
    "borrower_credit_score_current",                    # 70 - FICO prestatario actual (CSV pos 71)
    "co_borrower_credit_score_current",                 # 71 - FICO co-prestatario actual (CSV pos 72)
    # === ORIGINACIÓN ===
    "mortgage_insurance_type",                          # 72 - Responsable MI: 1/2/3/Null (CSV pos 73)
    # === PERFORMANCE MENSUAL ===
    "servicing_activity_indicator",                     # 73 - Cambio en servicing Y/N (CSV pos 74)
    # === MODIFICACIÓN ===
    "current_period_modification_loss_amount",          # 74 - Pérdida por modificación período (CSV pos 75)
    "cumulative_modification_loss_amount",              # 75 - Pérdida acumulada modificaciones (CSV pos 76)
    "current_period_credit_event_net_gain_or_loss",     # 76 - Ganancia/pérdida evento crédito período (CSV pos 77)
    "cumulative_credit_event_net_gain_or_loss",         # 77 - Ganancia/pérdida acumulada eventos crédito (CSV pos 78)
    # === ORIGINACIÓN ===
    "special_eligibility_program",                      # 78 - F=HFA/H=HomeReady/R=RefiNow (CSV pos 79)
    # === LIQUIDACIÓN ===
    "foreclosure_principal_write_off_amount",           # 79 - Cancelación principal foreclosure (CSV pos 80)
    # === ORIGINACIÓN ===
    "relocation_mortgage_indicator",                    # 80 - Hipoteca reubicación Y/N (CSV pos 81)
    # === TEMPORAL ===
    "zero_balance_code_change_date",                    # 81 - Fecha cambio zero balance code MMYYYY (CSV pos 82)
    # === PERFORMANCE MENSUAL ===
    "loan_holdback_indicator",                          # 82 - Retención temporal Y/N (CSV pos 83)
    # === TEMPORAL ===
    "loan_holdback_effective_date",                     # 83 - Fecha cambio holdback MMYYYY (CSV pos 84)
    # === LIQUIDACIÓN ===
    "delinquent_accrued_interest",                      # 84 - Interés acumulado no cobrado (CSV pos 85)
    # === COLATERAL ===
    "property_valuation_method",                        # 85 - Método valuación A/W/C/P/R (CSV pos 86)
    # === ORIGINACIÓN ===
    "high_balance_loan_indicator",                      # 86 - Saldo supera límite conforming Y/N (CSV pos 87)
    # === ARM ===
    "arm_initial_fixed_rate_period_lte_5_yr_indicator", # 87 - ARM período fijo ≤5 años Y/N (CSV pos 88)
    "arm_product_type",                                 # 88 - ARM tipo producto (CSV pos 89)
    "initial_fixed_rate_period",                        # 89 - ARM meses período fijo inicial (CSV pos 90)
    "interest_rate_adjustment_frequency",               # 90 - ARM meses entre ajustes (CSV pos 91)
    "next_interest_rate_adjustment_date",               # 91 - ARM próxima fecha ajuste MMYYYY (CSV pos 92)
    "next_payment_change_date",                         # 92 - ARM próxima fecha cambio pago MMYYYY (CSV pos 93)
    "arm_index",                                        # 93 - ARM índice de referencia (CSV pos 94)
    "arm_cap_structure",                                # 94 - ARM estructura topes (CSV pos 95)
    "initial_interest_rate_cap_up_percent",             # 95 - ARM tope 1er ajuste % (CSV pos 96)
    "periodic_interest_rate_cap_up_percent",            # 96 - ARM tope periódico % (CSV pos 97)
    "lifetime_interest_rate_cap_up_percent",            # 97 - ARM tope de por vida % (CSV pos 98)
    "mortgage_margin",                                  # 98 - ARM margen sobre índice (CSV pos 99)
    "arm_balloon_indicator",                            # 99 - ARM pago globo Y/N (CSV pos 100)
    "arm_plan_number",                                  #100 - ARM código plan estandarizado (CSV pos 101)
    # === ASISTENCIA ===
    "borrower_assistance_plan",                         #101 - Plan asistencia F/R/T/O/N (CSV pos 102)
    "high_loan_to_value_hltv_refinance_option_indicator",  #102 - Refi HLTV Y/N (CSV pos 103)
    # === IDENTIFICACIÓN ===
    "deal_name",                                        #103 - Nombre del deal CRT/CAS (CSV pos 104)
    # === LIQUIDACIÓN ===
    "repurchase_make_whole_proceeds_flag",              #104 - Flag recompra Y/N (CSV pos 105)
    # === ASISTENCIA ===
    "alternative_delinquency_resolution",               #105 - P=Deferral/C=COVID/D=Desastre (CSV pos 106)
    "alternative_delinquency_resolution_count",         #106 - Total resoluciones alternativas (CSV pos 107)
    "total_deferral_amount",                            #107 - Capital total diferido USD (CSV pos 108)
    "payment_deferral_modification_event_indicator",    #108 - Diferimiento contribuye a modif Y/N (CSV pos 109)
    # === PERFORMANCE MENSUAL ===
    "interest_bearing_upb",                             #109 - UPB excluyendo porción sin interés (CSV pos 110)
]

# Columnas a excluir al cargar (la columna vacía del delimitador inicial)
COLUMNS_TO_DROP = ["_empty"]

# ---------------------------------------------------------------------------
# Mapeo de nombres ANTIGUOS → NUEVOS para renombrar Parquets existentes
# (los 86 trimestres procesados con el esquema anterior)
# ---------------------------------------------------------------------------
_OLD_PERFORMANCE_COLUMNS = [
    "_empty",                              #  0
    "loan_sequence_number",                #  1
    "monthly_reporting_period",            #  2
    "current_loan_delinquency_status",     #  3
    "servicer_name",                       #  4
    "master_servicer",                     #  5
    "current_interest_rate",               #  6
    "current_actual_upb",                  #  7
    "original_interest_rate",              #  8
    "original_upb",                        #  9
    "upb_at_issuance",                     # 10
    "current_deferred_upb",               # 11
    "original_loan_term",                  # 12
    "origination_date",                    # 13
    "first_payment_date",                  # 14
    "loan_age",                            # 15
    "remaining_months_to_maturity",        # 16
    "adjusted_remaining_months",           # 17
    "maturity_date",                       # 18
    "original_ltv",                        # 19
    "original_cltv",                       # 20
    "number_of_borrowers",                 # 21
    "original_dti",                        # 22
    "borrower_credit_score",               # 23
    "co_borrower_credit_score",            # 24
    "first_time_homebuyer_flag",           # 25
    "loan_purpose",                        # 26
    "property_type",                       # 27
    "number_of_units",                     # 28
    "occupancy_status",                    # 29
    "property_state",                      # 30
    "msa",                                 # 31
    "zip_code_short",                      # 32
    "mortgage_insurance_pct",              # 33
    "amortization_type",                   # 34
    "prepayment_penalty_flag",             # 35
    "interest_only_indicator",             # 36
    "interest_only_first_payment_date",    # 37
    "months_to_amortization",              # 38
    "current_loan_delinquency_status_2",   # 39
    "modification_flag",                   # 40
    "mortgage_insurance_cancellation",     # 41
    "zero_balance_code",                   # 42
    "zero_balance_effective_date",         # 43
    "last_paid_installment_date",          # 44
    "foreclosure_date",                    # 45
    "disposition_date",                    # 46
    "foreclosure_costs",                   # 47
    "ppce",                                # 48
    "asset_recovery_costs",                # 49
    "misc_holding_expenses",               # 50
    "holding_taxes_insurance",             # 51
    "net_sale_proceeds",                   # 52
    "credit_enhancement_proceeds",         # 53
    "repurchase_make_whole_proceeds",      # 54
    "other_foreclosure_proceeds",          # 55
    "non_interest_bearing_upb",            # 56
    "principal_forgiveness_upb",           # 57
    "repurchase_make_whole_proceeds_flag", # 58
    "foreclosure_principal_write_off",     # 59
    "servicing_activity_indicator",        # 60
] + [f"col_{i}" for i in range(61, 110)]   # 61-109

# Diccionario old_name → new_name (sólo para columnas que cambiaron de nombre)
OLD_TO_NEW_COLUMN_MAP = {
    old: new
    for old, new in zip(_OLD_PERFORMANCE_COLUMNS[1:], PERFORMANCE_COLUMNS[1:])
    if old != new
}

# Columnas clave para análisis (subconjunto más relevante para clustering)
KEY_NUMERIC_COLUMNS = [
    "current_interest_rate",
    "current_actual_upb",
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "remaining_months_to_maturity",
    "original_loan_to_value_ratio_ltv",
    "original_combined_loan_to_value_ratio_cltv",
    "number_of_borrowers",
    "debt_to_income_dti",
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "mortgage_insurance_percentage",
]

KEY_CATEGORICAL_COLUMNS = [
    "channel",
    "current_loan_delinquency_status",
    "first_time_home_buyer_indicator",
    "loan_purpose",
    "property_type",
    "occupancy_status",
    "property_state",
    "amortization_type",
    "modification_flag",
    "zero_balance_code",
]

# Columna que identifica al préstamo (unir datos longitudinales)
LOAN_ID_COLUMN = "loan_identifier"
DATE_COLUMN = "monthly_reporting_period"

# ============================================================================
# 4. PARÁMETROS DE PROCESAMIENTO (MEMORIA Y CHUNKS)
# ============================================================================

# Umbral para activar lectura por chunks (en GB descomprimidos)
CHUNK_THRESHOLD_GB = 2.0

# Filas por chunk según tamaño del archivo
CHUNK_SIZE_ROWS = 500_000        # Archivos medianos (2-10 GB)
CHUNK_SIZE_ROWS_LARGE = 250_000  # Archivos grandes (>10 GB)
LARGE_FILE_THRESHOLD_GB = 10.0   # A partir de aquí se reduce el chunk

# Umbrales de memoria RAM
MIN_AVAILABLE_RAM_GB = 3.0       # Pausar si RAM disponible < este valor
MEMORY_WARNING_THRESHOLD = 80    # % de uso → advertencia
MEMORY_CRITICAL_THRESHOLD = 90   # % de uso → pausa forzada
MEMORY_CHECK_INTERVAL = 30       # Segundos entre verificaciones de RAM

# Reintentos y espera cuando la RAM está saturada
MEMORY_WAIT_SECONDS = 10         # Esperar antes de reintentar
MEMORY_MAX_RETRIES = 5           # Máximo de reintentos por memoria

# ============================================================================
# 5. PARÁMETROS DE RED (NFS / Wi-Fi)
# ============================================================================

# Rutas NFS
NFS_MOUNT_POINT = Path("/mnt/datasets/")
NETWORK_TIMEOUT_SECONDS = 120
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY = 5          # Segundos entre reintentos de red
PROGRESS_BAR_UPDATE_MB = 50      # Actualizar barra cada N MB leídos

# ============================================================================
# 6. PARÁMETROS DE RAY (COMPUTACIÓN DISTRIBUIDA)
# ============================================================================

RAY_ENABLED = True   # Ray instalado en ambas máquinas
RAY_HEAD_PORT = 6379
RAY_MASTER_IP = "192.168.1.17"  # IP del portátil (Master) — ajustar si cambia DHCP
RAY_WORKER_IP = "192.168.1.15"  # IP del servidor (Worker) — ajustar si cambia DHCP
RAY_WORKER_SSH_USER = "mrsasayo_mesa"
RAY_WORKER_SSH_KEY = "~/.ssh/id_ed25519_proyecto"  # Llave SSH (password auth deshabilitado)
RAY_MASTER_CPUS = 7   # Portátil (Intel i5-1155G7: 4C/8T) → 8-1=7 para procesamiento
RAY_WORKER_CPUS = 3   # Servidor (AMD Athlon 3000G: 2C/4T) → 4-1=3 para procesamiento
RAY_OBJECT_STORE_MB = 4000  # ~4 GB de Object Store compartido

# ============================================================================
# 7. PARÁMETROS DE MODELOS
# ============================================================================

RANDOM_STATE = 42

# PCA / Análisis Factorial
PCA_VARIANCE_THRESHOLD = 0.90
N_FACTORS_RANGE = (5, 30)

# K-Means
KMEANS_K_RANGE = range(3, 15)
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# Gaussian Mixture Models
GMM_N_COMPONENTS_RANGE = range(3, 15)
GMM_COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]

# Clustering Jerárquico
HIERARCHICAL_METHODS = ["ward", "complete", "average"]
HIERARCHICAL_MAX_CLUSTERS = 15

# Deep Learning — VAE (Variational Autoencoder)
VAE_LATENT_DIM = 32
VAE_HIDDEN_DIMS = [256, 128, 64]
VAE_EPOCHS = 50
VAE_BATCH_SIZE = 1024
VAE_LEARNING_RATE = 1e-3

# ============================================================================
# 8. NOMBRES DE MODELOS Y ARCHIVOS DE SALIDA
# ============================================================================

MODEL_NAMES = {
    "pca": "pca_model.pkl",
    "factor_analysis": "factor_analysis_model.pkl",
    "vae": "vae_model.pt",
    "kmeans": "kmeans_model.pkl",
    "gmm": "gmm_model.pkl",
    "hierarchical": "hierarchical_labels.npy",
}

TABLE_NAMES = {
    "file_inventory": "file_inventory.csv",
    "column_consistency": "column_consistency_check.csv",
    "descriptive_stats": "descriptive_statistics.csv",
    "variance_explained": "pca_variance_explained.csv",
    "factor_loadings": "factor_loadings.csv",
    "cluster_metrics": "cluster_metrics_comparison.csv",
    "cluster_profiles": "cluster_risk_profiles.csv",
    "silhouette_scores": "silhouette_scores.csv",
}

FIGURE_NAMES = {
    "scree_plot": "scree_plot.png",
    "elbow_plot": "elbow_plot.png",
    "silhouette_plot": "silhouette_analysis.png",
    "dendrogram": "dendrogram_ward.png",
    "radar_profiles": "radar_risk_profiles.png",
    "latent_space_2d": "latent_space_2d.png",
    "latent_space_3d": "latent_space_3d.png",
    "vae_loss_curve": "vae_training_loss.png",
}

# ============================================================================
# 9. FORMATO DE OUTPUT
# ============================================================================

# Parquet
PARQUET_COMPRESSION = "zstd"       # Compresión Zstd (mejor ratio que snappy)
PARQUET_COMPRESSION_LEVEL = 3     # Nivel Zstd: 1=rápido, 22=máximo. 3=balance
PARQUET_MAX_FILE_SIZE_GB = 2.0    # Dividir en múltiples archivos si supera este tamaño
PARQUET_ROW_GROUP_SIZE = 500_000  # Filas por row group en Parquet

# Figuras
FIGURE_DPI = 300               # Resolución de las figuras
FIGURE_FORMAT = "png"
LOG_FILE = DATA_PROCESSED_PATH / "processing_log.txt"

# ============================================================================
# 10. VALORES CENTINELA DE FREDDIE MAC
# ============================================================================
# Freddie Mac usa estos valores para indicar datos faltantes.
# Se reemplazan por NaN antes de calcular estadísticas.

SENTINEL_VALUES = {
    "borrower_credit_score_at_origination": [9999],
    "co_borrower_credit_score_at_origination": [9999],
    "borrower_credit_score_at_issuance": [9999],
    "co_borrower_credit_score_at_issuance": [9999],
    "borrower_credit_score_current": [9999],
    "co_borrower_credit_score_current": [9999],
    "debt_to_income_dti": [999],
    "original_loan_to_value_ratio_ltv": [999],
    "original_combined_loan_to_value_ratio_cltv": [999],
    "mortgage_insurance_percentage": [999],
    "number_of_borrowers": [99],
    "current_loan_delinquency_status": ["XX"],
}

# Columnas que se espera sean numéricas (para forzar conversión)
EXPECTED_NUMERIC_COLUMNS = [
    "original_interest_rate", "current_interest_rate",
    "original_upb", "upb_at_issuance", "current_actual_upb",
    "original_loan_term", "loan_age",
    "remaining_months_to_legal_maturity", "remaining_months_to_maturity",
    "original_loan_to_value_ratio_ltv", "original_combined_loan_to_value_ratio_cltv",
    "number_of_borrowers", "debt_to_income_dti",
    "borrower_credit_score_at_origination", "co_borrower_credit_score_at_origination",
    "mortgage_insurance_percentage", "number_of_units",
    "months_to_amortization",
    "upb_at_the_time_of_removal",
    "scheduled_principal_current", "total_principal_current",
    "unscheduled_principal_current",
    "foreclosure_costs", "property_preservation_and_repair_costs",
    "asset_recovery_costs", "miscellaneous_holding_expenses_and_credits",
    "associated_taxes_for_holding_property", "net_sales_proceeds",
    "credit_enhancement_proceeds", "repurchase_make_whole_proceeds",
    "other_foreclosure_proceeds",
    "modification_related_non_interest_bearing_upb",
    "principal_forgiveness_amount",
    "original_list_price", "current_list_price",
    "borrower_credit_score_at_issuance", "co_borrower_credit_score_at_issuance",
    "borrower_credit_score_current", "co_borrower_credit_score_current",
    "current_period_modification_loss_amount", "cumulative_modification_loss_amount",
    "current_period_credit_event_net_gain_or_loss",
    "cumulative_credit_event_net_gain_or_loss",
    "foreclosure_principal_write_off_amount",
    "delinquent_accrued_interest",
    "initial_fixed_rate_period", "interest_rate_adjustment_frequency",
    "initial_interest_rate_cap_up_percent", "periodic_interest_rate_cap_up_percent",
    "lifetime_interest_rate_cap_up_percent", "mortgage_margin",
    "arm_plan_number",
    "alternative_delinquency_resolution_count",
    "total_deferral_amount",
    "interest_bearing_upb",
]

# Columnas de fecha (formato MMYYYY)
DATE_COLUMNS = [
    "monthly_reporting_period", "origination_date", "first_payment_date",
    "maturity_date", "interest_only_first_principal_and_interest_payment_date",
    "zero_balance_effective_date", "last_paid_installment_date",
    "foreclosure_date", "disposition_date", "repurchase_date",
    "original_list_start_date", "current_list_start_date",
    "zero_balance_code_change_date", "loan_holdback_effective_date",
    "next_interest_rate_adjustment_date", "next_payment_change_date",
]

# Top-N categorías a rastrear en el perfilado
TOP_CATEGORIES_LIMIT = 500

# Bins para histogramas acumulados
HISTOGRAM_N_BINS = 200

# ============================================================================
# 10. PRINT INICIAL (VERIFICACIÓN)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURACIÓN — Situación 3: Portafolio Hipotecario")
    print("=" * 60)
    print(f"  Raíz del proyecto:      {PROJECT_ROOT}")
    print(f"  Archivo ZIP:            {ZIP_FILE_PATH}")
    print(f"  ZIP existe:             {ZIP_FILE_PATH.exists()}")
    print(f"  Total columnas:         {len(PERFORMANCE_COLUMNS)}")
    print(f"  Chunk threshold:        {CHUNK_THRESHOLD_GB} GB")
    print(f"  Chunk size (mediano):   {CHUNK_SIZE_ROWS:,} filas")
    print(f"  Chunk size (grande):    {CHUNK_SIZE_ROWS_LARGE:,} filas")
    print(f"  Min RAM disponible:     {MIN_AVAILABLE_RAM_GB} GB")
    print(f"  Ray habilitado:         {RAY_ENABLED}")
    print(f"  Random state:           {RANDOM_STATE}")
    print("=" * 60)
