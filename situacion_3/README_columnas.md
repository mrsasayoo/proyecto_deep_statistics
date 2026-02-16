# Diccionario de Datos - Performance File

Dataset: Fannie Mae Single-Family Loan-Level, Performance File (CAS/CRT)

> **110 columnas** definidas en el diccionario. La **posicion 1 (Reference Pool ID)** no
> esta presente en los archivos CSV de este dataset. Las columnas 2-110 del diccionario
> se mapean a las posiciones 1-109 del archivo pipe-delimited (despues del campo vacio
> generado por el delimitador inicial).

---

## Identificacion

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 1 | `reference_pool_id` | Reference Pool ID | ALPHA-NUMERIC | X(4) | Identificador único del pool de referencia al que pertenece el préstamo en operaciones CRT |
| 2 | `loan_identifier` | Loan Identifier | ALPHA-NUMERIC | X(12) | Identificador único del préstamo hipotecario (no corresponde a otros IDs de Fannie Mae) |
| 11 | `upb_at_issuance` | UPB at Issuance | NUMERIC | 9(10).99 | Saldo insoluto del préstamo a la fecha de corte del pool de referencia CRT |
| 104 | `deal_name` | Deal Name | ALPHA-NUMERIC | X(200) | Título de la emisión de la serie (nombre del deal CRT/CAS/CIRT) |

## Temporal

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 3 | `monthly_reporting_period` | Monthly Reporting Period | DATE | MMYYYY | Mes y año del corte de información reportado por el servicer |
| 14 | `origination_date` | Origination Date | DATE | MMYYYY | Fecha de emisión del pagaré hipotecario individual |
| 15 | `first_payment_date` | First Payment Date | DATE | MMYYYY | Fecha del primer pago programado según los documentos del préstamo |
| 19 | `maturity_date` | Maturity Date | DATE | MMYYYY | Mes y año en que el préstamo está programado para pagarse completamente |
| 38 | `interest_only_first_principal_and_interest_payment_date` | Interest Only First Principal And Interest Payment Date | DATE | MMYYYY | Para préstamos interest-only: fecha del primer pago de capital e interés totalmente amortizante |
| 45 | `zero_balance_effective_date` | Zero Balance Effective Date | DATE | MMYYYY | Fecha en que el saldo del préstamo fue reducido a cero |
| 47 | `repurchase_date` | Repurchase Date | DATE | MMYYYY | Fecha en que ocurrió la recompra del préstamo (Reversed Credit Event) |
| 51 | `last_paid_installment_date` | Last Paid Installment Date | DATE | MMYYYY | Fecha de vencimiento de la última cuota pagada recibida para el préstamo |
| 52 | `foreclosure_date` | Foreclosure Date | DATE | MMYYYY | Fecha en que se completó legalmente la ejecución hipotecaria (foreclosure) |
| 53 | `disposition_date` | Disposition Date | DATE | MMYYYY | Fecha en que Fannie Mae transfirió la propiedad a un tercero o se satisfizo la obligación |
| 82 | `zero_balance_code_change_date` | Zero Balance Code Change Date | DATE | MMYYYY | Fecha más reciente en que se identificó un cambio en el Zero Balance Code del préstamo |
| 84 | `loan_holdback_effective_date` | Loan Holdback Effective Date | DATE | MMYYYY | Fecha del cambio más reciente en el indicador de retención del préstamo |

## Entidades

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 4 | `channel` | Channel | ALPHA-NUMERIC | X(1) | Canal de originación: R=Retail / C=Correspondent / B=Broker |
| 5 | `seller_name` | Seller Name | ALPHA-NUMERIC | X(50) | Nombre de la entidad que entregó el préstamo a Fannie Mae (muestra 'Other' si representa <1% del volumen) |
| 6 | `servicer_name` | Servicer Name | ALPHA-NUMERIC | X(50) | Nombre del servicer primario del préstamo (en blanco antes de dic-2001) |
| 7 | `master_servicer` | Master Servicer | ALPHA-NUMERIC | X(10) | Servicer maestro del préstamo (siempre Fannie Mae) |

## Originacion

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 8 | `original_interest_rate` | Original Interest Rate | NUMERIC | 9(2).999 | Tasa de interés original del préstamo según el pagaré hipotecario |
| 10 | `original_upb` | Original UPB | NUMERIC | 9(10).99 | Monto original del préstamo en dólares según el pagaré al momento de originar |
| 13 | `original_loan_term` | Original Loan Term | NUMERIC | 9(3) | Número de meses de pagos programados al momento de originar el préstamo |
| 20 | `original_loan_to_value_ratio_ltv` | Original Loan to Value Ratio (LTV) | NUMERIC | 9(3) | Relación porcentual entre el monto del préstamo y el valor de la propiedad al originar |
| 21 | `original_combined_loan_to_value_ratio_cltv` | Original Combined Loan to Value Ratio (CLTV) | NUMERIC | 9(3) | Relación entre todos los préstamos conocidos al originar y el valor de la propiedad |
| 22 | `number_of_borrowers` | Number of Borrowers | NUMERIC | 9(2) | Número de personas obligadas a pagar el préstamo hipotecario |
| 23 | `debt_to_income_dti` | Debt-To-Income (DTI) | NUMERIC | 9(2) | Ratio de deuda total mensual sobre ingreso mensual del prestatario al originar |
| 24 | `borrower_credit_score_at_origination` | Borrower Credit Score at Origination | NUMERIC | 9(3) | Puntaje FICO clásico del prestatario principal al momento de originación |
| 25 | `co_borrower_credit_score_at_origination` | Co-Borrower Credit Score at Origination | NUMERIC | 9(3) | Puntaje FICO clásico del co-prestatario al momento de originación |
| 26 | `first_time_home_buyer_indicator` | First Time Home Buyer Indicator | ALPHA-NUMERIC | X(1) | Indica si el prestatario es comprador de vivienda por primera vezg este sc (Y/N/Null) |
| 27 | `loan_purpose` | Loan Purpose | ALPHA-NUMERIC | X(1) | Propósito del préstamo: C=Cash-Out Refi / R=Refinance / P=Purchase / U=Refi no especificado |
| 34 | `mortgage_insurance_percentage` | Mortgage Insurance Percentage | NUMERIC | 9(3).99 | Porcentaje original de cobertura del seguro hipotecario del préstamo asegurado |
| 35 | `amortization_type` | Amortization Type | ALPHA-NUMERIC | X(3) | Tipo de amortización al originar: ARM=Tasa variable / FRM=Tasa fija |
| 36 | `prepayment_penalty_indicator` | Prepayment Penalty Indicator | ALPHA-NUMERIC | X(1) | Indica si el prestatario tiene penalidad por pago anticipado de capital (Y/N) |
| 37 | `interest_only_loan_indicator` | Interest Only Loan Indicator | ALPHA-NUMERIC | X(1) | Indica si el préstamo requiere solo pagos de interés por un período inicial (Y/N) |
| 39 | `months_to_amortization` | Months to Amortization | NUMERIC | 9(3) | Para interest-only: meses desde el mes actual hasta el primer pago de capital e interés |
| 73 | `mortgage_insurance_type` | Mortgage Insurance Type | ALPHA-NUMERIC | X(1) | Responsable del pago del seguro hipotecario: 1=Borrower / 2=Lender / 3=Enterprise / Null=Sin MI |
| 79 | `special_eligibility_program` | Special Eligibility Program | ALPHA-NUMERIC | X(1) | Programa especial de elegibilidad: F=HFA Preferred / H=HomeReady / R=RefiNow / O=Otro |
| 81 | `relocation_mortgage_indicator` | Relocation Mortgage Indicator | ALPHA-NUMERIC | X(1) | Indica si el préstamo es de reubicación (para empleados trasladados por empleadores) (Y/N) |
| 87 | `high_balance_loan_indicator` | High Balance Loan Indicator | ALPHA-NUMERIC | X(1) | Indica si el saldo original supera el límite conforming general hasta el límite de área de alto costo (Y/N) |

## Performance Mensual

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 9 | `current_interest_rate` | Current Interest Rate | NUMERIC | 9(2).999 | Tasa de interés vigente para el período de pago actual (actualizada si hay modificación) |
| 12 | `current_actual_upb` | Current Actual UPB | NUMERIC | 9(10).99 | Saldo insoluto actual real basado en pagos efectivamente recibidos del prestatario |
| 16 | `loan_age` | Loan Age | NUMERIC | 9(3) | Número de meses calendario desde la fecha de originación del préstamo |
| 17 | `remaining_months_to_legal_maturity` | Remaining Months to Legal Maturity | NUMERIC | 9(3) | Meses restantes hasta la fecha de vencimiento legal definida en los documentos |
| 18 | `remaining_months_to_maturity` | Remaining Months To Maturity | NUMERIC | 9(3) | Meses restantes para amortizar el saldo a cero considerando prepagos (en blanco si modificado) |
| 40 | `current_loan_delinquency_status` | Current Loan Delinquency Status | ALPHA-NUMERIC | X(2) | Número de meses de mora del prestatario según los documentos hipotecarios (XX=desconocido) |
| 41 | `loan_payment_history` | Loan Payment History | ALPHA-NUMERIC | X(48) | Historial de pagos codificado de los últimos 24 meses (00=corriente / 01=30-59d / 02=60-89d / etc.) |
| 43 | `mortgage_insurance_cancellation_indicator` | Mortgage Insurance Cancellation Indicator | ALPHA-NUMERIC | X(2) | Estado del seguro hipotecario: Y=cancelado / M=vigente / NA=nunca tuvo MI |
| 48 | `scheduled_principal_current` | Scheduled Principal Current | NUMERIC | 9(10).99 | Pago mínimo de capital obligatorio para el período de reporte según términos del préstamo |
| 49 | `total_principal_current` | Total Principal Current | NUMERIC | 9(10).99 | Cambio total en el UPB actual entre el período anterior y el actual |
| 50 | `unscheduled_principal_current` | Unscheduled Principal Current | NUMERIC | 9(10).99 | Capital recibido en exceso del pago programado (prepago parcial del período) |
| 74 | `servicing_activity_indicator` | Servicing Activity Indicator | ALPHA-NUMERIC | X(1) | Indica si hubo un cambio en la actividad de servicing durante el período de reporte (Y/N) |
| 83 | `loan_holdback_indicator` | Loan Holdback Indicator | ALPHA-NUMERIC | X(1) | Indica si el préstamo está en estado de retención temporal por evaluación de Fannie Mae (Y/N/Null) |
| 110 | `interest_bearing_upb` | Interest Bearing UPB | NUMERIC | 9(10).99 | UPB actual del préstamo excluyendo cualquier porción sin intereses resultado de modificación elegible |

## Colateral

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 28 | `property_type` | Property Type | ALPHA-NUMERIC | X(2) | Tipo de propiedad: CO=Condo / CP=Co-op / PU=PUD / MH=Manufactured / SF=Single-Family |
| 29 | `number_of_units` | Number of Units | NUMERIC | 9(1) | Número de unidades que componen la propiedad hipotecada (1 a 4) |
| 30 | `occupancy_status` | Occupancy Status | ALPHA-NUMERIC | X(1) | Estado de ocupación al originar: P=Principal / S=Second / I=Investor / U=Unknown |
| 31 | `property_state` | Property State | ALPHA-NUMERIC | X(2) | Abreviatura de dos letras del estado donde se ubica la propiedad |
| 32 | `metropolitan_statistical_area_msa` | Metropolitan Statistical Area (MSA) | ALPHA-NUMERIC | X(5) | Código numérico del área estadística metropolitana de la propiedad (00000 si no aplica) |
| 33 | `zip_code_short` | Zip Code Short | ALPHA-NUMERIC | X(3) | Primeros tres dígitos del código postal de la propiedad |
| 86 | `property_valuation_method` | Property Valuation Method | ALPHA-NUMERIC | X(1) | Método de valuación de la propiedad: A=Avalúo / W=Waiver / C=Waiver+datos condición / P=Waiver+AVM / R=GSE Refi |

## Modificacion

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 42 | `modification_flag` | Modification Flag | ALPHA-NUMERIC | X(1) | Indica si el préstamo ha sido modificado en algún momento (Y/N) |
| 63 | `modification_related_non_interest_bearing_upb` | Modification-Related Non-Interest Bearing UPB | NUMERIC | 9(10).99 | Porción del saldo que no acumula intereses como resultado de una modificación elegible |
| 75 | `current_period_modification_loss_amount` | Current Period Modification Loss Amount | NUMERIC | 9(10).99 | Pérdida calculada para el préstamo resultante de un evento de modificación en el período actual |
| 76 | `cumulative_modification_loss_amount` | Cumulative Modification Loss Amount | NUMERIC | 9(10).99 | Pérdida acumulada calculada para el préstamo resultante de uno o más eventos de modificación |
| 77 | `current_period_credit_event_net_gain_or_loss` | Current Period Credit Event Net Gain or Loss | NUMERIC | 9(10).99 | Ganancia o pérdida neta realizada por evento de crédito en el período actual (positivo=pérdida) |
| 78 | `cumulative_credit_event_net_gain_or_loss` | Cumulative Credit Event Net Gain or Loss | NUMERIC | 9(10).99 | Ganancia o pérdida neta acumulada resultante de eventos de crédito del préstamo |

## Liquidacion

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 44 | `zero_balance_code` | Zero Balance Code | ALPHA-NUMERIC | X(3) | Código de razón por la que el saldo llegó a cero (01=prepago / 03=short sale / 09=REO / etc.) |
| 46 | `upb_at_the_time_of_removal` | UPB at the Time of Removal | NUMERIC | 9(10).99 | Saldo insoluto del préstamo en el momento de su remoción del pool o liquidación |
| 54 | `foreclosure_costs` | Foreclosure Costs | NUMERIC | 9(10).99 | Gastos asociados a obtener el título de la propiedad y mantener servicios durante el proceso |
| 55 | `property_preservation_and_repair_costs` | Property Preservation and Repair Costs | NUMERIC | 9(10).99 | Gastos de mantenimiento y reparación de la propiedad durante el proceso de disposición |
| 56 | `asset_recovery_costs` | Asset Recovery Costs | NUMERIC | 9(10).99 | Gastos de desalojo de ocupantes y bienes personales de la propiedad post-foreclosure |
| 57 | `miscellaneous_holding_expenses_and_credits` | Miscellaneous Holding Expenses and Credits | NUMERIC | 9(10).99 | Gastos y créditos varios durante la tenencia (HOA_seguros_ingresos_por_renta_etc.) |
| 58 | `associated_taxes_for_holding_property` | Associated Taxes for Holding Property | NUMERIC | 9(10).99 | Pago de impuestos asociados a la tenencia de la propiedad durante el proceso |
| 59 | `net_sales_proceeds` | Net Sales Proceeds | NUMERIC | 9(10).99 | Efectivo neto recibido de la venta de la propiedad después de comisiones y gastos |
| 60 | `credit_enhancement_proceeds` | Credit Enhancement Proceeds | NUMERIC | 9(10).99 | Ingresos de seguros hipotecarios y acuerdos de recourse para limitar la exposición crediticia |
| 61 | `repurchase_make_whole_proceeds` | Repurchase Make Whole Proceeds | NUMERIC | 9(10).99 | Montos recibidos bajo acuerdos de representación y garantía por recompra del préstamo |
| 62 | `other_foreclosure_proceeds` | Other Foreclosure Proceeds | NUMERIC | 9(10).99 | Montos distintos al precio de venta recibidos post-foreclosure (ej. ingresos de redención) |
| 64 | `principal_forgiveness_amount` | Principal Forgiveness Amount | NUMERIC | 9(10).99 | Reducción formal del saldo acordada entre prestamista y prestatario en una modificación |
| 80 | `foreclosure_principal_write_off_amount` | Foreclosure Principal Write-off Amount | NUMERIC | 9(10).99 | Montos determinados como incobrables bajo leyes estatales por prescripción del foreclosure |
| 85 | `delinquent_accrued_interest` | Delinquent Accrued Interest | NUMERIC | 9(10).99 | Interés acumulado no cobrado calculado cuando el préstamo incurre en un evento de crédito |
| 105 | `repurchase_make_whole_proceeds_flag` | Repurchase Make Whole Proceeds Flag | ALPHA-NUMERIC | X(1) | Indica si Fannie Mae recibió ingresos por recompra bajo acuerdos de representación y garantía (Y/N) |

## Precio de Propiedad

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 65 | `original_list_start_date` | Original List Start Date | DATE | MMYYYY | Fecha acordada para iniciar el proceso de venta de la propiedad con el broker |
| 66 | `original_list_price` | Original List Price | NUMERIC | 9(10).99 | Precio inicial al que la propiedad fue ofertada en venta |
| 67 | `current_list_start_date` | Current List Start Date | DATE | MMYYYY | Fecha más reciente de inicio de listado de la propiedad para venta |
| 68 | `current_list_price` | Current List Price | NUMERIC | 9(10).99 | Precio actual al que la propiedad está siendo ofertada en venta |

## Credito Dinamico

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 69 | `borrower_credit_score_at_issuance` | Borrower Credit Score At Issuance | NUMERIC | 9(3) | Puntaje FICO Score 5 del prestatario principal más reciente a la fecha de emisión del CRT |
| 70 | `co_borrower_credit_score_at_issuance` | Co-Borrower Credit Score At Issuance | NUMERIC | 9(3) | Puntaje FICO Score 5 del co-prestatario más reciente a la fecha de emisión del CRT |
| 71 | `borrower_credit_score_current` | Borrower Credit Score Current | NUMERIC | 9(3) | Puntaje FICO Score 5 del prestatario principal más reciente disponible (dinámico) |
| 72 | `co_borrower_credit_score_current` | Co-Borrower Credit Score Current | NUMERIC | 9(3) | Puntaje FICO Score 5 del co-prestatario más reciente disponible (dinámico) |

## ARM (Adjustable-Rate Mortgage)

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 88 | `arm_initial_fixed_rate_period_lte_5_yr_indicator` | ARM Initial Fixed-Rate Period <= 5 YR Indicator | ALPHA-NUMERIC | X(1) | Para ARM: indica si el período inicial de tasa fija es de 5 años o menos (Y/N) |
| 89 | `arm_product_type` | ARM Product Type | ALPHA-NUMERIC | X(100) | Para ARM: describe el período fijo inicial_la frecuencia de ajuste y el plazo original del préstamo |
| 90 | `initial_fixed_rate_period` | Initial Fixed-Rate Period | NUMERIC | 9(4) | Para ARM: número de meses entre el inicio de acumulación de interés y la primera fecha de cambio de tasa |
| 91 | `interest_rate_adjustment_frequency` | Interest Rate Adjustment Frequency | NUMERIC | 9(4) | Para ARM: número de meses entre cambios de tasa programados tras el período inicial |
| 92 | `next_interest_rate_adjustment_date` | Next Interest Rate Adjustment Date | DATE | MMYYYY | Para ARM: próxima fecha en que la tasa de interés está sujeta a cambio |
| 93 | `next_payment_change_date` | Next Payment Change Date | DATE | MMYYYY | Para ARM: próxima fecha en que el monto del pago del prestatario puede cambiar |
| 94 | `arm_index` | Index | ALPHA-NUMERIC | X(100) | Para ARM: descripción del índice sobre el que se basan los ajustes de la tasa de interés |
| 95 | `arm_cap_structure` | ARM Cap Structure | ALPHA-NUMERIC | X(10) | Para ARM: estructura de topes de tasa en formato numérico (inicial/periódico/de por vida) |
| 96 | `initial_interest_rate_cap_up_percent` | Initial Interest Rate Cap Up Percent | NUMERIC | 9(2).9999 | Para ARM: máximo porcentaje de aumento de tasa en la primera fecha de cambio |
| 97 | `periodic_interest_rate_cap_up_percent` | Periodic Interest Rate Cap Up Percent | NUMERIC | 9(2).9999 | Para ARM: máximo porcentaje de aumento de tasa en cada fecha de cambio subsecuente |
| 98 | `lifetime_interest_rate_cap_up_percent` | Lifetime Interest Rate Cap Up Percent | NUMERIC | 9(2).9999 | Para ARM: máximo porcentaje de aumento de tasa durante toda la vida del préstamo |
| 99 | `mortgage_margin` | Mortgage Margin | NUMERIC | 9(2).9999 | Para ARM: margen que se suma al índice para establecer la nueva tasa en cada fecha de ajuste |
| 100 | `arm_balloon_indicator` | ARM Balloon Indicator | ALPHA-NUMERIC | X(1) | Para ARM: indica si el préstamo tiene característica de pago globo al vencimiento (Y/N) |
| 101 | `arm_plan_number` | ARM Plan Number | NUMERIC | 9(4) | Para ARM: código del plan estandarizado bajo el que el préstamo fue entregado a Fannie Mae |

## Asistencia al Prestatario

| Pos | Campo (snake_case) | Nombre Original | Tipo | Long. | Descripcion |
|----:|:-------------------|:----------------|:-----|:------|:------------|
| 102 | `borrower_assistance_plan` | Borrower Assistance Plan | ALPHA-NUMERIC | X(1) | Tipo de plan de asistencia activo: F=Forbearance / R=Repayment / T=Trial / O=Otro / N=Ninguno |
| 103 | `high_loan_to_value_hltv_refinance_option_indicator` | High Loan to Value (HLTV) Refinance Option Indicator | ALPHA-NUMERIC | X(1) | Indica si el préstamo original fue refinanciado bajo la opción HLTV permaneciendo en el pool (Y/N) |
| 106 | `alternative_delinquency_resolution` | Alternative Delinquency Resolution | ALPHA-NUMERIC | X(1) | Solución de mitigación de pérdida: P=Payment Deferral / C=COVID Deferral / D=Desastre Natural / 7=N/A |
| 107 | `alternative_delinquency_resolution_count` | Alternative Delinquency Resolution Count | NUMERIC | 9(3) | Número total de resoluciones alternativas de mora reportadas por el servicer para el préstamo |
| 108 | `total_deferral_amount` | Total Deferral Amount | NUMERIC | 9(10).99 | Monto total de capital diferido sin intereses relacionado con una o más resoluciones alternativas |
| 109 | `payment_deferral_modification_event_indicator` | Payment Deferral Modification Event Indicator | ALPHA-NUMERIC | X(1) | Indica si un diferimiento de pago está contribuyendo a un evento de modificación (Y/N/7=N/A) |

