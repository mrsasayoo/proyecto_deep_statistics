# Documentación Técnica: Configuración de Clúster de Cómputo Distribuido para Entrenamiento de Inteligencia Artificial

**Autor:** Nicolás Zapata Obando 
**Institución:** Universidad Autónoma de Occidente — Cali, Colombia  
**Carrera:** Ingeniería de Datos e Inteligencia Artificial (7.º semestre)  
**Fecha de elaboración del proyecto:** Enero 2026  
**Fecha de documentación:** Febrero 2026

---

# SESIÓN 1: Configuración Inicial del Clúster

**Objetivo:** Instalación de Ubuntu Server en dual boot, configuración de montaje de disco, y establecimiento de compartición NFS entre el portátil y el PC de mesa.

**Período:** Enero 2026

---

## Tabla de Contenidos — Sesión 1

1. [Introducción y Motivación](#1-introducción-y-motivación)
2. [Auditoría de Hardware Disponible](#2-auditoría-de-hardware-disponible)
   - 2.1 [Portátil — Lenovo IdeaPad 3 14ITL6](#21-portátil--lenovo-ideapad-3-14itl6)
   - 2.2 [PC de Mesa — Ensamblado](#22-pc-de-mesa--ensamblado)
   - 2.3 [Tabla Comparativa de Recursos](#23-tabla-comparativa-de-recursos)
   - 2.4 [Diagnóstico Técnico y Limitaciones](#24-diagnóstico-técnico-y-limitaciones)
3. [Estrategia Arquitectónica: Clúster Híbrido de CPU](#3-estrategia-arquitectónica-clúster-híbrido-de-cpu)
   - 3.1 [Modelo de Computación Distribuida](#31-modelo-de-computación-distribuida)
   - 3.2 [Justificación de Ubuntu Server 24.04 LTS](#32-justificación-de-ubuntu-server-2404-lts)
   - 3.3 [Topología de Red y Almacenamiento](#33-topología-de-red-y-almacenamiento)
4. [Fase 1: Preparación del Medio de Instalación](#4-fase-1-preparación-del-medio-de-instalación)
   - 4.1 [Descarga de la Imagen ISO](#41-descarga-de-la-imagen-iso)
   - 4.2 [Creación de la USB Booteable con Rufus](#42-creación-de-la-usb-booteable-con-rufus)
5. [Fase 2: Particionado del Disco en Windows](#5-fase-2-particionado-del-disco-en-windows)
   - 5.1 [Reducción del Volumen C: en Administración de Discos](#51-reducción-del-volumen-c-en-administración-de-discos)
   - 5.2 [Desactivación de Inicio Rápido](#52-desactivación-de-inicio-rápido)
6. [Fase 3: Instalación de Ubuntu Server 24.04 (Dual Boot)](#6-fase-3-instalación-de-ubuntu-server-2404-dual-boot)
   - 6.1 [Arranque desde USB — Menú GRUB](#61-arranque-desde-usb--menú-grub)
   - 6.2 [Selección del Tipo de Instalación](#62-selección-del-tipo-de-instalación)
   - 6.3 [Configuración de Red](#63-configuración-de-red)
   - 6.4 [Configuración de Almacenamiento — El Desafío Crítico](#64-configuración-de-almacenamiento--el-desafío-crítico)
   - 6.5 [Resolución con cfdisk: Particionado Manual desde Terminal](#65-resolución-con-cfdisk-particionado-manual-desde-terminal)
   - 6.6 [Asignación de Tipos de Partición (EFI System y Linux Filesystem)](#66-asignación-de-tipos-de-partición-efi-system-y-linux-filesystem)
   - 6.7 [Configuración Final de Montajes en el Instalador](#67-configuración-final-de-montajes-en-el-instalador)
   - 6.8 [Error de Instalación y Resolución](#68-error-de-instalación-y-resolución)
   - 6.9 [Configuración Post-Instalación (Ubuntu Pro, SSH, Snaps)](#69-configuración-post-instalación-ubuntu-pro-ssh-snaps)
   - 6.10 [Reinicio y Verificación del Dual Boot](#610-reinicio-y-verificación-del-dual-boot)
7. [Fase 4: Montaje del Disco de Datos (HDD 1TB)](#7-fase-4-montaje-del-disco-de-datos-hdd-1tb)
   - 7.1 [Identificación del Disco con lsblk](#71-identificación-del-disco-con-lsblk)
   - 7.2 [Configuración de /etc/fstab](#72-configuración-de-etcfstab)
   - 7.3 [El Concepto de Montaje en Linux vs. Windows](#73-el-concepto-de-montaje-en-linux-vs-windows)
   - 7.4 [Verificación del Montaje](#74-verificación-del-montaje)
8. [Fase 5: Configuración del Servidor NFS](#8-fase-5-configuración-del-servidor-nfs)
   - 8.1 [Instalación y Configuración en el Servidor (PC de Mesa)](#81-instalación-y-configuración-en-el-servidor-pc-de-mesa)
   - 8.2 [Archivo /etc/exports — Reglas de Compartición](#82-archivo-etcexports--reglas-de-compartición)
   - 8.3 [Verificación de IP del Servidor](#83-verificación-de-ip-del-servidor)
9. [Fase 6: Configuración del Cliente NFS (Portátil)](#9-fase-6-configuración-del-cliente-nfs-portátil)
   - 9.1 [Instalación del Cliente y Montaje Remoto](#91-instalación-del-cliente-y-montaje-remoto)
   - 9.2 [Verificación de la Conexión Distribuida](#92-verificación-de-la-conexión-distribuida)
   - 9.3 [Consideraciones de Red: Wi-Fi vs. Ethernet](#93-consideraciones-de-red-wi-fi-vs-ethernet)
10. [Fundamentos Técnicos Profundos](#10-fundamentos-técnicos-profundos)
    - 10.1 [Tipos de Partición GPT y su Propósito](#101-tipos-de-partición-gpt-y-su-propósito)
    - 10.2 [NFS: Arquitectura y Funcionamiento](#102-nfs-arquitectura-y-funcionamiento)
    - 10.3 [El Archivo /etc/fstab: Anatomía Completa](#103-el-archivo-etcfstab-anatomía-completa)
11. [Conclusiones](#11-conclusiones)
12. [Índice de Figuras](#12-índice-de-figuras)

---

## 1. Introducción y Motivación

Este documento describe el proceso completo de diseño, configuración e implementación de un **clúster de cómputo distribuido de bajo costo** orientado al entrenamiento de modelos de Deep Learning, construido a partir de dos computadoras convencionales interconectadas en una red local.

### Contexto Académico

En el séptimo semestre de la carrera de **Ingeniería de Datos e Inteligencia Artificial** en la Universidad Autónoma de Occidente, se presentó el requisito de entrenar algoritmos de inteligencia artificial como parte de una asignatura de 9 créditos. El profesor adelantó que uno de los datasets de trabajo tendría un tamaño aproximado de **55 GB de almacenamiento**, lo cual superaba significativamente las capacidades individuales de cualquiera de los equipos disponibles.

### Problema Técnico

El estudiante disponía de dos máquinas con recursos complementarios:

- Un **portátil** con buen procesador y RAM, pero almacenamiento limitado (256 GB SSD), corriendo Linux.
- Un **PC de mesa** con almacenamiento abundante (1 TB HDD), pero procesador modesto, corriendo Windows 10.

La pregunta central fue: *¿Es posible paralelizar ambas computadoras para que, al entrenar algoritmos de inteligencia artificial, se utilicen los recursos de ambas máquinas simultáneamente, a pesar de tener sistemas operativos diferentes?*

### Solución Adoptada

Se optó por implementar un **clúster de computación distribuida** basado en:

1. **Unificación de sistemas operativos** mediante la instalación de Ubuntu Server 24.04 LTS en dual boot sobre el PC de mesa.
2. **Centralización de datos** en el disco de 1 TB mediante un servidor **NFS (Network File System)**.
3. **Interconexión por red local** (inicialmente Wi-Fi, con migración planificada a Ethernet) para el acceso compartido a los datos y la coordinación del entrenamiento distribuido.

---

## 2. Auditoría de Hardware Disponible

Antes de diseñar la arquitectura del clúster, se realizó una auditoría exhaustiva del hardware disponible utilizando comandos nativos de cada sistema operativo. Este paso es fundamental en cualquier proyecto de ingeniería de datos para dimensionar correctamente los recursos y anticipar cuellos de botella.

### 2.1 Portátil — Lenovo IdeaPad 3 14ITL6

**Sistema Operativo:** Ubuntu 24.04.3 LTS  
**Kernel:** Linux 6.8.0-90-generic  
**Arquitectura:** x86-64  
**Firmware:** GGCN55WW (Lenovo, 2023-06-12)

**Comandos utilizados para la auditoría:**

```bash
hostnamectl      # Información general del sistema
lscpu            # Detalle del procesador
free -h          # Memoria RAM en formato legible
df -h            # Espacio en disco y particiones
lsblk            # Dispositivos de bloque (discos físicos)
```

**Resultados:**

| Componente | Especificación |
|:---|:---|
| **Procesador** | Intel Core i5-1155G7 @ 2.50 GHz (Turbo hasta 4.5 GHz) |
| **Núcleos / Hilos** | 4 núcleos / 8 hilos |
| **Caché** | L1d: 192 KiB, L1i: 128 KiB, L2: 5 MiB, L3: 8 MiB |
| **RAM Total** | 19 GB (13 GB disponibles en uso normal) |
| **Almacenamiento** | 238.5 GB NVMe SSD (133 GB libres) |
| **GPU** | Intel Iris Xe (integrada) |
| **Virtualización** | VT-x habilitada |

**Características relevantes del procesador:**
- Soporte para instrucciones AVX-512 (aceleración de operaciones vectoriales, relevante para frameworks de IA).
- Arquitectura Tiger Lake de 11.ª generación con optimizaciones para Intel OneDNN.

### 2.2 PC de Mesa — Ensamblado

**Sistema Operativo original:** Windows 10 Home (Build 19045)  
**Hostname:** DESKTOP-XXXXXXX

**Comandos utilizados para la auditoría (PowerShell):**

```powershell
systeminfo                                                    # Resumen del sistema
Get-CimInstance Win32_Processor | Select-Object Name          # Procesador
Get-Volume                                                    # Volúmenes de disco
Get-PhysicalDisk                                              # Discos físicos
wmic path win32_VideoController get name                      # GPU
```

**Resultados:**

| Componente | Especificación |
|:---|:---|
| **Procesador** | AMD Athlon 3000G with Radeon Vega Graphics @ 3.5 GHz |
| **Núcleos / Hilos** | 2 núcleos / 4 hilos |
| **RAM Total** | 14.267 MB (~14 GB), 9.741 MB disponibles |
| **Almacenamiento SSD** | Patriot P210 256 GB (28 GB libres en C:) |
| **Almacenamiento HDD** | TOSHIBA MQ04ABF100 — 931.51 GB (779.8 GB libres en D:) |
| **GPU** | AMD Radeon Vega (integrada) |
| **Red** | Realtek PCIe GbE (Ethernet), IP: 192.168.1.3 |
| **BIOS** | American Megatrends Inc. P4.00 (2020-07-16), UEFI habilitada |
| **Hyper-V** | Virtualización habilitada en firmware |

### 2.3 Tabla Comparativa de Recursos

| Componente | **Portátil (Linux — Master)** | **PC de Mesa (Windows → Linux — Worker)** |
|:---|:---|:---|
| **Procesador** | Intel i5-1155G7 (4C/8T, hasta 4.5 GHz) | AMD Athlon 3000G (2C/4T, 3.5 GHz) |
| **Potencia CPU** | Alta — 11.ª gen. con AVX-512 | Básica — Gama de entrada |
| **RAM** | 19 GB (13 GB disponibles) | 14 GB (9.7 GB disponibles, ~4 GB menos con Windows) |
| **Almacenamiento rápido** | 238 GB NVMe SSD (133 GB libres) | 256 GB SATA SSD (28 GB libres) |
| **Almacenamiento masivo** | — | **1 TB HDD** (779.8 GB libres) |
| **GPU** | Intel Iris Xe (integrada, sin CUDA) | Radeon Vega (integrada, sin CUDA) |
| **Conectividad** | Wi-Fi (sin puerto Ethernet nativo) | Realtek PCIe GbE (Ethernet nativo) |

### 2.4 Diagnóstico Técnico y Limitaciones

1. **Sin GPU dedicada NVIDIA:** Ninguna máquina dispone de tarjeta gráfica con soporte CUDA. Todo el entrenamiento será **100% por CPU**, lo que hace indispensable la paralelización para tiempos razonables.

2. **Desbalance de cómputo:** El portátil es aproximadamente 3 veces más potente que el PC de mesa en capacidad de procesamiento (8 hilos vs. 4 hilos, con frecuencias superiores y caché significativamente mayor).

3. **Cuello de botella de almacenamiento:** El dataset de 55 GB no cabe cómodamente en el SSD del portátil (133 GB libres, pero con sistema operativo y software instalado). El disco de **1 TB del PC de mesa** es el único lugar viable para alojar los datos de forma permanente.

4. **Desbalance de conectividad:** El PC de mesa tiene Ethernet nativo (ideal para transferencias estables), mientras que el portátil solo dispone de Wi-Fi (requiere adaptador USB-Ethernet para rendimiento óptimo).

---

## 3. Estrategia Arquitectónica: Clúster Híbrido de CPU

### 3.1 Modelo de Computación Distribuida

Se decidió implementar una arquitectura **Master-Worker** para entrenamiento distribuido:

```
┌─────────────────────────┐         Red Local (LAN)         ┌─────────────────────────┐
│     PORTÁTIL (Master)   │◄════════════════════════════════►│    PC DE MESA (Worker)  │
│  Ubuntu 24.04 Desktop   │        Ethernet / Wi-Fi          │  Ubuntu 24.04 Server    │
│  i5-1155G7 (8 hilos)    │                                  │  Athlon 3000G (4 hilos) │
│  19 GB RAM              │                                  │  14 GB RAM              │
│  238 GB SSD             │         NFS (puerto 2049)        │  256 GB SSD + 1 TB HDD  │
│                         │◄────────────────────────────────►│  Servidor de Datos      │
│  Código + Ejecución     │         SSH (puerto 22)          │  Dataset 55 GB          │
└─────────────────────────┘                                  └─────────────────────────┘
```

**Roles:**
- **Portátil (Master Node):** Estación de trabajo principal. Gestiona el flujo del algoritmo y aporta 8 hilos de procesamiento. El usuario programa y controla todo desde aquí.
- **PC de Mesa (Worker Node):** Servidor de datos y nodo de cómputo auxiliar. Aporta 4 hilos de procesamiento y aloja el dataset de 55 GB en su disco de 1 TB, compartiéndolo por NFS.

**Recursos combinados totales:** 12 hilos de ejecución y ~30 GB de RAM efectiva para el modelo.

### 3.2 Justificación de Ubuntu Server 24.04 LTS

La elección de Ubuntu Server 24.04 LTS para el PC de mesa se fundamentó en cuatro razones técnicas:

1. **Paridad con el portátil:** Al tener la misma versión (24.04) en ambas máquinas, se garantiza que las versiones del kernel, Python, PyTorch, Ray y demás librerías sean idénticas. En computación distribuida, las diferencias de versiones son la causa número uno de errores crípticos.

2. **Eficiencia de recursos (Server vs. Desktop):** Ubuntu Server no incluye interfaz gráfica (GUI). Mientras Windows 10 consume entre 4-6 GB de RAM solo por estar encendido y Ubuntu Desktop consume ~2-3 GB, Ubuntu Server consume menos de **500 MB**. Esto libera casi **3-4 GB de RAM adicionales** para el modelo de Deep Learning.

3. **LTS (Long Term Support):** Soporte por 5 años con parches de seguridad y drivers optimizados, incluyendo los controladores para el procesador AMD Athlon 3000G y la integrada Radeon Vega.

4. **Estándar de la industria:** El 90% de los servidores de AWS, Google Cloud y Azure ejecutan Ubuntu. Dominar su administración es una habilidad profesional directa.

### 3.3 Topología de Red y Almacenamiento

**Protocolo de datos:** NFS v4 (Network File System) para compartir el directorio de datasets.  
**Protocolo de control:** SSH (Secure Shell) para administración remota del servidor sin monitor.  
**Conexión física:** Cable Ethernet al router (servidor), Wi-Fi (portátil) con plan de migración a adaptador USB-Ethernet.

---

## 4. Fase 1: Preparación del Medio de Instalación

### 4.1 Descarga de la Imagen ISO

Se descargó la imagen oficial de Ubuntu Server 24.04.3 LTS desde el sitio de Canonical:

- **URL de descarga:** `https://releases.ubuntu.com/24.04/ubuntu-24.04.3-live-server-amd64.iso`
- **Arquitectura:** AMD64 (compatible con procesadores x86-64, incluyendo el AMD Athlon 3000G)
- **Peso del archivo:** ~2.6 GB
- **Tipo:** Live Server (instalador interactivo en modo texto, sin interfaz gráfica)

Se verificó que la versión coincidiera con la del portátil (Ubuntu 24.04.3 LTS) para garantizar paridad total del kernel y las librerías base.

![Figura 1 — Archivo ISO de Ubuntu Server 24.04.3 descargado](imagenes/1.png)

**Figura 1:** Interfaz de descarga mostrando el archivo `ubuntu-24.04.3-live-server-amd64.iso` correctamente descargado. Se observa que el archivo tiene la extensión `.iso`, que es un formato de imagen de disco óptico estándar utilizado para distribución de sistemas operativos.

![Figura 2 — Confirmación de versión de Ubuntu en el portátil](imagenes/2.png)

**Figura 2:** Terminal del portátil mostrando la versión del sistema operativo ya instalado: Ubuntu 24.04.3 LTS. Esta verificación confirma la paridad de versiones entre ambas máquinas, requisito fundamental para el funcionamiento del entrenamiento distribuido.

**Procedimiento para llegar a esta imagen:** Se abrió un navegador web en el portátil (que ya tenía Windows  instalado), se navegó al sitio oficial de Ubuntu (`ubuntu.com/download/server`) y se descargó directamente el archivo ISO de la versión Server.

### 4.2 Creación de la USB Booteable con Rufus

Para crear el medio de instalación arrancable, se utilizó **Rufus** en Windows (la PC de mesa aún tenía Windows 10 en ese momento).

**Configuración crítica de Rufus:**

| Parámetro | Valor | Justificación |
|:---|:---|:---|
| **Dispositivo** | USB de ≥8 GB | Capacidad mínima para alojar la ISO |
| **Esquema de partición** | **GPT** | Obligatorio para sistemas UEFI modernos y compatibilidad con el Dual Boot |
| **Sistema de destino** | **UEFI (no CSM)** | La BIOS del PC de mesa (American Megatrends) opera en modo UEFI |
| **Modo de escritura** | Modo Imagen ISO | Recomendado por Rufus para distribuciones de Linux |

![Figura 3 — Rufus abierto con la imagen ISO cargada](imagenes/3.png)

**Figura 3:** Interfaz de "Propiedades de windows" mostrando la imagen ISO de Ubuntu Server seleccionada. Rufus es una herramienta gratuita para Windows que permite "quemar" imágenes ISO a memorias USB de forma que estas sean arrancables (booteable). Se observa que inicialmente no había dispositivos conectados.

![Figura 4 — Configuración de Rufus con esquema GPT](imagenes/4.png)

**Figura 4:** Rufus configurado con el esquema de partición GPT y sistema de destino UEFI. Esta configuración es fundamental: si se dejara en MBR, el instalador no podría coexistir correctamente con el gestor de arranque de Windows (que ya está en modo UEFI/GPT), causando que el menú Dual Boot no apareciera al reiniciar.

**Procedimiento para llegar a estas imágenes:** Se abrió Rufus en Windows, se conectó una memoria USB de 8 GB, se seleccionó el archivo ISO descargado previamente y se configuraron los parámetros de partición GPT/UEFI antes de iniciar el proceso de escritura.

---

## 5. Fase 2: Particionado del Disco en Windows

### 5.1 Reducción del Volumen C: en Administración de Discos

Antes de instalar Ubuntu, fue necesario liberar espacio en el SSD (Patriot P210 de 256 GB) donde reside Windows. Esto se hizo desde la herramienta nativa **Administración de discos** de Windows.

**Procedimiento:**
1. Clic derecho en el botón Inicio → **Administración de discos**.
2. Clic derecho sobre la partición `C:` (NTFS, ~237.93 GB) → **Reducir volumen**.
3. Se ingresó **60,000 MB** (~60 GB) como espacio a reducir.
4. Se confirmó la operación, generando un bloque de **~58.59 GB** marcado como "Espacio no asignado" (color negro).

**Decisión de diseño:** Se eligieron 60 GB para Ubuntu porque:
- Ubuntu Server mínimo requiere ~5 GB.
- Se necesitan ~15-20 GB adicionales para entornos de Python (Conda/Pip), librerías de IA (PyTorch, TensorFlow, Ray) y archivos temporales.
- El espacio restante (~35 GB) proporciona margen para logs, modelos intermedios y asignación swap.

![Figura 5 — Rufus terminado exitosamente](imagenes/5.png)

**Figura 5:** Administración de discos de Windows mostrando el resultado de la operación de reducción del volumen. Se observan claramente: la partición C: reducida, y un bloque de **58.59 GB** de espacio **no asignado** (en negro) en el Disco 1 (SSD Patriot P210). Este espacio fue deliberadamente dejado sin formato ni asignación de volumen para que el instalador de Ubuntu lo reconociera como espacio disponible.

**Procedimiento para llegar a esta imagen:** Tras crear la USB booteable con Rufus, se abrió Administración de discos (`diskmgmt.msc`), se identificó el SSD (Disco 1), se redujo el volumen C: generando el espacio no asignado, y se verificó que apareciera correctamente antes de proceder al reinicio.

![Figura 6 — Administración de discos con espacio no asignado](imagenes/6.png)

**Figura 6:** Rufus muestra el estado "PREPARADO" (barra verde) indicando que la USB booteable fue creada exitosamente. El proceso tomó varios minutos dependiendo de la velocidad de escritura de la USB.

### 5.2 Desactivación de Inicio Rápido

Windows 10 incluye una función llamada **Inicio rápido** (Fast Startup) que hiberna parcialmente el sistema al apagar, bloqueando los discos para escritura externa. Esto puede impedir que Linux monte correctamente las particiones NTFS.

**Procedimiento:**
1. Panel de Control → Hardware y sonido → Opciones de energía.
2. "Elegir el comportamiento de los botones de inicio/apagado".
3. "Cambiar la configuración actualmente no disponible".
4. **Desactivar** la casilla "Activar inicio rápido (recomendado)".
5. Guardar cambios.

**Justificación técnica:** Con el inicio rápido activo, Windows escribe un archivo de hibernación (`hiberfil.sys`) que marca los volúmenes NTFS como "sucios". Linux, al detectar esta marca, rechaza montar la partición en modo lectura-escritura para evitar corrupción de datos. Desactivar esta función asegura que el apagado de Windows sea completo y los discos queden liberados.

---

## 6. Fase 3: Instalación de Ubuntu Server 24.04 (Dual Boot)

### 6.1 Arranque desde USB — Menú GRUB

Con la USB conectada al PC de mesa, se reinició el equipo y se accedió al **Boot Menu** de la BIOS (tecla F11 o F12 en la mayoría de placas American Megatrends). Se seleccionó la opción **"UEFI: [Nombre de la USB]"** para arrancar desde el medio de instalación.

![Figura 7 — Menú GRUB del instalador USB](imagenes/7.jpeg)

**Figura 7:** Menú GRUB (Grand Unified Bootloader) que aparece al arrancar desde la USB de Ubuntu Server. Este menú se presenta al usuario al bootear desde la memoria USB y ofrece cuatro opciones:

| Opción | Descripción Técnica |
|:---|:---|
| **Try or Install Ubuntu Server** | Opción estándar. Carga el instalador en RAM para el proceso de instalación. Es la ruta normal para instalar el sistema. |
| **Ubuntu Server with the HWE kernel** | *Hardware Enablement*: kernel más reciente para hardware extremadamente nuevo. Innecesario para el Athlon 3000G (hardware bien soportado). |
| **Boot from next volume** | Salta la USB y arranca desde el siguiente dispositivo (el disco de Windows). Útil si se entró a la USB por error. |
| **UEFI Firmware Settings** | Acceso directo a la BIOS/UEFI de la placa base, sin necesidad de adivinar la tecla de acceso (F2, Del, etc.). |

**Procedimiento para llegar a esta imagen:** Se apagó el PC de mesa desde Windows con la opción de inicio rápido desactivada, se conectó la USB booteable, se encendió el equipo y se presionó repetidamente la tecla del Boot Menu (F11) al ver el logo de la placa base. Se seleccionó la opción UEFI de la USB, lo cual cargó el GRUB del instalador.

Se seleccionó **"Try or Install Ubuntu Server"** y se presionó Enter.

### 6.2 Selección del Tipo de Instalación

![Figura 8 — Selección del tipo de instalación](imagenes/8.jpeg)

**Figura 8:** Pantalla de selección del tipo de instalación de Ubuntu Server. Las opciones disponibles son:

- **(X) Ubuntu Server:** Instalación estándar con un conjunto de herramientas básicas de administración (utilidades de red, editores de texto, etc.). Sigue siendo extremadamente ligero (~500 MB de RAM al arrancar). **Esta fue la opción seleccionada.**
- **Ubuntu Server (Minimized):** Versión ultramínima diseñada para microservicios en la nube donde los humanos raramente interactúan. Carece de herramientas esenciales como `curl`, `wget` y editores de texto que serían necesarios para la configuración manual.
- **Search for third-party drivers:** Opción adicional para que el instalador busque controladores propietarios para el hardware AMD Athlon y gráficos Radeon Vega.

**Nota sobre error posterior:** En el primer intento de instalación, se activó la opción "Search for third-party drivers". Esto causó un error durante la instalación (ver Sección 6.8). En intentos posteriores se desactivó esta opción, resolviendo el problema.

### 6.3 Configuración de Red

![Figura 9 — Configuración de red automática](imagenes/9.jpeg)

**Figura 9:** Pantalla de configuración de red del instalador. La interfaz de red `enp5s0` (correspondiente al controlador Realtek PCIe GbE de la placa base) fue detectada automáticamente. El servidor DHCP del router asignó la dirección IP **192.168.1.3/24**, idéntica a la que Windows tenía asignada. Esto indica que el router está reservando la misma IP para la dirección MAC de esa tarjeta de red, lo cual es ideal para la configuración NFS posterior (IP estable).

**Procedimiento:** Se conectó el cable Ethernet al router antes de iniciar la instalación. El instalador detectó la interfaz automáticamente y obtuvo configuración IP vía DHCP. Se presionó "Done" sin modificar nada.

### 6.4 Configuración de Almacenamiento — El Desafío Crítico

Esta fue la sección más compleja y problemática de toda la instalación. El instalador de Ubuntu Server (Subiquity) ofrece dos modos:

- **Use an entire disk** (modo guiado): Borra todo el disco seleccionado. **PELIGROSO** — habría eliminado Windows y todos los datos.
- **Custom storage layout** (modo manual): Permite seleccionar particiones específicas. **OBLIGATORIO** para Dual Boot.

![Figura 10 — Pantalla inicial de almacenamiento (modo guiado)](imagenes/10.jpeg)

**Figura 10:** Pantalla de configuración de almacenamiento en modo guiado. <u>El instalador muestra el disco TOSHIBA de 1 TB como selección predeterminada.</u> Si se hubiera procedido con esta opción, se habrían borrado todos los datos del disco de 1 TB (juegos, archivos personales, etc.). Se cambió a **Custom storage layout** para control manual de las particiones.

![Figura 11 — Vista de discos en modo personalizado](imagenes/11.jpeg)

**Figura 11:** Vista de los discos detectados en modo de **almacenamiento personalizado** (Custom Storage Layout). Se observan dos discos físicos:

1. **Patriot P210 256GB** (SSD): Con tres particiones visibles — la EFI (partition 1), Windows NTFS (partition 2, 179 GB) y Recuperación de Windows (partition 3, 535 MB). **El espacio libre de ~58.6 GB no se muestra** explícitamente como una línea.
2. **TOSHIBA MQ04ABF100** (HDD 1TB): Con tres particiones — EFI (partition 1, 100 MB), MSR/Reserved (partition 2, 16 MB) y Datos NTFS (partition 3, 931.4 GB).

**Problema identificado:** El espacio no asignado de 58.6 GB creado en Windows estaba posicionado entre la partición 2 (Windows) y la partición 3 (Recuperación). El instalador Subiquity no lo renderizaba como una línea seleccionable "FREE SPACE", impidiendo crear directamente la partición de Ubuntu.

![Figura 12 — Menú contextual del disco Patriot](imagenes/12.jpeg)

**Figura 12:** Menú contextual al seleccionar la cabecera del disco Patriot P210. Las opciones disponibles son limitadas: información del disco, reformateo y uso como dispositivo de arranque. No se muestra la opción requerida "Add GPT Partition" debido a que el espacio libre no es reconocido por la interfaz del instalador.

![Figura 13 — Opciones de la partition 1 del disco Toshiba](imagenes/13.jpeg)

**Figura 13:** Menú de opciones al seleccionar la partition 1 (EFI, 100 MB, vfat) del disco Toshiba. Se intentó usarla como partición de arranque para el Dual Boot, pero las opciones disponibles (Format, Edit, Delete, Info) no incluían directamente "Use As Boot Device" en este contexto. La opción de edición (Edit) mostraba los campos Format y Mount en estado inhabilitado (gris).

#### Intentos de resolución del espacio "invisible"

Los siguientes pasos documentan los múltiples intentos por resolver el problema del espacio libre no visible en el instalador. Se intentaron diversas estrategias antes de llegar a la solución definitiva con `cfdisk`.

![Figura 14 — Intento de eliminar partition 3 del Patriot](imagenes/14.jpeg)

**Figura 14:** Intento de eliminar la partición de recuperación de Windows (partition 3, 535 MB) del disco Patriot para fusionar su espacio con los 58.6 GB no asignados y crear un bloque contiguo que el instalador pudiera reconocer.

![Figura 15 — Opciones de edición de partición](imagenes/15.jpeg)

**Figura 15:** Ventana de edición de una partición existente. Se observan opciones de formato (ext4, xfs, btrfs, swap, dejar sin formato) y punto de montaje. Sin embargo, la eliminación no estaba disponible para la partición de recuperación, ya que el instalador la protegía por seguridad.

![Figura 16 — Navegación entre opciones de formato](imagenes/16.jpeg)

**Figura 16:** Exploración de las opciones de formato disponibles en el editor de particiones del instalador. Se muestran los sistemas de archivos soportados: ext4 (estándar Linux), xfs (alto rendimiento), btrfs (copy-on-write), y swap (memoria virtual).

![Figura 17 — Intento de configuración de partición](imagenes/17.jpeg)

**Figura 17:** Continuación de la exploración de opciones de particionado. El instalador no permitía borrar la partición de recuperación y el espacio libre seguía sin aparecer como seleccionable.

![Figura 18 — Estado de la configuración durante los intentos](imagenes/18.jpeg)

**Figura 18:** Vista del estado del instalador durante los múltiples intentos de configuración. El mensaje superior continúa indicando que no hay discos montados, evidenciando que aún no se había logrado crear la partición raíz.

![Figura 19 — Vista completa de la configuración parcial](imagenes/19.jpeg)

**Figura 19:** Panorámica completa de la pantalla de configuración de almacenamiento después de los intentos fallidos. Tanto el resumen del sistema de archivos como las particiones visibles permanecen sin cambios significativos. Se decidió recurrir a la **terminal de emergencia** del instalador como solución definitiva.

### 6.5 Resolución con cfdisk: Particionado Manual desde Terminal

Ante la imposibilidad de crear la partición desde la interfaz gráfica del instalador (Subiquity), se optó por una solución de ingeniería directa: usar la herramienta de particionado en modo texto `cfdisk` desde una terminal de emergencia.

**Acceso a la terminal:**
Dentro del instalador, se presionó **`Ctrl + Alt + F2`** para acceder a una shell de texto alternativa (TTY2). Esta es una característica estándar de los sistemas Linux: múltiples terminales virtuales accesibles mediante combinaciones de teclas.

**Identificación del disco:**
```bash
lsblk
```
Se confirmó que el disco SSD Patriot P210 correspondía a `/dev/sdb` (sdb1=EFI, sdb2=Windows, sdb3=Recovery).

**Apertura de cfdisk:**
```bash
sudo cfdisk /dev/sdb
```

`cfdisk` es una herramienta de particionado interactiva basada en curses (interfaz de texto con menús navegables). Opera directamente sobre la tabla de particiones del disco, permitiendo crear, eliminar y modificar particiones sin las restricciones de la interfaz del instalador.

![Figura 20 — Terminal de emergencia y lsblk](imagenes/20.jpeg)

**Figura 20:** Terminal de emergencia (TTY2) del instalador con la salida del comando `lsblk`. Se identifica claramente la estructura de discos: `sda` (Toshiba 1 TB) y `sdb` (Patriot SSD 256 GB). La partición `sdb5` (que se creará más adelante) corresponderá al espacio donde se instalará Ubuntu.

![Figura 21 — Interfaz de cfdisk mostrando Free Space](imagenes/21.jpeg)

**Figura 21:** Interfaz de `cfdisk` abierta sobre `/dev/sdb` (Patriot SSD). **Aquí se revela lo que el instalador gráfico no mostraba:** una línea de **Free Space de 58.6G** claramente visible entre la partición 2 y la partición 3. Las opciones del menú inferior muestran: `[New]` para crear una nueva partición, `[Quit]` para salir, `[Help]` para ayuda, `[Write]` para escribir los cambios y `[Dump]` para volcado de información.

![Figura 22 — Partición creada exitosamente en cfdisk](imagenes/22.jpeg)

**Figura 22:** Resultado después de crear la nueva partición en `cfdisk`. La línea "Linux filesystem 58.6G" confirma que la partición fue creada exitosamente. Se seleccionó `[Write]` y se confirmó con `yes` para escribir la tabla de particiones al disco. Luego se salió con `[Quit]`.

**Procedimiento completo para llegar a estas imágenes:**
1. Estando en la pantalla naranja del instalador de almacenamiento, se presionó `Ctrl+Alt+F2` para abrir la terminal alternativa.
2. Se ejecutó `lsblk` para identificar que el SSD Patriot era `/dev/sdb`.
3. Se ejecutó `sudo cfdisk /dev/sdb`, que abrió la interfaz retro de particionado.
4. Se navegó con las flechas hasta la línea "Free space" de 58.6G.
5. Se seleccionó `[New]`, se dejó el tamaño predeterminado (todo el espacio disponible) y se presionó Enter.
6. Se seleccionó `[Write]`, se escribió `yes` para confirmar los cambios en la tabla de particiones.
7. Se seleccionó `[Quit]` para salir de cfdisk.
8. Se regresó al instalador naranja con `Ctrl+Alt+F1`.

Sin embargo, al regresar al instalador y refrescar la configuración de almacenamiento, surgió un nuevo problema: el instalador no permitía configurar los puntos de montaje ni el formato de la partición recién creada, porque `cfdisk` solo creó una partición genérica "Linux filesystem" sin tipo GPT específico. Fue necesario un paso adicional.

#### Creación del esquema de particiones EFI + Root

Como el instalador de Toshiba no permitía usar su partición EFI existente como dispositivo de arranque (las opciones estaban inhabilitadas), se adoptó el **Plan B**: crear una partición EFI propia directamente en el SSD Patriot.

Se volvió a la terminal (`Ctrl+Alt+F2`) y se ejecutó nuevamente `cfdisk /dev/sdb`. Esta vez se borró la partición recién creada y se crearon dos particiones en su lugar:

1. **Partición de arranque (EFI):** 512 MB
2. **Partición del sistema (Root):** ~58 GB (espacio restante)

![Figura 23 — Partición raíz configurada, visible en resumen](imagenes/23.jpeg)

**Figura 23:** Regreso al instalador Subiquity después del primer intento con `cfdisk`. Se observa en la parte superior el resumen del sistema de archivos mostrando la partición raíz (`/`) de 58.5 GB configurada correctamente. Sin embargo, aparece el mensaje de advertencia: **"To continue you need to: Select a boot disk"** — indicando que falta configurar la partición de arranque EFI.

![Figura 24 — Opciones inhabilitadas al editar partición EFI del Toshiba](imagenes/24.jpeg)

**Figura 24:** Intento de editar la partición 1 del disco Toshiba (EFI, 100 MB, vfat) para usarla como `/boot/efi`. Los campos de **Format** y **Mount** aparecen completamente inhabilitados (en gris), impidiendo cualquier modificación. El instalador Subiquity protege las particiones EFI existentes de otros sistemas operativos para evitar corrupción accidental del arranque.

![Figura 25 — Menú contextual sin opción "Use As Boot Device"](imagenes/25.jpeg)

**Figura 25:** Menú contextual de la partition 1 del disco Toshiba. Se esperaba encontrar la opción "Use As Boot Device" para designarla como partición de arranque compartida. Las opciones disponibles son limitadas a Format, Edit, Delete e Info, sin la opción de arranque. Esto confirmó la necesidad del Plan B: crear una partición EFI independiente en el SSD Patriot.

![Figura 26 — Interfaz cfdisk con las opciones de partición](imagenes/26.jpeg)

**Figura 26:** Segunda sesión de `cfdisk` sobre `/dev/sdb`. Se muestra la interfaz con las opciones del menú inferior: `[Delete]`, `[Quit]`, `[Type]`, `[Help]`, `[Write]`, `[Dump]`. En este paso se borró la partición única de 58.6 GB y se crearon dos nuevas: una de 512 MB para EFI y otra con el espacio restante (~58 GB) para el sistema de archivos raíz.

### 6.6 Asignación de Tipos de Partición (EFI System y Linux Filesystem)

Al intentar configurar las particiones en el instalador, se descubrió que la partición pequeña de 512 MB no ofrecía la opción **FAT32** como formato. La razón: `cfdisk` las había creado ambas con el tipo genérico "Linux filesystem", y el instalador Subiquity solo muestra FAT32 como opción cuando el tipo GPT de la partición es "EFI System".

![Figura 27 — Opciones de formato sin FAT32](imagenes/27.jpeg)

**Figura 27:** Pantalla de edición de la partición de 512 MB. El campo Format muestra únicamente opciones para sistemas Linux (ext4, xfs, btrfs, swap) pero **no FAT32**. Esto es porque el punto de montaje estaba configurado como `/` (raíz) y el tipo GPT de la partición era "Linux filesystem". El instalador inteligentemente oculta FAT32 cuando el montaje es `/` porque Linux no puede instalarse sobre FAT32.

**Solución:** Se regresó a la terminal (`Ctrl+Alt+F2`) y se usó la opción `[Type]` de `cfdisk` para cambiar el tipo GPT de la partición pequeña.

![Figura 28 — Lista de tipos de partición GPT en cfdisk](imagenes/28.jpeg)

**Figura 28:** Lista completa de tipos de partición GPT disponibles en `cfdisk`. Esta lista muestra todas las etiquetas estandarizadas que el firmware UEFI y los sistemas operativos utilizan para identificar el propósito de cada partición antes de intentar leerla. Los tipos más relevantes incluyen:

- **EFI System:** Partición VIP — la única que la BIOS/UEFI sabe leer nativamente antes de cargar cualquier sistema operativo. Contiene los gestores de arranque (`.efi`).
- **Microsoft Basic Data / Reserved / Recovery:** Etiquetas de Windows para datos (NTFS), espacio reservado y recuperación.
- **Linux filesystem:** Etiqueta genérica para particiones de datos Linux (ext4, xfs, btrfs).
- **Linux swap:** Memoria virtual / RAM de respaldo en disco.
- **Linux root (x86-64, ARM, IA-64...):** Tipos específicos por arquitectura de CPU para que el gestor de arranque cargue el kernel correcto.

![Figura 29 — Selección de tipo EFI System](imagenes/29.jpeg)

**Figura 29:** Selección del tipo "EFI System" para la partición de 512 MB. Al cursor se encuentra sobre la opción `EFI System` en la lista de tipos. Tras seleccionarla y confirmar con Enter, el campo "Type" de la partición cambió de "Linux filesystem" a "EFI System", lo cual es la señal que el instalador Subiquity necesita para ofrecer FAT32 como formato y `/boot/efi` como punto de montaje.

**Procedimiento:**
1. `Ctrl+Alt+F2` → terminal de emergencia.
2. `sudo cfdisk /dev/sdb` → abrir el particionador.
3. Navegar a la partición de 512 MB → Seleccionar `[Type]` → Elegir **"EFI System"**.
4. Navegar a la partición de ~58 GB → Verificar que su tipo sea **"Linux filesystem"**.
5. Seleccionar `[Write]` → Confirmar con `yes` → `[Quit]`.
6. `Ctrl+Alt+F1` → volver al instalador.

### 6.7 Configuración Final de Montajes en el Instalador

Después de definir correctamente los tipos de partición en `cfdisk`, se regresó al instalador y se refrescó la configuración de almacenamiento (Back → Custom storage layout → Done).

![Figura 30 — Particiones 4 y 5 visibles en el instalador](imagenes/30.jpeg)

**Figura 30:** El instalador Subiquity ahora muestra correctamente las dos nuevas particiones en el disco Patriot P210: **partition 4** (512 MB, tipo EFI) y **partition 5** (~58.1 GB, tipo Linux). Ambas aparecen como "unused" (sin usar), listas para ser configuradas con formato y punto de montaje.

Se procedió a configurar cada partición:

**Partition 4 (512 MB — Arranque):**
- Format: **fat32**
- Mount: **`/boot/efi`**

**Partition 5 (~58 GB — Sistema):**
- Format: **ext4**
- Mount: **`/`** (raíz)

![Figura 31 — Configuración de la partición 4 (EFI/boot)](imagenes/31.jpeg)

**Figura 31:** Configuración de la partition 4 (512 MB). El formato se estableció como **FAT32** (ahora disponible gracias al tipo "EFI System"). El punto de montaje fue asignado automáticamente por el instalador como `/boot/efi` al detectar el tipo EFI.

![Figura 32 — Configuración de la partición 5 (raíz)](imagenes/32.jpeg)

**Figura 32:** Configuración de la partition 5 (~58 GB). Se estableció el formato como **ext4** (sistema de archivos estándar de Linux, optimizado para rendimiento general y estabilidad) y el punto de montaje como **`/`** (raíz del sistema, donde residirá todo el sistema operativo, librerías y programas).

![Figura 33 — Resumen final del sistema de archivos](imagenes/33.jpeg)

**Figura 33:** **Resumen final del sistema de archivos** — el "semáforo verde" de la instalación. Se observan claramente las dos líneas requeridas:
1. **`/`** → 58.1 GB, ext4, en el SSD Patriot (sistema operativo).
2. **`/boot/efi`** → 512 MB, fat32, en el SSD Patriot (gestor de arranque GRUB).

El disco **Toshiba de 1 TB no aparece** en el resumen, confirmando que no será modificado durante la instalación — los datos de Windows están a salvo. El botón "Done" se activó correctamente.

### 6.8 Error de Instalación y Resolución

Durante el primer intento de instalación completa (después de configurar correctamente las particiones), se produjo un error.

![Figura 34 — Error durante la instalación](imagenes/34.jpeg)

**Figura 34:** Mensaje de error del instalador: **"Sorry, there was a problem completing the installation"**. Se ofrecen opciones para reiniciar el instalador, ver el reporte completo o cerrar el reporte. Este error ocurrió durante la fase de configuración del perfil de usuario y se atribuyó a:

1. **Conflicto de drivers:** La opción "Search for third-party drivers" estaba activada. El instalador intentó descargar y cargar controladores propietarios para los gráficos Radeon Vega, y este proceso falló (posiblemente por incompatibilidad o timeout de red).
2. **Datos residuales:** Los múltiples intentos previos de particionado dejaron datos parciales en las particiones que confundían al instalador.

**Resolución:**
1. Se reinició completamente el PC desde la USB.
2. En la pantalla de tipo de instalación, se **desmarcó** la opción "Search for third-party drivers".
3. Se repitió el proceso de particionado manual (las particiones ya existían de los pasos anteriores).
4. Se **re-formatearon** ambas particiones (fat32 para `/boot/efi` y ext4 para `/`) en lugar de usar la opción "Leave unchanged", eliminando cualquier dato residual.
5. La instalación procedió exitosamente.

### 6.9 Configuración Post-Instalación (Ubuntu Pro, SSH, Snaps)

Tras resolver el error, la instalación avanzó correctamente por las siguientes pantallas de configuración:

![Figura 35 — Pantalla de Ubuntu Pro](imagenes/35.jpeg)

**Figura 35:** Pantalla de "Upgrade to Ubuntu Pro". Se seleccionó **"Skip for now"** (Saltar por ahora). Ubuntu Pro es un servicio de Canonical con parches de seguridad extendidos (10 años) y herramientas de compliance, orientado a entornos empresariales de producción. Para un proyecto universitario, la versión estándar con 5 años de soporte LTS es más que suficiente.

![Figura 36 — Configuración de SSH](imagenes/36.jpeg)

**Figura 36:** Pantalla de configuración del servidor SSH. Se marcó la opción **"Instalar servidor OpenSSH"** con `[X]`. Esta es quizás la opción más crítica de toda la instalación post-particionado: sin el servidor SSH, sería imposible controlar el PC de mesa remotamente desde el portátil. La autenticación por contraseña también se dejó habilitada para la conexión inicial.

**¿Por qué SSH es fundamental?** Ubuntu Server no tiene interfaz gráfica. La única forma de interactuar con el servidor es por terminal. Con SSH instalado, desde el portátil se puede ejecutar `ssh usuario@192.168.1.3` y obtener una terminal completa del servidor como si se estuviera sentado frente a él — sin necesidad de monitor, teclado ni ratón conectados al PC de mesa.

![Figura 37 — Pantalla de importación de llaves SSH](imagenes/37.jpeg)

**Figura 37:** Submenú de importación de llaves SSH desde GitHub. Esta pantalla apareció automáticamente al activar la instalación de OpenSSH. Se seleccionó **"Cancelar"** ya que la importación de llaves SSH desde una cuenta de GitHub es una funcionalidad avanzada de autenticación sin contraseña que no era necesaria en esta etapa. La autenticación por contraseña configurada es suficiente para la red local.

![Figura 38 — Snaps de servidor destacados](imagenes/38.jpeg)

**Figura 38:** Pantalla de "Featured Server Snaps" mostrando paquetes de software preconfigurados disponibles para instalación directa. Opciones mostradas incluyen:

| Snap | Función | ¿Relevante para el proyecto? |
|:---|:---|:---|
| **microk8s** | Kubernetes ligero (orquestación de contenedores) | No — consume demasiada RAM |
| **nextcloud** | Nube privada tipo Google Drive | No — NFS es más eficiente para este caso |
| **docker** | Contenedores de aplicaciones | No ahora — se instalaría después si se necesita |
| **powershell** | Terminal de Microsoft para Linux | No — Bash es nativo y más apropiado |
| **prometheus** | Monitoreo de recursos (CPU, RAM) | Interesante pero no prioritario |
| **aws-cli / google-cloud-sdk** | CLIs de nube | No relevante para clúster local |
| **mosquitto** | Broker MQTT para IoT | No relevante |
| **lxd** | Contenedores de sistema completo | No necesario |
| **etcd** | Base de datos distribuida clave-valor | No necesario sin Kubernetes |
| **canonical-livepatch** | Actualizaciones de kernel sin reinicio | No necesario en entorno universitario |

**Decisión:** No se instaló ningún snap adicional. El objetivo era mantener el servidor lo más ligero posible para maximizar la RAM disponible (~14 GB) para el entrenamiento de modelos de Deep Learning. Cualquier herramienta adicional puede instalarse posteriormente con `sudo snap install <paquete>`.

### 6.10 Reinicio y Verificación del Dual Boot

Al finalizar la instalación, el sistema mostró el mensaje "Reboot Now". El protocolo de reinicio fue:

1. Se presionó "Reiniciar ahora" con la USB todavía conectada.
2. Al aparecer el mensaje **"Please remove the installation medium, then press ENTER"**, se desconectó la USB físicamente.
3. Se presionó Enter.
4. El PC se reinició y presentó el menú **GRUB** (Dual Boot) con las opciones:
   - **Ubuntu** (primera opción, seleccionada automáticamente tras 10 segundos).
   - Advanced options for Ubuntu.
   - **Windows Boot Manager** (para acceder a Windows cuando sea necesario).

Se dejó arrancar Ubuntu. La pantalla presentó un prompt de login en modo texto (terminal negra con letras blancas) solicitando usuario y contraseña — exactamente el comportamiento esperado de un Ubuntu Server sin GUI.

---

## 7. Fase 4: Montaje del Disco de Datos (HDD 1TB)

### 7.1 Identificación del Disco con lsblk

Una vez logueado en el servidor, el primer paso fue identificar los discos disponibles y sus UUIDs (identificadores únicos universales) para configurar el montaje permanente del disco de 1 TB.

```bash
lsblk -f
```

![Figura 39 — Salida de lsblk -f en Ubuntu Server instalado](imagenes/39.jpeg)

**Figura 39:** Salida del comando `lsblk -f` en el Ubuntu Server recién instalado. Se identifican claramente los dispositivos:

| Dispositivo | Sistema de archivos | UUID | Punto de montaje | Descripción |
|:---|:---|:---|:---|:---|
| `sda1` | vfat | — | — | EFI del Toshiba (sin usar por Linux) |
| `sda2` | — | — | — | MSR del Toshiba (reservado Windows) |
| **`sda3`** | **ntfs** | **`98DEB2EFDEB2C4B2`** | — | **Datos del HDD 1TB** — Etiqueta: "disco_solido" |
| `sdb1` | vfat | — | — | EFI original del Patriot (Windows) |
| `sdb2` | ntfs | — | — | Windows C: |
| `sdb4` | vfat | — | `/boot/efi` | Partición EFI de Ubuntu (512 MB) |
| `sdb5` | ext4 | — | `/` | Sistema Ubuntu (58 GB) |

La pieza clave es **`sda3`** con UUID `98DEB2EFDEB2C4B2` — es la partición NTFS del disco Toshiba de 1 TB donde se almacenarán los datasets.

**Nota curiosa:** La etiqueta del volumen dice "disco_solido" a pesar de ser un disco mecánico (HDD). Esta etiqueta fue asignada originalmente en Windows por el usuario.

### 7.2 Configuración de /etc/fstab

El archivo `/etc/fstab` (*File System Table*) es la "lista de tareas de arranque" de Linux. Cada vez que el computador se enciende, el kernel lee este archivo para saber qué discos tiene conectados y en qué carpetas debe "enchufarlos" (montarlos).

**Sin esta configuración:** El disco de 1 TB sería invisible para el sistema tras cada reinicio. Se tendría que ejecutar manualmente el comando `mount` cada vez, y si un script de entrenamiento intentara escribir en `/mnt/hdd` sin el disco montado, los datos terminarían en el SSD de 58 GB, llenándolo y colapsando el servidor.

**Procedimiento:**

```bash
# 1. Crear el directorio de montaje
sudo mkdir -p /mnt/hdd

# 2. Editar el archivo fstab
sudo nano /etc/fstab
```

Se agregó la siguiente línea al final del archivo:

```
UUID=98DEB2EFDEB2C4B2  /mnt/hdd  ntfs  defaults,uid=1000,gid=1000,umask=022  0  0
```

**Desglose de cada campo:**

| Campo | Valor | Significado |
|:---|:---|:---|
| **UUID** | `98DEB2EFDEB2C4B2` | Identificador único del disco de 1 TB. Más confiable que usar `/dev/sda3` porque los nombres de dispositivo pueden cambiar entre reinicios |
| **Punto de montaje** | `/mnt/hdd` | Carpeta donde aparecerán los archivos del disco |
| **Tipo** | `ntfs` | Sistema de archivos (formato Windows) — se mantiene NTFS para preservar compatibilidad con el arranque dual |
| **Opciones** | `defaults` | Opciones estándar: montaje automático al arrancar, lectura-escritura habilitada, ejecución de programas permitida |
| | `uid=1000` | Asigna la propiedad de todos los archivos al usuario con ID 1000 (el primer usuario creado, `[usuario_servidor]	`) — necesario porque NTFS no entiende nativamente los permisos de Linux |
| | `gid=1000` | Asigna el grupo principal al mismo usuario |
| | `umask=022` | Permisos resultantes: 755 (dueño=rwx, grupo=rx, otros=rx). El dueño puede leer, escribir y ejecutar; los demás solo pueden leer y ejecutar |
| **Dump** | `0` | No realizar copias de seguridad automáticas de este disco |
| **Pass** | `0` | No verificar errores del disco durante el arranque (evita que el boot se bloquee si el disco NTFS tiene un error menor de Windows) |

**Aplicación y verificación:**
```bash
sudo mount -a     # Ejecuta todas las entradas de fstab sin reiniciar
df -h              # Verifica el montaje
```

### 7.3 El Concepto de Montaje en Linux vs. Windows

Una pregunta técnica crítica surgida durante el proceso fue: *¿Por qué Linux no detecta el disco automáticamente como lo hace Windows?*

La respuesta radica en la **filosofía de diseño** de cada sistema:

| Característica | Windows | Linux (especialmente Server) |
|:---|:---|:---|
| **Paradigma** | Orientado al usuario final | Orientado al administrador de sistemas |
| **Identificación** | Letras automáticas (C:, D:, E:) | Carpetas manuales (/mnt/datos, /media/usb) |
| **Detección** | Asume que quieres usar el disco | Espera instrucciones explícitas del administrador |
| **Riesgo de colisión** | Bajo (letras únicas auto-asignadas) | Nulo (el admin controla cada ruta) |
| **Seguridad** | Menor (cualquier programa accede) | Mayor (permisos granulares por montaje) |

**¿Por qué es mejor la forma de Linux para un servidor de IA?**

1. **Ruta fija:** El servidor NFS necesita saber que `/mnt/hdd/datasets` **siempre** apuntará al disco de 1 TB. Si Linux asignara rutas automáticamente, al conectar un USB podría cambiar el orden y romper la configuración de red.
2. **Control de permisos:** La línea de `fstab` define exactamente quién puede leer/escribir, protegiendo los datos de accesos no autorizados.
3. **Estabilidad de arranque:** Un servidor que "adivina" dónde poner discos es un servidor que eventualmente fallará.

### 7.4 Verificación del Montaje

![Figura 40 — df -h mostrando el HDD montado exitosamente](imagenes/40.jpeg)

**Figura 40:** Salida del comando `df -h` confirmando el montaje exitoso del disco de 1 TB. La línea clave:

```
/dev/sda3  932G  106G  827G  12% /mnt/hdd
```

El disco Toshiba de 1 TB aparece montado en `/mnt/hdd` con **827 GB libres**, espacio más que suficiente para el dataset de 55 GB, modelos intermedios, logs de entrenamiento y versiones procesadas de los datos. Además, se ejecutó `ls /mnt/hdd` confirmando la visibilidad de las carpetas existentes de Windows (Epic Games, Riot Games, FL, etc.).

**Procedimiento para llegar a esta imagen:** Después de editar `/etc/fstab` con `sudo nano`, guardar con `Ctrl+O` y salir con `Ctrl+X`, se ejecutó `sudo mount -a` (sin errores) y luego `df -h` para verificar. El resultado positivo también implicó que las opciones de `fstab` (`uid`, `gid`, `umask`) fueron correctamente interpretadas por el driver NTFS de Linux.

---

## 8. Fase 5: Configuración del Servidor NFS

### 8.1 Instalación y Configuración en el Servidor (PC de Mesa)

Con el disco de datos montado y verificado, se procedió a instalar y configurar el **servidor NFS** (Network File System) que permitirá al portátil acceder a los datos del disco de 1 TB a través de la red local.

```bash
# Actualizar índice de paquetes
sudo apt update

# Instalar el servidor NFS del kernel
sudo apt install nfs-kernel-server -y

# Crear el directorio para los datasets dentro del HDD
sudo mkdir -p /mnt/hdd/datasets

# Asignar la propiedad al usuario para evitar errores de permisos
sudo chown -R $USER:$USER /mnt/hdd/datasets
```

**Explicación de cada comando:**

- `apt update`: Sincroniza el índice local de paquetes con los repositorios remotos de Ubuntu. Necesario antes de cualquier instalación para asegurar que se descargue la versión más reciente.
- `nfs-kernel-server`: Implementación de NFS integrada en el kernel de Linux. Es la más eficiente y con menor overhead (consumo de recursos) comparada con implementaciones en espacio de usuario.
- `mkdir -p /mnt/hdd/datasets`: Crea la carpeta dentro del disco de 1 TB montado. La flag `-p` crea directorios intermedios si no existen.
- `chown -R $USER:$USER`: Cambia la propiedad recursivamente al usuario actual, necesario para que el servidor NFS pueda leer y escribir sin restricciones de permisos.

### 8.2 Archivo /etc/exports — Reglas de Compartición

El archivo `/etc/exports` es equivalente a una "lista de invitados" para el servidor de archivos. Define qué carpetas se comparten, con quién y bajo qué condiciones.

```bash
sudo nano /etc/exports
```

Se agregó la siguiente línea:

```
/mnt/hdd/datasets *(rw,sync,no_subtree_check,no_root_squash)
```

**Desglose de las opciones de exportación:**

| Opción | Significado | Justificación |
|:---|:---|:---|
| `/mnt/hdd/datasets` | Carpeta a compartir | Subdirectorio específico dentro del HDD |
| `*` | Permitir acceso desde cualquier IP | Simplifica la configuración en red doméstica (puede restringirse a una IP específica para mayor seguridad) |
| `rw` | Lectura y escritura (Read/Write) | Necesario para que el portátil pueda guardar resultados de entrenamiento, logs y modelos intermedios |
| `sync` | Escritura síncrona | Los datos se escriben físicamente al disco antes de confirmar la operación, protegiendo contra corrupción si hay un corte de energía durante el entrenamiento |
| `no_subtree_check` | Desactivar verificación de subárbol | Mejora el rendimiento eliminando una comprobación de ruta que es innecesaria cuando se exporta un directorio simple |
| `no_root_squash` | Permitir acceso root remoto sin mapeo | Permite que si el portátil actúa como root, tenga privilegios root sobre la carpeta compartida |

**Activación de los cambios:**

```bash
sudo exportfs -ar                          # Re-exportar todas las entradas
sudo systemctl restart nfs-kernel-server   # Reiniciar el servicio NFS
```

### 8.3 Verificación de IP del Servidor

Para que el portátil sepa a dónde conectarse, se obtuvo la dirección IP del servidor:

```bash
ip a
```

![Figura 41 — Configuración NFS completada y dirección IP del servidor](imagenes/41.jpeg)

**Figura 41:** Terminal del servidor (PC de mesa) mostrando la ejecución exitosa de todos los comandos de configuración NFS. Se observa:

1. La exportación del directorio configurada con `exportfs -ar` (sin errores).
2. El reinicio del servicio NFS completado.
3. La salida de `ip a` mostrando la interfaz `enp5s0` con la dirección IP **`192.168.1.3/24`** — esta es la dirección que el portátil usará para conectarse al servidor de datos.

**Procedimiento para llegar a esta imagen:** Después de montar el HDD, se ejecutaron secuencialmente los comandos de instalación de NFS, creación del directorio de datasets, edición de `/etc/exports`, reinicio del servicio, y finalmente `ip a` para anotar la IP del servidor.

---

## 9. Fase 6: Configuración del Cliente NFS (Portátil)

### 9.1 Instalación del Cliente y Montaje Remoto

Con el servidor NFS activo en el PC de mesa, se pasó al portátil (Ubuntu Desktop 24.04) para configurar el acceso al disco compartido.

```bash
# Instalar las herramientas cliente NFS
sudo apt update && sudo apt install nfs-common -y

# Crear el punto de montaje local
sudo mkdir -p /mnt/datasets

# Montar el recurso compartido desde el servidor
sudo mount -t nfs 192.168.1.3:/mnt/hdd/datasets /mnt/datasets
```

**Explicación técnica del comando de montaje:**

- `-t nfs`: Especifica el tipo de sistema de archivos como NFS (Network File System).
- `192.168.1.3:/mnt/hdd/datasets`: Dirección IP del servidor seguida de la ruta exportada (tal como se definió en `/etc/exports`).
- `/mnt/datasets`: Directorio local donde se "proyectan" los archivos remotos. Desde el punto de vista de cualquier programa ejecutado en el portátil, los archivos aparecen como si estuvieran en un disco local.

### 9.2 Verificación de la Conexión Distribuida

![Figura 42 — df -h en portátil mostrando NFS montado exitosamente](imagenes/42.png)

**Figura 42:** Terminal del **portátil** (Ubuntu Desktop) mostrando la salida de `df -h` después del montaje NFS. La línea clave al final:

```
192.168.1.3:/mnt/hdd/datasets  932G  106G  827G  12% /mnt/datasets
```

Esta línea confirma el **éxito total de la configuración distribuida**: el portátil "ve" los 827 GB libres del disco del servidor como si fueran propios. Cualquier archivo creado por el portátil en `/mnt/datasets` se escribe físicamente en el HDD de 1 TB del PC de mesa, y viceversa — los cambios son visibles **en tiempo real** en ambas máquinas.

**Para montaje permanente (supervive reinicios):**

```bash
sudo nano /etc/fstab
# Agregar al final:
192.168.1.3:/mnt/hdd/datasets  /mnt/datasets  nfs  defaults,_netdev,rw  0  0
```

La opción `_netdev` es un indicador técnico que dice: "No intentes montar esta carpeta hasta que la red esté disponible". Sin esta opción, si el portátil arranca más rápido que la red Wi-Fi, intentaría conectar al servidor antes de tener IP, fallaría, y la carpeta quedaría vacía.

### 9.3 Consideraciones de Red: Wi-Fi vs. Ethernet

La conexión inicial se realizó por **Wi-Fi** por la ausencia de puerto Ethernet nativo en el portátil (Lenovo IdeaPad 3). Se documentaron las siguientes consideraciones para la migración futura a Ethernet:

| Característica | Wi-Fi (actual) | Ethernet (planificado) |
|:---|:---|:---|
| **Velocidad teórica** | 300-600 Mbps (Wi-Fi 5/6) | 1,000 Mbps (Gigabit) |
| **Latencia** | Variable, sujeta a interferencias | Constante y predecible |
| **Estabilidad** | Susceptible a microondas, muros, otras redes | Inmune a interferencias electromagnéticas |
| **Para Deep Learning** | Funcional para lotes pequeños, pero el entrenamiento es lento | Óptimo — lectura de datos en streaming sin interrupciones |

**Adaptador USB a Ethernet:** Se identificó como la solución óptima para el portátil. Los adaptadores USB 3.0 a Gigabit Ethernet son económicos, plug-and-play en Ubuntu (driver `r8152` incluido en el kernel) y proporcionan la estabilidad necesaria para transferencias sostenidas de grandes volúmenes de datos.

**Comportamiento del NFS al cambiar de red:** Cuando se cambie de Wi-Fi a Ethernet:
- La IP del **servidor** (192.168.1.3) no cambiará porque está conectado por cable y el router DHCP tiene reservación por MAC.
- La IP del **portátil** podría cambiar al conectar el adaptador Ethernet, pero esto es irrelevante: el comando `mount` apunta a la IP del servidor, no a la propia.
- Se recomienda ejecutar `sudo umount /mnt/datasets && sudo mount -a` (o simplemente reiniciar) para restablecer la conexión por la nueva interfaz de red.

---

## 10. Fundamentos Técnicos Profundos

### 10.1 Tipos de Partición GPT y su Propósito

El estándar GPT (GUID Partition Table) utiliza etiquetas UUID para identificar el propósito de cada partición. Esto permite que el firmware UEFI y los sistemas operativos identifiquen el contenido de una partición **antes** de intentar leerla, como etiquetas en cajas de una bodega.

| Categoría | Tipo | Propósito |
|:---|:---|:---|
| **Arranque** | EFI System | Partición FAT32 con los archivos `.efi` del gestor de arranque. Única partición que UEFI puede leer nativamente |
| **Windows** | Microsoft Basic Data | Particiones NTFS/FAT32 con datos de usuario o sistema |
| | Microsoft Reserved (MSR) | Espacio reservado por Windows para herramientas internas |
| | Windows Recovery | Partición de recuperación de fábrica |
| **Linux** | Linux filesystem | Partición genérica para ext4, xfs, btrfs, etc. |
| | Linux swap | Memoria virtual en disco (ampliación de RAM) |
| | Linux root (x86-64) | Partición raíz específica para arquitectura x86-64 |
| | Linux home | Partición `/home` con datos de usuarios |
| | Linux LVM | Logical Volume Manager — abstracción de discos virtuales |
| **Otras** | FreeBSD, Solaris, BIOS boot | Tipos para otros sistemas operativos y arranque heredado |

### 10.2 NFS: Arquitectura y Funcionamiento

**NFS (Network File System)** es un protocolo creado por Sun Microsystems en 1984 que permite acceder a archivos remotos como si fueran locales.

**Flujo de operación cuando un script de IA solicita un dato:**

```
Script Python (portátil)         Red Local          Servidor NFS (PC mesa)
       │                             │                        │
       ├─ open("/mnt/datasets/       │                        │
       │   imagen_542.jpg")          │                        │
       │                             │                        │
       ├─────── NFS READ request ────►                        │
       │                             │                        │
       │                             │    ├─ Lee del HDD 1TB  │
       │                             │    │ /mnt/hdd/datasets/│
       │                             │    │ imagen_542.jpg    │
       │                             │                        │
       │◄───── NFS READ response ────┤                        │
       │   (bytes de la imagen)      │                        │
       │                             │                        │
       ├─ Procesa los datos          │                        │
       │  (batch de entrenamiento)   │                        │
```

**Diferencia con Bluetooth:** Mientras Bluetooth opera en la capa PAN (Personal Area Network) con alcance de ~10 metros y velocidades de ~3 Mbps, NFS opera sobre TCP/IP en una LAN con alcances de ~100 metros por cable y velocidades de hasta 1 Gbps — órdenes de magnitud superior.

### 10.3 El Archivo /etc/fstab: Anatomía Completa

El archivo `/etc/fstab` sigue un formato de 6 columnas separadas por espacios o tabulaciones:

```
<dispositivo>  <montaje>  <tipo_fs>  <opciones>  <dump>  <pass>
```

**Opciones comunes y su significado:**

| Opción | Descripción |
|:---|:---|
| `defaults` | Equivale a `rw,suid,dev,exec,auto,nouser,async` — las opciones estándar seguras |
| `auto` / `noauto` | Montar (o no) automáticamente al ejecutar `mount -a` o durante el arranque |
| `rw` / `ro` | Lectura-escritura / Solo lectura |
| `uid=N` | Asignar propiedad de archivos al usuario con ID N (útil para NTFS/FAT32) |
| `gid=N` | Asignar grupo de archivos al grupo con ID N |
| `umask=XXX` | Máscara de permisos (se resta de 777: umask=022 → permisos 755) |
| `_netdev` | Esperar a que la red esté disponible antes de montar (para NFS, CIFS, iSCSI) |
| `nofail` | No abortar el arranque si el montaje falla |

---

## 11. Conclusiones

### Logros Técnicos

1. **Clúster funcional:** Se configuró exitosamente un clúster de cómputo distribuido de dos nodos (Master-Worker) utilizando hardware convencional de consumo, sin inversión adicional en infraestructura.

2. **Unificación de sistemas:** Se instaló Ubuntu Server 24.04 LTS en modo Dual Boot sobre el PC de mesa (que originalmente ejecutaba Windows 10), preservando la capacidad de arrancar a Windows cuando sea necesario.

3. **Almacenamiento centralizado:** Se montó el disco HDD de 1 TB como volumen permanente en Linux y se configuró NFS para compartirlo con el portátil, creando un sistema de almacenamiento en red (NAS) funcional.

4. **Acceso remoto:** Se instaló y configuró el servidor SSH, permitiendo la administración completa del PC de mesa sin necesidad de monitor, teclado o ratón dedicados.

5. **Resolución de problemas complejos:** Se documentó la resolución de un problema de particionado avanzado donde el instalador gráfico Subiquity no reconocía espacio libre "sandwiched" entre particiones, requiriendo intervención directa con `cfdisk` y configuración manual de tipos de partición GPT.

### Recursos Finales del Clúster

| Recurso | Valor Total |
|:---|:---|
| Hilos de procesamiento | 12 (8 + 4) |
| RAM disponible para IA | ~28 GB (13 + 14, sin overhead de Windows) |
| Almacenamiento para datasets | 827 GB (en disco de 1 TB) |
| Protocolo de datos | NFS v4 sobre TCP/IP |
| Protocolo de control | SSH (OpenSSH) |
| Conexión actual | Wi-Fi (con migración planificada a Ethernet) |

### Lecciones de Ingeniería

1. **La importancia de la homogeneidad:** Mantener versiones idénticas del sistema operativo y las librerías en todos los nodos es la medida más importante para evitar errores en computación distribuida.

2. **Control manual vs. automatización:** La filosofía de Linux de requerir configuración explícita (`fstab`, `exports`) es una fortaleza, no una debilidad. Proporciona estabilidad y predictibilidad — exactamente lo que un servidor de producción necesita.

3. **Particionado como disciplina:** El proceso de crear las particiones manualmente con `cfdisk` demostró que entender las tablas de particiones GPT, los tipos de sistema de archivos y la jerarquía de montaje de Linux es conocimiento fundamental para cualquier ingeniero de datos.

4. **Infraestructura primero:** Antes de escribir una sola línea de código de IA, fue necesario asegurar que los datos fueran accesibles, que las máquinas se comunicaran, y que los permisos fueran correctos. Esta mentalidad de "infraestructura como fundamento" es la base de la ingeniería de datos moderna.

### Trabajo Futuro

- Instalación del adaptador USB-Ethernet para el portátil (mejora de rendimiento de red).
- Sincronización de entornos de Python (Conda/Miniconda) con versiones idénticas en ambos nodos.
- Instalación de frameworks optimizados para CPU: Intel OneDNN, OpenVINO.
- Configuración de Ray o PyTorch Distributed (DDP) para entrenamiento distribuido efectivo.
- Implementación de DataLoaders con streaming para manejar los 55 GB sin cargarlos completos en RAM.
- Monitoreo de tráfico de red con `nload` e `iftop` durante el entrenamiento.

---

## 12. Índice de Figuras

| Figura | Descripción | Archivo |
|:---|:---|:---|
| Figura 1 | Archivo ISO de Ubuntu Server 24.04.3 descargado | `imagenes/1.png` |
| Figura 2 | Confirmación de versión Ubuntu en el portátil | `imagenes/2.png` |
| Figura 3 | Rufus abierto con imagen ISO cargada | `imagenes/3.png` |
| Figura 4 | Configuración de Rufus (GPT/UEFI) | `imagenes/4.png` |
| Figura 5 | Rufus terminado exitosamente (barra verde) | `imagenes/5.png` |
| Figura 6 | Administración de discos — espacio no asignado de 58.59 GB | `imagenes/6.png` |
| Figura 7 | Menú GRUB del instalador USB | `imagenes/7.jpeg` |
| Figura 8 | Selección del tipo de instalación (Server vs Minimized) | `imagenes/8.jpeg` |
| Figura 9 | Configuración de red automática (DHCP, 192.168.1.3) | `imagenes/9.jpeg` |
| Figura 10 | Almacenamiento modo guiado (peligro de borrado) | `imagenes/10.jpeg` |
| Figura 11 | Discos en modo personalizado (espacio libre invisible) | `imagenes/11.jpeg` |
| Figura 12 | Menú contextual del disco Patriot | `imagenes/12.jpeg` |
| Figura 13 | Opciones de partition 1 del Toshiba | `imagenes/13.jpeg` |
| Figura 14 | Intento de eliminar partición de recuperación | `imagenes/14.jpeg` |
| Figura 15 | Opciones de edición de partición | `imagenes/15.jpeg` |
| Figura 16 | Opciones de formato disponibles | `imagenes/16.jpeg` |
| Figura 17 | Intento de configuración de partición | `imagenes/17.jpeg` |
| Figura 18 | Estado durante intentos de configuración | `imagenes/18.jpeg` |
| Figura 19 | Vista completa de configuración parcial | `imagenes/19.jpeg` |
| Figura 20 | Terminal de emergencia con lsblk | `imagenes/20.jpeg` |
| Figura 21 | cfdisk mostrando Free Space de 58.6G | `imagenes/21.jpeg` |
| Figura 22 | Partición creada exitosamente en cfdisk | `imagenes/22.jpeg` |
| Figura 23 | Partición raíz en resumen, falta boot disk | `imagenes/23.jpeg` |
| Figura 24 | Opciones inhabilitadas al editar EFI del Toshiba | `imagenes/24.jpeg` |
| Figura 25 | Menú sin opción "Use As Boot Device" | `imagenes/25.jpeg` |
| Figura 26 | Segunda sesión de cfdisk con opciones de partición | `imagenes/26.jpeg` |
| Figura 27 | Formato sin FAT32 (tipo GPT incorrecto) | `imagenes/27.jpeg` |
| Figura 28 | Lista de tipos GPT en cfdisk | `imagenes/28.jpeg` |
| Figura 29 | Selección de tipo EFI System | `imagenes/29.jpeg` |
| Figura 30 | Particiones 4 y 5 visibles en el instalador | `imagenes/30.jpeg` |
| Figura 31 | Configuración de partition 4 (FAT32, /boot/efi) | `imagenes/31.jpeg` |
| Figura 32 | Configuración de partition 5 (ext4, /) | `imagenes/32.jpeg` |
| Figura 33 | Resumen final del sistema de archivos | `imagenes/33.jpeg` |
| Figura 34 | Error de instalación (third-party drivers) | `imagenes/34.jpeg` |
| Figura 35 | Pantalla Ubuntu Pro (Skip for now) | `imagenes/35.jpeg` |
| Figura 36 | Instalación de servidor OpenSSH | `imagenes/36.jpeg` |
| Figura 37 | Importación de llaves SSH (cancelado) | `imagenes/37.jpeg` |
| Figura 38 | Featured Server Snaps (ninguno seleccionado) | `imagenes/38.jpeg` |
| Figura 39 | lsblk -f en Ubuntu Server instalado | `imagenes/39.jpeg` |
| Figura 40 | df -h con HDD de 1 TB montado (827 GB libres) | `imagenes/40.jpeg` |
| Figura 41 | NFS configurado e IP del servidor (192.168.1.3) | `imagenes/41.jpeg` |
| Figura 42 | NFS montado en portátil — Clúster distribuido funcional | `imagenes/42.png` |

---

# SESIÓN 2: Optimización y Automatización

**Objetivo:** Implementar montaje automático bajo demanda con autofs, gestionar archivos de datasets, y documentar procedimientos de apagado seguro del servidor.

**Período:** Febrero 2026

---

## Tabla de Contenidos — Sesión 2

13. [Migración a Autofs: Montaje Automático Bajo Demanda](#13-migración-a-autofs-montaje-automático-bajo-demanda)
    - 13.1 [Motivación para Migrar de fstab a Autofs](#131-motivación-para-migrar-de-fstab-a-autofs)
    - 13.2 [¿Qué es Autofs?](#132-qué-es-autofs)
    - 13.3 [Implementación de Autofs](#133-implementación-de-autofs)
    - 13.4 [Consideraciones sobre el Timeout](#134-consideraciones-sobre-el-timeout)
    - 13.5 [Análisis del Incidente de Apagado Bloqueado](#135-análisis-del-incidente-de-apagado-bloqueado)
    - 13.6 [Problema de IP Dinámica](#136-problema-de-ip-dinámica)
14. [Gestión de Datasets: Copia y Descompresión de Archivos](#14-gestión-de-datasets-copia-y-descompresión-de-archivos)
    - 14.1 [Contexto](#141-contexto)
    - 14.2 [Procedimiento Implementado](#142-procedimiento-implementado)
    - 14.3 [Ventajas del Enfoque Distribuido](#143-ventajas-del-enfoque-distribuido)
    - 14.4 [Estructura Final de Directorios](#144-estructura-final-de-directorios)
15. [Conclusiones de la Sesión 2](#15-conclusiones-de-la-sesión-2)
16. [Documentos Complementarios](#16-documentos-complementarios)

---

## 13. Migración a Autofs: Montaje Automático Bajo Demanda

### 13.1 Motivación para Migrar de fstab a Autofs

Durante las primeras semanas de operación del clúster, se identificaron limitaciones con el montaje estático definido en `/etc/fstab`:

| Problema | Descripción | Impacto |
|:---|:---|:---|
| **Boot bloqueado** | Si el servidor estaba apagado al iniciar el portátil, el boot esperaba hasta 90 segundos intentando montar NFS | Retrasos innecesarios al arrancar |
| **Sin reconexión automática** | Si el servidor se reiniciaba, el portátil no reconectaba automáticamente | Requería desmontar y remontar manualmente |
| **Conexión permanente** | El NFS se mantenía montado incluso cuando no se usaba | Consumo constante de recursos de red |
| **Apagado del servidor bloqueado** | Al apagar el servidor con el portátil conectado, el sistema se bloqueaba esperando desmontar `/mnt/hdd` | **Bloqueos de 7+ minutos** (ver sección 13.5) |

### 13.2 ¿Qué es Autofs?

Autofs (Automounter) es un servicio que monta sistemas de archivos **bajo demanda** — solo cuando se accede al directorio — y los desmonta automáticamente después de un período de inactividad configurable.

```
Funcionamiento de Autofs:
═══════════════════════════

1. Usuario: ls /mnt/datasets
2. Kernel detecta acceso a punto de montaje autofs
3. Kernel notifica al demonio automount
4. automount ejecuta: mount -t nfs 192.168.1.15:/mnt/hdd/datasets /mnt/datasets
5. Operación ls procede normalmente
6. Después de N segundos sin acceso → automount ejecuta: umount /mnt/datasets
```

**Ventajas técnicas:**

- **Transparente para aplicaciones:** Scripts Python, Jupyter, y comandos de terminal funcionan idéntico
- **Resiliente a desconexiones:** Si el servidor se reinicia, el siguiente acceso reconecta automáticamente
- **Eficiente en recursos:** La conexión TCP solo se mantiene mientras se usa
- **Compatible con entrenamiento de IA:** Los accesos continuos de lectura de datos resetean el timer de inactividad

### 13.3 Implementación de Autofs

**Instalación en el portátil:**
```bash
sudo apt update && sudo apt install autofs -y
```

**Configuración de `sudo nano /etc/auto.master`**
al final añadir la siguiente linea
```
/mnt  /etc/auto.nfs  --timeout=600
```

- `/mnt`: Directorio base donde autofs gestionará subdirectorios
- `/etc/auto.nfs`: Archivo de mapeo que define las reglas específicas
- `--timeout=600`: Desmontar después de 10 minutos de inactividad

**Configuración de `/etc/auto.nfs`:**
```
datasets  -fstype=nfs,rw,soft,intr,timeo=10,retrans=3  192.168.1.15:/mnt/hdd/datasets
```

**Opciones críticas:**

| Opción | Justificación |
|:---|:---|
| `soft` | Si el servidor no responde, retorna error en vez de bloquear indefinidamente |
| `intr` | Permite cancelar operaciones con Ctrl+C si el servidor está lento |
| `timeo=10` | Timeout de 1 segundo (10 décimas) antes de reintentar |
| `retrans=3` | Intenta 3 veces antes de fallar |

**Desactivación de entrada en fstab:**
```bash
sudo nano /etc/fstab
# Comentar la línea:
# 192.168.1.15:/mnt/hdd/datasets  /mnt/datasets  nfs  defaults,_netdev,rw  0  0
```

**Activación:**
```bash
sudo systemctl restart autofs
ls /mnt/datasets  # Monta automáticamente en ~1 segundo
```

### 13.4 Consideraciones sobre el Timeout

**Caso de uso: Entrenamiento de Inteligencia Artificial**

El timeout de 600 segundos (10 minutos) fue seleccionado estratégicamente:

```python
# Escenario típico de entrenamiento:
for epoch in range(100):
    for batch in dataloader:  # Lee de /mnt/datasets cada ~2 segundos
        # Cada lectura RESETEA el timer de autofs
        images = batch['images']  # Acceso a NFS
        train_step(images)        # CPU/GPU local
    
    # Guardar checkpoint
    torch.save(model, '/mnt/datasets/checkpoints/model.pt')  # Acceso a NFS
    # Timer se resetea → NO se desmonta durante el entrenamiento
```

**Comportamiento esperado:**

| Escenario | Timer se resetea | ¿Se desmonta? |
|:---|:---:|:---|
| Script leyendo datos activamente | ✅ Cada acceso | ❌ Nunca |
| Terminal abierta en `/mnt/datasets` | ✅ Shell mantiene activo | ❌ No |
| Usuario cerró todo y se fue | ❌ Sin accesos | ✅ Después de 10 min |
| Servidor se reinicia durante entrenamiento | — | ⚠️ Error, pero portátil no se congela (soft mount) |

**Alternativas de timeout:**

- `--timeout=0`: NUNCA se desmonta automáticamente. **No recomendado** — causa bloqueos al apagar el servidor
- `--timeout=300`: 5 minutos. Bueno para desarrollo interactivo
- `--timeout=1800`: 30 minutos. Para entrenamientos con checkpoints muy espaciados

### 13.5 Análisis del Incidente de Apagado Bloqueado

**Fecha del incidente:** 12 de febrero de 2026, ~21:01 horas

**Contexto:** Se ejecutó `sudo poweroff` en el servidor sin desmontar previamente el NFS del portátil. El sistema entró en un deadlock que duró más de 7 minutos.

**Cronología extraída de logs (`journalctl -b -1`):**

```
21:01:04 - Inicio del apagado
         - ✅ Servicios NFS se detuvieron correctamente en 1 segundo

21:01:05 - Intento de desmontar /mnt/hdd
         - ⏳ Sistema esperando...

21:02:35 - ⚠️ TIMEOUT después de 90 segundos
         - systemd: "mnt-hdd.mount: Unmounting timed out. Terminating."

21:04:05 - ⛔ TIMEOUT después de 3 minutos
         - systemd envía SIGKILL a procesos:
           * umount (PID 2343)
           * mount.ntfs (PID 629)

21:04:59 - 🔴 Proceso umount BLOQUEADO (122+ segundos)
         - Estado "D" (uninterruptible sleep)
         - Esperando I/O del disco que nunca llegó

21:05:35 - ❌ FALLO FINAL (4 minutos 30 segundos)
         - systemd abandona el desmontaje
         - Procesos zombies siguen corriendo
         - Consumió: 9min 24s CPU, 9.4GB RAM

21:09:05 - 🔴 umount aún bloqueado (368+ segundos)
         - También proceso lvm bloqueado
         - Sistema completamente congelado

~21:11:00 - Apagado forzado manual (botón físico)
```

**Causa raíz identificada:**

Deadlock entre servicios NFS y montaje del disco:

```
Ciclo de bloqueo:
umount → espera → mount.ntfs → espera → file locks → espera → cliente NFS desconectado
  ↑                                                                           ↓
  └─────────────────────────── ninguno libera primero ────────────────────────┘
```

Aunque el servicio `nfs-kernel-server` se marcó como "Stopped", no esperó a que el portátil liberara los archivos abiertos. El portátil tenía autofs montado con `timeout=0` (sin desmontaje automático), manteniendo conexiones "lazy" que no se notificaron al servidor.

**Lección aprendida:** Con `timeout=600`, si han pasado más de 10 minutos desde el último uso del disco, autofs ya habrá desmontado `/mnt/datasets` automáticamente. Esto previene el 90% de los bloqueos de apagado.

Ver documento complementario: `procedimiento_apagado_servidor.md` (Sección 16).

### 13.6 Problema de IP Dinámica

#### El Problema Descubierto

Al intentar acceder a `/mnt/datasets` después de reiniciar el servidor, autofs no pudo montar el directorio:

```bash
nicolas@nznicolas:~$ ls /mnt/datasets
ls: no se puede acceder a '/mnt/datasets': No existe el archivo o el directorio
```

**Diagnóstico inicial:**

```bash
# Verificar conectividad con la IP configurada en /etc/auto.nfs
nicolas@nznicolas:~$ ping -c 2 192.168.1.3
# No hubo respuesta

# Verificar IP actual del servidor (desde el servidor)
mrsasayo_mesa@nicolasmesa:~$ ip a show enp5s0
# Mostró: inet 192.168.1.15/24
```

**Causa raíz identificada:** La IP del servidor cambió de `192.168.1.3` a `192.168.1.15` después de un reinicio. El router asigna direcciones IP dinámicamente mediante DHCP sin reservación, lo cual significa que cada vez que el servidor se reinicia o el lease DHCP expira, puede recibir una IP diferente.

#### Solución Temporal: Actualización del Archivo de Mapeo

> 📍 **EJECUTAR EN: Portátil**

```bash
# Editar el archivo de mapeo de autofs
sudo nano /etc/auto.nfs
```

**Cambio realizado:**

```diff
- datasets  -fstype=nfs,rw,soft,intr,timeo=10,retrans=3  192.168.1.3:/mnt/hdd/datasets
+ datasets  -fstype=nfs,rw,soft,intr,timeo=10,retrans=3  192.168.1.15:/mnt/hdd/datasets
```

```bash
# Reiniciar autofs para aplicar cambios
sudo systemctl restart autofs

# Verificar funcionamiento
ls /mnt/datasets
# ✅ Funcionó correctamente
```

**Limitación de esta solución:** Si el servidor se reinicia nuevamente, la IP podría cambiar otra vez, rompiendo el montaje NFS.

#### Solución Permanente Implementada: IP Estática en el Servidor

Para evitar que la IP cambie en el futuro, se configuró una dirección IP estática directamente en el servidor utilizando Netplan.

> 📍 **EJECUTAR EN: Servidor (PC de Mesa)**

**Paso 1: Identificar la interfaz de red y gateway**

```bash
# Ver interfaces de red disponibles
ip a

# Salida relevante:
# enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP>
#     inet 192.168.1.15/24 brd 192.168.1.255 scope global dynamic

# Ver gateway (puerta de enlace)
ip route show default

# Salida:
# default via 192.168.1.1 dev enp5s0 proto dhcp src 192.168.1.15 metric 100
```

**Información recopilada:**

| Parámetro | Valor |
|:---|:---|
| Interfaz | `enp5s0` |
| IP actual (a fijar) | `192.168.1.15` |
| Máscara de subred | `/24` (255.255.255.0) |
| Gateway (router) | `192.168.1.1` |
| DNS | `8.8.8.8`, `8.8.4.4` (Google DNS, confiables) |

**Paso 2: Respaldar configuración actual**

```bash
# Crear copia de seguridad del archivo de configuración
sudo cp /etc/netplan/50-cloud-init.yaml /etc/netplan/50-cloud-init.yaml.backup

# Verificar que existe el backup
ls -l /etc/netplan/*.backup
```

**Paso 3: Editar configuración de Netplan**

```bash
sudo nano /etc/netplan/50-cloud-init.yaml
```

**Contenido ANTERIOR (configuración DHCP):**

```yaml
network:
  version: 2
  ethernets:
    enp5s0:
      dhcp4: true
```

**Contenido NUEVO (IP estática):**

```yaml
network:
  version: 2
  ethernets:
    enp5s0:
      dhcp4: no
      addresses:
        - 192.168.1.15/24
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

**Explicación de cada campo:**

| Campo | Valor | Significado |
|:---|:---|:---|
| `version: 2` | Netplan v2 | Versión del formato de configuración |
| `ethernets` | Sección | Configuración de interfaces Ethernet |
| `enp5s0` | Nombre de interfaz | Controlador Realtek PCIe GbE detectado por el kernel |
| `dhcp4: no` | Desactivar DHCP | El servidor NO solicitará IP dinámica |
| `addresses` | Lista de IPs | IPs estáticas asignadas a la interfaz |
| `192.168.1.15/24` | IP + máscara | IP fija con red clase C privada |
| `routes` | Tabla de rutas | Define cómo llegar a otras redes |
| `to: default` | Ruta predeterminada | Tráfico a cualquier red no local |
| `via: 192.168.1.1` | Gateway | Router que conecta con internet |
| `nameservers` | Servidores DNS | Traducen nombres de dominio a IPs |
| `8.8.8.8, 8.8.4.4` | DNS de Google | Públicos, rápidos y confiables |

**Paso 4: Validar sintaxis de la configuración**

```bash
# Verificar que no haya errores de sintaxis YAML
sudo netplan try

# Salida esperada:
# Do you want to keep these settings?
# 
# Press ENTER before the timeout to accept the new configuration
# 
# Changes will revert in 120 seconds
```

**¿Qué hace `netplan try`?**

Es un comando de seguridad: aplica la configuración temporalmente por 120 segundos. Si pierdes conectividad de red (por un error en la configuración), automáticamente revierte a la configuración anterior. Esto evita quedar "bloqueado fuera" del servidor.

**Si todo funciona correctamente:**

```bash
# Presionar ENTER para confirmar
# La configuración se mantiene

# Aplicar permanentemente
sudo netplan apply
```

**Paso 5: Verificación de la configuración**

```bash
# Ver la IP asignada
ip a show enp5s0 | grep inet

# Salida esperada:
# inet 192.168.1.15/24 brd 192.168.1.255 scope global enp5s0

# Verificar ruta predeterminada
ip route show default

# Salida esperada:
# default via 192.168.1.1 dev enp5s0 proto static

# Verificar DNS
resolvectl status enp5s0 | grep "DNS Servers"

# Salida esperada:
# DNS Servers: 8.8.8.8 8.8.4.4

# Probar conectividad con internet
ping -c 2 google.com

# Salida esperada:
# 64 bytes from ... icmp_seq=1 ttl=... time=...
# Conexión exitosa
```

**Paso 6: Reiniciar servidor para verificar persistencia**

```bash
sudo reboot
```

Después del reinicio:

```bash
# Verificar que la IP se mantuvo
ip a show enp5s0 | grep inet

# inet 192.168.1.15/24 (no cambió)
```

#### Verificación desde el Portátil

> 📍 **EJECUTAR EN: Portátil**

```bash
# Verificar conectividad con la IP estática
ping -c 2 192.168.1.15

# Verificar que autofs monta correctamente
ls /mnt/datasets

# Verificar montaje
df -h | grep datasets
# 192.168.1.15:/mnt/hdd/datasets  932G  196G  736G  22% /mnt/datasets
```

#### Ventajas de la IP Estática vs. Reservación DHCP

| Característica | Reservación DHCP | IP Estática (netplan) |
|:---|:---:|:---:|
| **Independencia del router** | ❌ Depende del router | ✅ Configuración en el servidor |
| **Portabilidad** | ❌ Se pierde al cambiar router | ✅ Funciona con cualquier router |
| **Control total** | ⚠️ Requiere acceso admin al router | ✅ Control desde el servidor |
| **Persistencia** | ✅ Sobrevive reinicios del servidor | ✅ Sobrevive reinicios del servidor |
| **Facilidad de configuración** | ⚠️ Varía según marca de router | ✅ Estándar en Ubuntu |
| **Mejor para servidores** | ⚠️ Aceptable | ✅ Recomendado |

**Conclusión:** Se seleccionó IP estática en el servidor porque proporciona control total sin depender de la configuración del router, que puede cambiar o ser inaccesible.

#### Estado Final

```
Configuración NFS Actualizada:
═══════════════════════════════

Servidor (PC de Mesa):
├─ IP: 192.168.1.15/24 (ESTÁTICA)
├─ Gateway: 192.168.1.1
├─ DNS: 8.8.8.8, 8.8.4.4
├─ NFS exportando: /mnt/hdd/datasets
└─ Configuración: /etc/netplan/50-cloud-init.yaml

Portátil (Cliente):
├─ Autofs configurado para: 192.168.1.15:/mnt/hdd/datasets
├─ Montaje automático bajo demanda
├─ Timeout: 600 segundos (10 minutos)
└─ Estado: Funcional y estable
```

---

## 14. Gestión de Datasets: Extracción de Archivos Pesados

### 14.1 Contexto

Para la asignatura de Analítica de Datos, se recibieron archivos comprimidos que totalizan más de 30 GB:

- `carlos_andres_ferro.zip`
- `cristian_garcia_eduardo.zip` (con estructura de tareas/proyecto ya descomprimida)
- **`Performance.zip`** (~55 GB) — Dataset principal descargado desde internet

**Desafío identificado:** Dada la limitación de almacenamiento del portátil (256 GB SSD), se decidió almacenar y descomprimir los datasets directamente en el servidor NFS.

**Descarga del dataset principal:** Por accesibilidad, `Performance.zip` (55 GB) se descargó desde internet usando Windows y se colocó directamente en el directorio `/mnt/datasets/` del disco duro del servidor. Posteriormente se apagó Windows y se accedió a Linux para gestionarlo.

### 14.2 Procedimiento Implementado

#### Fase 1: Archivos Pequeños de Asignaturas

> **EJECUTAR EN: Portátil**

**Paso 1: Copia al servidor NFS**
```bash
# Copiar archivos desde el portátil al servidor
cp /home/nicolas/Escritorio/Academico/analitica_datos/carlos_andres_ferro.zip /mnt/datasets/
cp /home/nicolas/Escritorio/Academico/analitica_datos/cristian_garcia_eduardo.zip /mnt/datasets/
```

⚠️ **Nota sobre transferencia Wi-Fi:** Como la conexión es por Wi-Fi, la copia de archivos grandes (>1 GB) demora considerable tiempo (se planea una conexion ethernet).

**Paso 2: Descompresión en el servidor**
```
⚠️ desde el servidor o por via ssh por eficiencia
⚠️ si tu disco es un hdd(mecanico), mirar fase 2, paso 3
```
```bash
cd /mnt/datasets
unzip carlos_andres_ferro.zip
unzip cristian_garcia_eduardo.zip
```

**Paso 3: Eliminación de archivos .zip originales**
```bash
# Liberar espacio eliminando los .zip ya descomprimidos
rm /mnt/datasets/carlos_andres_ferro.zip
rm /mnt/datasets/cristian_garcia_eduardo.zip
```

**Estado tras Fase 1:**
```
/mnt/datasets/
├── carlos_andres_ferro/
│   └── [contenido del proyecto de Carlos]
└── cristian_garcia_eduardo/
    └── tareas/
        └── proyecto/
            └── situacion_3/
                └── dataset/  ← Directorio creado por la descompresión
```

#### Fase 2: Dataset Grande Performance.zip (55 GB)

**Estado inicial:** `Performance.zip` (55 GB) está en `/mnt/datasets/` tras ser descargado desde Windows.

**Paso 1: Verificar ubicación actual del archivo**

> 📍 **EJECUTAR EN: Portátil**

```bash
# Confirmar que Performance.zip está en la raíz del servidor
ls -lh /mnt/datasets/Performance.zip
```

**Paso 2: Mover Performance.zip a su ubicación final**

```bash
# Mover al directorio del proyecto (ya creado por la descompresión de cristian_garcia_eduardo.zip)
mv /mnt/datasets/Performance.zip \
   /mnt/datasets/cristian_garcia_eduardo/tareas/proyecto/situacion_3/dataset/

# Verificar que se movió correctamente
ls -lh /mnt/datasets/cristian_garcia_eduardo/tareas/proyecto/situacion_3/dataset/Performance.zip
```

**Decisión sobre la extracción de Performance.zip:**

Después de evaluar las opciones, se decidió **NO descomprimir** el archivo `Performance.zip` (55 GB). En su lugar, se trabajará con él directamente en formato comprimido desde Python.

**Justificación técnica:**

1. **Eficiencia de espacio:** El archivo comprimido ocupa ~55 GB, mientras que descomprimido ocuparía significativamente más espacio en el disco mecánico (HDD)

2. **Soporte nativo en Python:** Bibliotecas como `pandas` y `zipfile` permiten leer archivos CSV directamente desde archivos ZIP sin necesidad de extracción previa:

3. **Rendimiento aceptable:** El overhead de descompresión en tiempo real es mínimo comparado con el tiempo de lectura desde un HDD mecánico

4. **Simplificación del flujo de trabajo:** Evita el proceso de extracción documentado en la sección anterior (configuración de kernel, ionice, etc.)

**Estructura final del dataset:**

```
/mnt/datasets/cristian_garcia_eduardo/tareas/proyecto/situacion_3/dataset/
├── Performance.zip  ← Archivo comprimido (se trabaja directamente con este)
└── [otros archivos del proyecto]
```

### 14.3 Ventajas del Enfoque Distribuido

| Aspecto | Ventaja |
|:---|:---|
| **Almacenamiento** | Los 256 GB del portátil se mantienen libres para sistema y software |
| **Acceso centralizado** | Todos los datasets en una ubicación, accesibles desde cualquier nodo del clúster |
| **Backups** | Una sola ubicación para respaldar (el disco de 1 TB del servidor) |
| **Persistencia** | Los datos sobreviven reinicios del portátil |
| **Colaboración** | Otros usuarios pueden acceder a los mismos datos vía NFS |
| **Eficiencia** | Archivos grandes se mantienen comprimidos, optimizando uso de espacio |

---

