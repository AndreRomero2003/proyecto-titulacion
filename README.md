

---

# Protocol Runner para Realce de Imágenes Médicas

## Rev-5 ACCEL – Optimización Experimental y Evaluación Masiva

Este repositorio contiene la implementación del protocolo experimental (**Rev-5**) asociado al manuscrito académico:

**“Mejora estructural de imágenes de radiodiagnóstico, mediante un pipeline secuencial CLAHE – Filtro Guiado – Unsharp en entornos con recursos computacionales limitados”**

El script principal `ups_rev5_protocol_runner_ACCEL.py` implementa un **sistema completo de optimización de hiperparámetros y evaluación masiva**, diseñado para encontrar configuraciones óptimas de realce estructural en imágenes médicas multimodales (RX, Mamografía, Ecografía).

> **Novedad principal (Rev-5 ACCEL):**
> El cálculo de métricas de calidad de imagen (IQA) ha sido **acelerado por GPU (CUDA)** mediante `torch` y `pyiqa`, utilizando *batch processing*, lo que reduce drásticamente los tiempos de evaluación respecto a versiones previas (Rev-4 y anteriores).

---

## 🚀 Características Principales

* **Dataset multimodal balanceado:**
  Soporte nativo para:

  * Rayos X (RX)
  * Mamografía (MAMO)
  * Ecografía (ECO)

* **Catálogo de pipelines:**
  Evaluación de **15 pipelines deterministas (P00–P15)** basados en combinaciones de:

  * C → CLAHE
  * G → Guided Filter
  * U → Unsharp Mask

* **Evaluación exhaustiva de calidad:**
  Cálculo de **49 métricas IQA**, organizadas en:

  * Fidelidad estructural

  * Preservación de bordes

  * Ruido y artefactos

  * Costo computacional

  * Métricas de auditoría

  > *Nota:* En Rev-5 se eliminaron `IW_SSIM`, `MedIQA_NR` y `energía_J` por eficiencia computacional.

* **Optimización híbrida en tres etapas:**

  1. **Grid Search**
  2. **NSGA-II (algoritmo genético multiobjetivo)**
  3. **Optuna – Bayesian Optimization (TPE)**

* **Sistema de Gates G1–G5:**
  Compuertas automáticas que descartan configuraciones que violan restricciones críticas de:

  * Fidelidad
  * Brillo
  * Ruido
  * Calidad NR
  * Costo computacional

* **Arquitectura híbrida CPU/GPU:**

  * Preprocesamiento y filtros → CPU
  * Métricas IQA pesadas → GPU

---

## 📋 Requisitos Previos

### Sistema

* **Sistema Operativo:** Linux o Windows
* **Python:** 3.9 o superior (3.10 recomendado)

### Hardware

* **CPU:** Multinúcleo (16+ hilos recomendado)
* **GPU:** NVIDIA con soporte CUDA (crítico para ACCEL)

  * El modo CPU es posible con `--deploy_cpu_only`, pero **muy lento**
* **RAM:**

  * 16 GB mínimo
  * 32 GB o más recomendado para datasets grandes

---

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 2. Crear entorno e instalar dependencias

```bash
conda create -n img_enhancement python=3.10
conda activate img_enhancement

pip install numpy opencv-contrib-python scikit-image scipy pandas \
            matplotlib psutil pydicom tqdm optuna \
            pyiqa torch torchvision
```

> **Importante:**
> Instala `torch` con soporte CUDA acorde a tu versión.
> Ver instrucciones oficiales en [https://pytorch.org/](https://pytorch.org/)

---

## 🗂️ Preparación del Dataset

### Opción A – Estructura de carpetas (recomendada)

```text
/dataset/
├── RX/
├── MAMO/
└── ECO/
```

### Opción B – Archivo CSV (manifest)

```csv
path,modality
/data/img1.png,RX
/data/img2.dcm,MAMO
/data/img3.jpg,ECO
```

---

## ▶️ Ejecución del Protocolo

### Ejecución completa

```bash
python ups_rev5_protocol_runner_ACCEL.py \
    --data_root ./mi_dataset \
    --out_dir ./resultados_run1 \
    --device cuda \
    --n_jobs 16
```

### Ejecución avanzada (pipeline específico)

```bash
python ups_rev5_protocol_runner_ACCEL.py \
    --manifest_csv ./datos.csv \
    --out_dir ./runs/experimento_P10 \
    --pipelines P10 \
    --save_outputs \
    --grid_max 2000 \
    --ga_pop 50 \
    --bo_trials 80 \
    --gpu_batch 64
```

---

## 📊 Salidas del Sistema

En `out_dir` se generan automáticamente:

1. `evidence_pack/` – Tablas y figuras listas para el manuscrito
2. `theta_star_all_pipelines.json` – Hiperparámetros óptimos
3. `per_image_metrics_holdout.csv` – Métricas por imagen
4. `stats_holdout.csv` – Estadísticos por modalidad
5. `aggregate_medians_by_pipeline.csv` – Comparativa global
6. `outputs_examples/` – Imágenes antes/después (opcional)

---

## 🧠 Detalles del Protocolo Rev-5

* **Partición fija:**

  * HPO: 60%
  * Validación: 20%
  * Hold-out: 20%

* **Selección final:**
  El mejor candidato se elige **solo en validación** y se reporta **honestamente** en hold-out.

---

# ============================================

# SEGUNDA PARTE

# Implementación Base del Pipeline (GUI en CPU)

# ============================================

## Procesador de Imágenes Médicas

### Pipeline CLAHE + Filtro Guiado + Unsharp Mask

### Información del Proyecto

**Universidad:** Universidad Politécnica Salesiana
**Carrera:** Ingeniería en Computación
**Grupo:** 67

**Autores:**

* Andre Alessandro Romero Martínez
* Daniel Luis Montaleza Ortiz

---

## Descripción General

Esta aplicación representa la **implementación base determinista en CPU** del pipeline propuesto en el artículo académico, y constituye el **punto de partida experimental** sobre el cual se construyen:

* Las grillas de hiperparámetros
* Los pipelines P00–P15
* El protocolo Rev-5 ACCEL

El sistema permite **exploración manual e interactiva** del espacio θ = {θ_C, θ_G, θ_U} mediante una interfaz gráfica.

---

## Pipeline Secuencial

1. **CLAHE**

   * ClipLimit, TileGridSize
2. **Filtro Guiado**

   * Radius, Epsilon
3. **Unsharp Mask**

   * Amount, Sigma, Threshold

---

## Interfaz Gráfica (GUI)

* CustomTkinter (tema oscuro)
* Sliders en tiempo real
* Comparación original vs procesada
* Zoom interactivo
* Métricas de CPU, RAM, GPU, tiempo
* Soporte JPG, PNG y DICOM

---

## Requisitos (GUI)

```bash
pip install opencv-python numpy Pillow customtkinter psutil pydicom pynvml
```

* **CPU:** Dual-core o superior
* **RAM:** 4 GB (8 GB recomendado)
* **GPU:** No requerida

---

## Relación con el Protocolo Rev-5

Esta implementación corresponde a:

* Definición del pipeline determinista
* Validación visual de configuraciones
* Fundamento del espacio de búsqueda
* Verificación de viabilidad en CPU

Las funciones avanzadas (optimización, métricas masivas, gates, GPU) **no se ejecutan aquí**, sino en `ups_rev5_protocol_runner_ACCEL.py`.

---

## Licencia y Contexto Académico

Este software se desarrolla como parte de un **trabajo de titulación** para la obtención del título de **Ingeniero en Ciencias de la Computación**.

Los derechos patrimoniales han sido cedidos a la **Universidad Politécnica Salesiana**, conforme al certificado de cesión del **15 de enero de 2026**.

---

## Referencia Académica

> Montaleza Ortiz, L. D., & Romero Martínez, A. A. (2026).
> *Mejora estructural de imágenes de radiodiagnóstico, mediante un pipeline secuencial CLAHE – Filtro Guiado – Unsharp en entornos con recursos computacionales limitados*.
> Universidad Politécnica Salesiana.

---

Si deseas, en el siguiente paso puedo:

* Ajustar el README para **repositorio público vs privado**
* Crear una **versión corta para reviewers**
* Generar el **`CITATION.cff`**
* Alinear exactamente los nombres con las **secciones del artículo**

---

**Nota:** Este README documenta la implementación práctica del pipeline propuesto. Los resultados experimentales, métricas de evaluación y análisis comparativos se reportan en el artículo académico completo.
