# Procesador de Imágenes Médicas - Pipeline CLAHE + Filtro Guiado + Unsharp Mask

## Información del Proyecto

**Universidad:** Universidad Politécnica Salesiana  
**Carrera:** Ingeniería en Computación  
**Grupo:** 67

### Miembros del Equipo
- Andre Alessandro Romero Martínez
- Daniel Luis Montaleza Ortiz

---

## Descripción General

Este proyecto forma parte del desarrollo del artículo académico *"Mejora estructural de imágenes de radiodiagnóstico, mediante un pipeline secuencial CLAHE - Filtro Guiado - Unsharp en entornos con recursos computacionales limitados"*.

La aplicación implementa un pipeline determinista de procesamiento de imágenes médicas ejecutable en CPU, diseñado para mejorar la calidad visual de imágenes de radiodiagnóstico (radiografías, mamografías, ecografías) sin requerir hardware especializado.

---

## Características Principales

### Pipeline Secuencial de Tres Etapas

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Realza el contraste local de la imagen
   - Parámetros ajustables: Clip Limit y Tile Size
   - Mejora la visibilidad de estructuras anatómicas

2. **Filtro Guiado (Guided Filter)**
   - Suaviza la imagen preservando bordes importantes
   - Parámetros ajustables: Radio y Epsilon
   - Reduce ruido sin perder detalles diagnósticos

3. **Máscara de Enfoque (Unsharp Mask)**
   - Realza bordes y detalles finos
   - Parámetros ajustables: Amount, Radio (Sigma) y Threshold
   - Mejora la nitidez de estructuras clínicas

### Interfaz Gráfica de Usuario

- **Diseño moderno** con CustomTkinter (tema oscuro)
- **Panel de control lateral** con controles deslizantes para todos los parámetros
- **Visualización lado a lado** de imagen original vs procesada
- **Funcionalidad de zoom** con rueda del mouse y botones dedicados
- **Métricas en tiempo real**: CPU, RAM, GPU, tiempo de procesamiento
- **Soporte multi-formato**: JPG, PNG, DICOM

---

## Requisitos del Sistema

### Dependencias Python

```bash
pip install opencv-python
pip install numpy
pip install Pillow
pip install customtkinter
pip install psutil
pip install pydicom
pip install pynvml  # Opcional, para monitoreo GPU NVIDIA
```

### Requisitos Mínimos de Hardware

- **CPU:** Procesador de doble núcleo o superior
- **RAM:** 4 GB (8 GB recomendado)
- **GPU:** No requerida (opcional para monitoreo)
- **SO:** Windows, Linux o macOS con Python 3.7+

---

## Estructura del Código

### Clase Principal: `MedicalImageProcessor`

#### Inicialización (`__init__`)

Define los parámetros iniciales del pipeline basados en la investigación:

- **CLAHE:** clipLimit=2, tileSize=8×8
- **Filtro Guiado:** radius=3, epsilon=1e-2
- **Unsharp Mask:** amount=0.6, sigma=0.8, threshold=0.0

Estos valores están alineados con las grillas discretas definidas en la metodología del artículo.

#### Métodos Principales

**`load_image()`**
- Carga imágenes estándar (JPG/PNG) o archivos DICOM
- Manejo multihilo para archivos DICOM pesados

**`load_dicom()`**
- Procesa archivos DICOM con pydicom
- Aplica VOI LUT (Value of Interest Look-Up Table)
- Normaliza intensidades a rango [0, 255]
- Maneja fotometría MONOCHROME1/MONOCHROME2
- Extrae metadatos clínicos relevantes

**`process_image()`**
- Aplica el pipeline secuencial completo:
  1. Conversión a espacio LAB para CLAHE
  2. Aplicación de CLAHE en canal L (luminancia)
  3. Filtro Guiado para suavizado preservando bordes
  4. Máscara de Enfoque para realce de detalles
- Ejecuta en hilo separado para mantener UI responsiva
- Calcula métricas de rendimiento en tiempo real

**`get_system_metrics()`**
- Monitorea uso de CPU (%)
- Registra consumo de RAM (MB)
- Detecta uso de GPU NVIDIA si está disponible
- Cuenta núcleos físicos/lógicos e hilos del proceso

**`display_image()`**
- Renderiza imágenes con zoom adaptativo
- Centra imágenes en canvas
- Maneja redimensionamiento con interpolación LANCZOS

---

## Grillas de Parámetros (Basadas en Literatura)

Todas las grillas están fundamentadas en la revisión de literatura del artículo:

### CLAHE
- **clipLimit:** {1, 2, 3, 4, 6, 8, 12, 16}
- **tileGridSize:** {(4,4), (6,6), (8,8), (12,12), (16,16), (24,24), (32,32)}

### Filtro Guiado
- **radius:** {2, 3, 4, 6, 8, 12, 16, 24, 32}
- **epsilon:** {1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}

### Unsharp Mask
- **amount:** {0.1, 0.2, 0.3, 0.6, 1.0, 1.6, 2.0, 2.2}
- **sigma:** {0.5, 0.8, 1.2, 1.8, 2.5, 3.0, 3.5}
- **threshold:** {0.0, 0.005, 0.01, 0.02, 0.05, 0.1}

---

## Uso de la Aplicación

### Ejecución

```bash
python medical_processor_gui.py
```

### Flujo de Trabajo

1. **Cargar imagen:** Botón "Cargar imagen" → seleccionar archivo
2. **Ajustar parámetros:** Usar sliders para modificar θ_C, θ_G, θ_U
3. **Observar resultado:** Visualización automática de imagen procesada
4. **Zoom:** Rueda del mouse o botones +/- para inspeccionar detalles
5. **Resetear:** Botón "Resetear Parámetros" restaura valores iniciales
6. **Descargar:** Botón "Descargar Imagen" guarda resultado procesado

### Métricas Mostradas

- **Tiempo de procesamiento** (segundos)
- **Uso de CPU** (%)
- **Núcleos:** Físicos/Lógicos
- **Hilos del proceso** (incluyendo GUI)
- **Consumo de RAM** (MB)
- **Uso de GPU** (MB, si disponible)

---

## Procesamiento de Imágenes DICOM

### Metadatos Extraídos

- Nombre del paciente
- Fecha del estudio
- Modalidad (RX, CT, MR, etc.)
- Dimensiones (filas × columnas)
- Window Center/Width
- Interpretación fotométrica
- Bits asignados
- Transfer Syntax UID

### Normalización de Intensidades

```python
# Aplicación de ventana (windowing)
min_val = windowCenter - windowWidth / 2
max_val = windowCenter + windowWidth / 2

# Normalización a [0, 255]
pixel_array = ((pixel_array - min_val) / (max_val - min_val)) * 255
```

### Manejo de Fotometría

- **MONOCHROME2:** Valores altos = brillante (estándar)
- **MONOCHROME1:** Valores altos = oscuro → se invierte

---

## Arquitectura de Procesamiento

### Pipeline Matemático

**Etapa 1 - CLAHE:**
```
I_rgb → I_lab → CLAHE(L, clipLimit, tileSize) → I_clahe
```

**Etapa 2 - Filtro Guiado:**
```
I_clahe → GuidedFilter(I_clahe, radius, ε) → I_filtered
```

**Etapa 3 - Unsharp Mask:**
```
I_blurred = GaussianBlur(I_filtered, σ)
I_sharpened = I_filtered + α * (I_filtered - I_blurred) if |diff| > τ_u
```

### Optimización de Rendimiento

- **Procesamiento asíncrono:** Threading para operaciones pesadas
- **Debouncing:** Delay de 300ms en sliders para evitar procesamiento excesivo
- **Normalización eficiente:** Operaciones vectorizadas con NumPy
- **Memoria controlada:** Liberación explícita de imágenes intermedias

---

## Relación con el Artículo Académico

Este código implementa:

- **Sección 3.1:** Pipeline secuencial determinista (θ* = θ_C, θ_G, θ_U)
- **Tabla 2:** Espacio de hiperparámetros con grillas discretas
- **Figura 2:** Flujo de procesamiento completo
- **Bloque D (Tabla 3):** Métricas de costo computacional (tiempo, RAM, CPU)

El panel de control permite:
- Exploración manual del espacio θ
- Validación visual de configuraciones candidatas
- Análisis de trade-offs contraste-ruido-nitidez
- Verificación de viabilidad en CPU sin GPU

---

## Limitaciones y Consideraciones

### Almacenamiento Persistente

⚠️ **ADVERTENCIA:** Este código **NO utiliza** `localStorage` ni `sessionStorage` (APIs de navegador no disponibles en aplicaciones de escritorio).

Toda la información se mantiene en memoria durante la sesión. Si se requiere persistencia:
- Implementar guardado/carga de configuraciones en archivos JSON
- Usar bases de datos locales (SQLite)
- Serializar estado con Pickle

### Manejo de Errores DICOM

Algunos archivos DICOM pueden fallar por:
- Transfer Syntax no soportado por pydicom
- Pixel data comprimido (JPEG 2000, RLE)
- Metadatos faltantes o corruptos

Solución: Usar `force=True` en `pydicom.dcmread()` y manejo de excepciones robusto.

### Rendimiento en Imágenes Grandes

Para imágenes >4MP (p.ej., mamografías digitales):
- El procesamiento puede tomar 2-5 segundos en CPU modesta
- Considerar downsampling previo para preview interactivo
- La GUI permanece responsiva gracias al threading

---

## Trabajo Futuro

- [ ] Integración de optimización de hiperparámetros (Grid + GA + BO)
- [ ] Cálculo automático de 49 métricas (FR/NR/ROI)
- [ ] Comparación con los 15 pipelines del catálogo
- [ ] Evaluación MOS integrada con interfaz para radiólogos
- [ ] Exportación de reportes con métricas detalladas
- [ ] Modo batch para procesamiento de múltiples imágenes
- [ ] Implementación de gates de seguridad perceptual

---

## Contacto y Contribuciones

Este proyecto es parte de un trabajo de titulación académico. Para consultas técnicas o colaboraciones:

- **Autores:** Andre Alessandro Romero Martínez, Daniel Luis Montaleza Ortiz
- **Tutor:** Joe Frand Llerena Izquierdo
- **Institución:** Universidad Politécnica Salesiana - Sede Guayaquil
- **Año:** 2026

---

## Licencia

Este software se desarrolla con fines académicos como parte del trabajo de titulación para obtener el título de Ingeniero en Ciencias de la Computación.

Los derechos patrimoniales han sido cedidos a la Universidad Politécnica Salesiana según certificado de cesión del 15 de enero de 2026.

---

## Referencias

Para la fundamentación teórica y metodológica completa, consultar el artículo académico adjunto:

> Montaleza Ortiz, L. D., & Romero Martínez, A. A. (2026). *Mejora estructural de imágenes de radiodiagnóstico, mediante un pipeline secuencial CLAHE - Filtro Guiado - Unsharp en entornos con recursos computacionales limitados*. Universidad Politécnica Salesiana.

---

**Nota:** Este README documenta la implementación práctica del pipeline propuesto. Los resultados experimentales, métricas de evaluación y análisis comparativos se reportan en el artículo académico completo.
