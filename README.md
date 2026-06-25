# FutBotMX Vision Challenge

Este proyecto presenta una solución de visión por computadora para el análisis automático de partidos de fútbol robótico de la Copa FutBotMX. La propuesta combina SAM 3, fine-tuning específico del dominio, técnicas de seguimiento de objetos y transformación de perspectiva para segmentar, rastrear y analizar robots y balón, generando visualizaciones tácticas y métricas del juego.

## Instalación

El proyecto utiliza `uv` para la gestión de dependencias y entornos virtuales. Las dependencias locales se encuentran definidas en el archivo `pyproject.toml`.

### Crear y activar el entorno virtual

```powershell
uv venv .venv
.\.venv\Scripts\activate
```

### Instalar PyTorch con soporte CUDA

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Instalar dependencias del proyecto

```powershell
uv sync
```

## Requisitos de hardware y software

### Fine-tuning de YOLOv8n

El ajuste fino del detector se realizó localmente en una computadora portátil con la siguiente configuración:

* GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
* Memoria RAM: 16 GB
* Sistema operativo: Ubuntu 26.04 LTS
* Python: 3.12

YOLOv8n fue seleccionado por ser una arquitectura ligera que permite realizar fine-tuning en hardware de consumo sin necesidad de GPUs de gama alta. Los requerimientos de memoria dependen del tamaño del conjunto de datos, la resolución de entrada y el tamaño del lote utilizado durante el entrenamiento.

### Segmentación y análisis con SAM 3

Debido a los mayores requerimientos computacionales de SAM 3, las etapas de segmentación, tracking, transformación de perspectiva y generación de visualizaciones se ejecutaron en Google Colab utilizando aceleración por GPU.

La notebook principal del proyecto incluye todas las instrucciones de instalación y las dependencias necesarias para reproducir el pipeline completo.

## Ejecución

El proyecto está organizado en dos etapas independientes:

* `finetuning/`: entrenamiento y evaluación del detector YOLOv8n.
* `notebooks/pipeline_fine_tuning.ipynb`: segmentación con SAM 3, tracking, homografía, análisis táctico y generación de visualizaciones.

Para reproducir el pipeline completo, basta con ejecutar las notebooks incluidas en el repositorio siguiendo el orden descrito anteriormente.


## Cómo se hizo el fine-tuning

Para adaptar el detector al contexto específico de la Copa FutBotMX, se realizó un fine-tuning de YOLOv8n usando imágenes extraídas de partidos de fútbol robótico. Las imágenes fueron etiquetadas manualmente en tres clases: `robot_team_a`, `robot_team_b` y `ball`, permitiendo que el modelo aprendiera a distinguir entre robots de ambos equipos y el balón dentro del entorno real de competencia.

El dataset final contiene 116 imágenes etiquetadas, divididas en 92 imágenes para entrenamiento y 24 para validación. Se utilizó el paquete `ultralytics` con aceleración por GPU CUDA. Durante el entrenamiento se aplicaron aumentos de datos como variaciones de color, escala, traslación y volteo, con el objetivo de mejorar la robustez del detector ante cambios de iluminación, posición y orientación de los objetos.

El modelo base utilizado fue YOLOv8n, elegido por ser una arquitectura ligera y adecuada para inferencia rápida. Después del entrenamiento, se conservó el mejor checkpoint validado como `finetuning/models/YOLOv8n_robots.pt`, que es el modelo usado posteriormente para detectar robots y balón en nuevos videos.

## Pipeline de procesamiento

El pipeline se desarrolló en Google Colab y combina un detector especializado con SAM 3 para segmentar y analizar los partidos de fútbol robótico.

1. **Fine-tuning de YOLOv8n:** Se entrenó un modelo YOLOv8n con 116 imágenes etiquetadas manualmente para detectar tres clases: *equipo_a*, *equipo_b* y *balón*.

2. **Detección por fotograma:** El modelo YOLOv8n ajustado se aplica a cada frame del video para generar *bounding boxes* de robots y balón.

3. **Refinamiento con SAM 3:** Las cajas detectadas por YOLO se usan como prompts para SAM 3, obteniendo máscaras más precisas de cada objeto.

4. **Asignación de clases:** Las máscaras generadas por SAM 3 conservan la clase original de YOLO mediante asociación por IoU.

5. **Tracking temporal:** Las detecciones segmentadas se procesan con ByteTrack para mantener la identidad de los objetos entre frames y reconstruir trayectorias.

6. **Transformación de perspectiva:** Se aplica una homografía para proyectar las posiciones y máscaras detectadas a una vista cenital de la cancha.

7. **Análisis del juego:** Se registran posiciones, clases, identificadores de tracking, confianza y área de máscara en un archivo `tactical_log.csv`. También se implementa detección básica de goles a partir del cruce del balón por líneas definidas.

8. **Visualización final:** Se genera un video con tres vistas: video original, vista cenital transformada y video anotado con segmentación, tracking, marcador y mapa táctico superpuesto.

## Reel de Instagram

El reel del proyecto se encuentra disponible públicamente en el siguiente enlace:

**Instagram:** [Agregar enlace aquí]

El video resume los principales resultados del pipeline, incluyendo la segmentación, el seguimiento de objetos y las visualizaciones tácticas generadas.

## Dashboards y visualizaciones

A partir de las trayectorias y métricas extraídas del partido, se desarrollaron visualizaciones orientadas al análisis táctico del juego robótico.


## Licencia

Este proyecto se distribuye bajo la licencia MIT.

El uso de SAM 3 está sujeto a los términos y condiciones de la licencia oficial de Meta.

## Créditos

**Gerardo Macías**

* Diseño e implementación del pipeline de visión por computadora.
* Preparación y etiquetado del conjunto de datos.
* Fine-tuning del detector YOLOv8n.
