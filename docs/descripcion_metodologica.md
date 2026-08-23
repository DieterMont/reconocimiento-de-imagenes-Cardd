# Descripción metodológica

## Enfoque general

El proyecto sigue una adaptación pragmática de **CRISP-DM** complementada con principios de **CRISP-ML(Q)** para mejorar trazabilidad, reproducibilidad y control de calidad experimental.

## Aplicación de CRISP-DM

### 1. Comprensión del problema

Se plantea una comparación metodológicamente consistente entre arquitecturas de detección y segmentación aplicadas a daños vehiculares en imágenes del dataset CarDD.

### 2. Comprensión de los datos

El notebook inspecciona estructura del dataset, anotaciones, clases, resolución de imágenes, distribución de instancias y posibles problemas de calidad.

### 3. Preparación de datos

Se validan archivos, se limpian anotaciones inválidas, se documentan exclusiones, se generan artefactos persistentes de preprocessing y se exportan formatos compatibles para detección y segmentación.

### 4. Modelado

Se organiza el entrenamiento por bloques:

- Detección: YOLOv8-det, Faster R-CNN y RT-DETR
- Segmentación: YOLOv8-seg, Mask R-CNN y Mask2Former

Cada modelo tiene una subsección propia de hiperparametrización previa al entrenamiento.

### 5. Evaluación

Se comparan primero modelos dentro del mismo bloque y después se realiza una interpretación global por familia arquitectónica.

### 6. Despliegue

Se deja una app web local para demostración y una configuración Docker orientada a reproducibilidad.

## Aplicación de CRISP-ML(Q)

El proyecto incorpora controles explícitos sobre:

- Calidad del dataset y de las anotaciones
- Reproducibilidad mediante semillas y persistencia de configuración
- Observabilidad experimental mediante MLflow
- Trazabilidad de artefactos
- Declaración explícita de limitaciones del entorno

## Uso de MLflow

Cada corrida debe registrar:

- Nombre del modelo
- Tipo de tarea
- Hiperparámetros
- Métricas de entrenamiento y validación
- Artefactos exportados
- Ruta estable al peso final

## Organización de la comparación

- Comparación intra-bloque de detección
- Comparación intra-bloque de segmentación
- Interpretación global por familia arquitectónica
- Análisis conjunto de desempeño, robustez y costo de inferencia

## Limitaciones controladas

Durante la preparación inicial del proyecto se detectó:

- Ausencia del dataset CarDD en el workspace
- Instalación actual de PyTorch sin CUDA activa

Ambas condiciones quedan documentadas para evitar ejecuciones engañosas o resultados no reproducibles.

