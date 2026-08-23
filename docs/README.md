# Índice de documentación

Este directorio concentra la documentación operativa y metodológica del proyecto de tesis sobre CarDD.

## Documentos

- `guia_ejecucion.md`: pasos para montar el entorno, preparar el dataset, correr el notebook, MLflow y la app.
- `guia_demostracion.md`: guía para usar la app y explicar resultados durante una defensa o demostración.
- `descripcion_metodologica.md`: adaptación de CRISP-DM y CRISP-ML(Q), trazabilidad experimental y criterios de comparación.

## Observaciones

- El repositorio fue preparado para un entorno Windows 11 con GPU NVIDIA RTX 2050.
- Si no hay soporte CUDA en PyTorch, el notebook y la app muestran explícitamente que se está usando CPU.
- Si el dataset CarDD no está disponible localmente, el notebook no inventa estructura ni resultados: documenta el bloqueo y detiene solo las fases dependientes de datos.

