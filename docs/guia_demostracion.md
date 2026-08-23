# Guía de demostración

## Objetivo de la demo

Mostrar de forma clara cómo los modelos de detección y segmentación responden ante imágenes vehiculares con daños y cómo se comparan sus métricas finales.

## Flujo sugerido de demostración

1. Abra la app.
2. Muestre el dispositivo activo para inferencia.
3. Cargue una imagen vehicular.
4. Presente la tabla de métricas finales por modelo.
5. Compare visualmente resultados de detección y segmentación.
6. Destaque fortalezas y debilidades por familia arquitectónica.

## Qué explicar durante la demo

- Diferencia entre detección y segmentación.
- Por qué se comparan arquitecturas one-stage, two-stage y transformers.
- Qué métricas son apropiadas para cada bloque.
- Qué modelo logra mejor equilibrio entre precisión y costo computacional.
- Qué modelo es más conveniente para una app local de demostración.

## Señales importantes para interpretar resultados

- `mAP50` y `mAP50-95` ayudan a comparar la calidad de detección.
- `mIoU` y `Dice` ayudan a comparar la calidad de segmentación.
- El tiempo de inferencia es clave para la viabilidad en demo.
- Un modelo puede tener mejor métrica pero peor usabilidad si el costo computacional es alto.

## Si faltan modelos o métricas

- La app mostrará advertencias claras si faltan pesos, métricas o manifiestos.
- No use resultados vacíos como si fueran resultados experimentales.
- Explique que el sistema está preparado para cargar artefactos persistidos cuando el entrenamiento haya sido completado.

