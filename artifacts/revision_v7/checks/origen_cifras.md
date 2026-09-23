# Origen de cifras dudosas (T1.2) — revisión V7, 20/09/2026

**0.5662 (Mask R-CNN, mAP@0.5 de validación).** Sale del run MLflow `c263d286de7b4975b81cd2a5c41ba6f9` (`mask_rcnn_segmentation`, experimento 520322221845217915): valor de `val_map50` en la época 8 (`metrics/val_map50`, línea 8; también en `artifacts/mask_rcnn_training_history.csv` de ese run). El notebook lo llama "el mejor run limpio de Mask R-CNN" (celda 46).

**0.5724 (Mask R-CNN con SWA).** Sale de la salida impresa de la celda 47 del notebook: "SWA: promedio de 8 épocas (desde la 11) -> val_mAP50=0.5724 (mejor checkpoint individual: 0.5649)". No aparece en ninguna métrica de MLflow ni en `artifacts/metrics/mask_rcnn_training_history.csv`, donde el máximo de val_mAP50 es 0.5649 en la época 6.

**Consecuencias para la tesis.**
1. 0.5662 y 0.5724 provienen de corridas y momentos distintos. Comparar "de 0.5662 a 0.5724" mezcla un checkpoint de un run anterior con un promedio SWA de otro.
2. El historial guardado en `artifacts/metrics/` (máximo 0.5649, época 6) no es el del run c263d286 (máximo 0.5662, época 8). Hay al menos dos corridas de Mask R-CNN.
3. No se puede afirmar, con los archivos disponibles, que el modelo evaluado en prueba sea el promedio SWA. Se debe confirmar con el autor cuál checkpoint se guardó en `artifacts/models/mask_rcnn.pth` (celda 47 o posterior). Hasta entonces el texto debe decir "en la salida del notebook" y no atribuir la mejora al modelo final.

**"Cinco técnicas adicionales" (Sección 12/5.5).** Las cinco están en el texto de la V6 (anclas K-Means, box_positive_fraction, SWA, hard-negative mining, recalibración de umbral). Se confirman en la configuración guardada (`anchor_sizes`, `box_positive_fraction=0.5`, `swa_enabled=True`, `hard_negative_oversample=3.0`, `conf_threshold=0.4`). El hard-negative mining figura como descartado en el texto pero la configuración final conserva `hard_negative_oversample=3.0` con `hard_negative_start_epoch=0`; confirmar si estuvo activo en el entrenamiento final.

**Historial de YOLO.** `results.csv` de ambos runs de Ultralytics contiene cuatro corridas concatenadas (reinicios de época en las filas 50, 150 y 170 en detección; 50, 170 y 190 en segmentación). La corrida final es la última (100 épocas): mejor época 98 (mAP50-95 caja de validación 0.5187) en detección y 97 (0.5675) en segmentación. La tesis debe citar solo esa última corrida.
