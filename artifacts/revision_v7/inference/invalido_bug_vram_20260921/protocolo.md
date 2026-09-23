# Protocolo de medicion de inferencia (revision V7)
Fecha: 2026-09-21T04:07:42
- Imagenes: 374 del conjunto de prueba de CarDD, lote 1, mismo orden en los cuatro modelos.
- Modo: model.eval(), torch.inference_mode(), FP32 (AMP solo si se pidio --amp, dato aparte).
- Calentamiento: 20 inferencias por modelo y configuracion antes de medir.
- Sincronizacion: torch.cuda.synchronize() antes de iniciar y al terminar cada medicion; reloj time.perf_counter().
- Repeticiones: 3 pasadas completas por frontera; se resume media, desviacion estandar, mediana, p95 y FPS sobre todas las medidas.
- Frontera A (solo modelo): imagen ya en GPU; incluye pre y postproceso internos de cada modelo (redimension/normalizacion en torchvision; NMS y mascaras).
  Excluye lectura de disco, decodificacion JPEG, copia CPU->GPU y conversion a numpy. YOLO recibe el tensor ya con letterbox 640x640 (fuera de la medicion); torchvision redimensiona dentro del modelo: asimetria declarada.
- Frontera B (punta a punta): desde el archivo hasta cajas, puntajes y mascaras en CPU como numpy.
- Umbrales: "comun" conf 0.25 y NMS 0.5 en los cuatro; "operativo" 0.25 en YOLO y 0.40 en R-CNN (score_thresh del modelo).
- Intervalos: bootstrap percentil 95 % sobre imagenes (2000 remuestras, semilla 20260920).
- Entorno: ver entorno.txt (antes y despues de medir).
