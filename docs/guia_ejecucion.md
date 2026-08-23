# Guía de ejecución

## 1. Preparación del entorno

1. Cree un entorno virtual con Python 3.11.
2. Active el entorno.
3. Instale primero una versión de PyTorch compatible con CUDA dentro de `.venv`.
4. Instale el resto de dependencias usando siempre el Python de `.venv`.
5. Verifique que el `torch` activo sea el de `.venv`, no el global del sistema.

Comandos recomendados en Windows:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_torch_env.py
```

Si `check_torch_env.py` reporta que se está usando `C:\Program Files\Python311\python.exe` o un `torch` `+cpu`, no continúe con entrenamiento.
Si Jupyter muestra errores como `Error loading ... c10.dll or one of its dependencies`, evite arrancarlo desde un host contaminado por variables globales y use el lanzador limpio del proyecto.
Ese lanzador también habilita `JUPYTER_ALLOW_INSECURE_WRITES=true` para sortear errores de permisos `SetFileSecurity` en Windows.

## 2. Preparación del dataset CarDD

1. Descargue CarDD desde su fuente oficial.
2. Ubique el dataset en `data/raw/CarDD` o defina la variable `CARDD_RAW_DIR`.
3. Verifique que las imágenes y anotaciones estén accesibles desde el notebook.
4. Si el layout del dataset difiere del supuesto COCO-like, el notebook lo reportará y deberá adaptarse la etapa de normalización antes de continuar.

## 3. Generación del notebook base

Si necesita regenerar el notebook desde el script fuente:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

## 4. Ejecución del notebook

1. Inicie Jupyter Lab con `.venv`.
2. Abra `notebooks/tesis_carDD_pipeline.ipynb`.
3. Ejecute las celdas en orden.
4. Confirme en la sección inicial qué intérprete detectó el notebook y qué dispositivo detectó PyTorch.
5. No inicie entrenamientos hasta que la validación del dataset y los artefactos de preprocessing concluyan sin errores.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_jupyter.ps1
```

Alternativa recomendada si hay errores de DLL:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_jupyter.ps1
```

Ese launcher abre Jupyter en segundo plano para evitar esperas largas en la terminal.

Para cerrarlo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_jupyter.ps1
```

Una vez abierto Jupyter:

1. Abra `notebooks/tesis_carDD_pipeline.ipynb`.
2. Seleccione el kernel `Python (.venv) - Tesis CarDD`.
3. Verifique en la primera sección del notebook que `using_project_venv = True`.
Ese kernel usa un nombre interno exclusivo del proyecto para evitar colisiones con kernels antiguos.

## 5. MLflow

1. Levante la interfaz local:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri ./mlruns
```

2. Revise que cada experimento registre parámetros, métricas, artefactos y rutas a pesos.

## 6. App de demostración

1. Ejecute:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_streamlit.ps1
```

Alternativa recomendada si hay errores de DLL:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_streamlit.ps1
```

2. Cargue una imagen de prueba.
3. Revise el estado de los modelos, métricas finales y dispositivo activo.

## 7. Docker

1. Construya y ejecute:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

2. Acceda a `http://localhost:8501`.
3. Si Docker no expone GPU en Windows 11 Home, use el modo CPU como fallback y documente esa limitación en la demostración.
