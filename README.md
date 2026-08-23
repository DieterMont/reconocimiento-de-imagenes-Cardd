# Proyecto de tesis CarDD

Proyecto reproducible para la investigación:

**"Evaluación comparativa de arquitecturas de deep learning para la identificación y delimitación de daños vehiculares en imágenes del dataset CarDD".**

El repositorio está organizado para seguir un flujo tipo CRISP-DM complementado con CRISP-ML(Q), con trazabilidad experimental en MLflow, un notebook principal como eje del trabajo y una app web de demostración preparada para ejecución local o en Docker.

## Estado actual del workspace

- Se creó la estructura base del proyecto.
- Se dejó preparado el notebook principal `notebooks/tesis_carDD_pipeline.ipynb`.
- Se dejó una app de demostración en `app/app.py`.
- Se preparó Docker en `docker/`.
- Se documentó el flujo metodológico y de ejecución en `docs/`.
- El dataset CarDD **no está presente** en este workspace.
- El problema de PyTorch detectado no era del driver: había mezcla entre el Python global y el entorno del proyecto. El entorno definitivo del proyecto es `.venv`, con build CUDA funcional para notebook, MLflow y app.

## Estructura

```text
.
|-- app/
|   |-- app.py
|   `-- runtime.py
|-- artifacts/
|   |-- figures/
|   |-- metrics/
|   |-- models/
|   |-- predictions/
|   `-- preprocess/
|-- data/
|   |-- interim/
|   |-- processed/
|   `-- raw/
|-- docker/
|   |-- Dockerfile
|   `-- docker-compose.yml
|-- docs/
|   |-- README.md
|   |-- descripcion_metodologica.md
|   |-- guia_demostracion.md
|   `-- guia_ejecucion.md
|-- mlruns/
|-- notebooks/
|   `-- tesis_carDD_pipeline.ipynb
|-- scripts/
|   `-- build_notebook.py
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Requisitos

- Python 3.11
- Recomendado: entorno virtual dedicado
- Recomendado para entrenamiento: PyTorch con CUDA habilitado para la GPU RTX 2050

## Instalación

1. Cree y active un entorno virtual.
2. Instale PyTorch con CUDA dentro de `.venv`.
3. Instale las dependencias restantes usando el mismo intérprete:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Puede validar que está usando el entorno correcto con:

```powershell
.\.venv\Scripts\python.exe scripts/check_torch_env.py
```

Si Jupyter o VS Code muestran errores de DLL de `torch`, use el lanzador limpio:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_jupyter.ps1
```

Ese lanzador también activa `JUPYTER_ALLOW_INSECURE_WRITES=true` para evitar errores de permisos ACL sobre archivos runtime en Windows.
Además arranca Jupyter en segundo plano para no bloquear la terminal.

Para detener los procesos de Jupyter del proyecto:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_jupyter.ps1
```

## Ejecución del notebook

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_jupyter.ps1
```

Abra `notebooks/tesis_carDD_pipeline.ipynb`.
Seleccione el kernel `Python (.venv) - Tesis CarDD`.
Ese kernel se registra con un identificador interno único del proyecto para evitar colisiones con kernels viejos.

## Ejecución de MLflow

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri ./mlruns
```

## Ejecución de la app

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_streamlit.ps1
```

Si aparece un error de DLL relacionado con `torch`, abra la app con:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_streamlit.ps1
```

## Ejecución con Docker

```powershell
docker compose -f docker/docker-compose.yml up --build
```

## Documentación complementaria

- `docs/README.md`
- `docs/guia_ejecucion.md`
- `docs/guia_demostracion.md`
- `docs/descripcion_metodologica.md`
