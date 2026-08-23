$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvCandidates = @(".venv")
$venvDir = $null
$venvPython = $null
$jupyterConfigDir = Join-Path $projectRoot ".jupyter"
$jupyterDataDir = Join-Path $projectRoot ".jupyter_data"
$jupyterRuntimeDir = Join-Path $jupyterDataDir "runtime"
$ipythonDir = Join-Path $projectRoot ".ipython"

foreach ($candidateName in $venvCandidates) {
    $candidatePython = Join-Path $projectRoot "$candidateName\Scripts\python.exe"
    if (Test-Path $candidatePython) {
        $venvPython = $candidatePython
        $venvDir = Split-Path -Parent (Split-Path -Parent $candidatePython)
        break
    }
}

if (-not $venvPython) {
    throw "No se encontró el entorno del proyecto en .venv."
}

New-Item -ItemType Directory -Force -Path $jupyterConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path $jupyterDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $jupyterRuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $ipythonDir | Out-Null

# Evita que VS Code/Jupyter hereden un runtime de Python global incompatible.
Remove-Item Env:PYTHON_HOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

$env:JUPYTER_CONFIG_DIR = $jupyterConfigDir
$env:JUPYTER_DATA_DIR = $jupyterDataDir
$env:JUPYTER_RUNTIME_DIR = $jupyterRuntimeDir
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"
$env:IPYTHONDIR = $ipythonDir
$env:PYTHONNOUSERSITE = "1"
$env:JUPYTER_PREFER_ENV_PATH = "1"
$env:JUPYTER_PATH = (Join-Path $venvDir "share\jupyter")
$env:PROJECT_VENV_DIR = $venvDir
$env:PATH = "$venvDir\Lib\site-packages\torch\lib;$venvDir\Scripts;$env:PATH"

$hasJupyterLab = & $venvPython -B -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('jupyterlab') else 1)"
if ($LASTEXITCODE -eq 0) {
    Start-Process -FilePath $venvPython `
        -ArgumentList "-m", "jupyter", "lab", "--ServerApp.open_browser=True" `
        -WorkingDirectory $projectRoot
    Write-Host "Jupyter Lab iniciado."
    exit 0
}

$hasNotebook = & $venvPython -B -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('notebook') else 1)"
if ($LASTEXITCODE -eq 0) {
    Start-Process -FilePath $venvPython `
        -ArgumentList "-m", "notebook", "--browser" `
        -WorkingDirectory $projectRoot
    Write-Host "Jupyter Notebook iniciado."
    exit 0
}

throw "No se encontró jupyterlab ni notebook en el entorno del proyecto. Instale el stack con $venvPython -m pip install -r requirements.txt"
