$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvCandidates = @(".venv")
$venvDir = $null
$venvPython = $null
$kernelName = "tesis-cardd-project-venv"
$kernelDisplayName = "Python (.venv) - Tesis CarDD"

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

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHON_HOME -ErrorAction SilentlyContinue

$env:PYTHONNOUSERSITE = "1"
$env:PROJECT_VENV_DIR = $venvDir
$env:JUPYTER_PREFER_ENV_PATH = "1"
$env:JUPYTER_PATH = (Join-Path $venvDir "share\jupyter")
$env:PATH = "$venvDir\Lib\site-packages\torch\lib;$venvDir\Scripts;$env:PATH"

Write-Host "Instalando stack de notebook en el entorno del proyecto..."
& $venvPython -m pip install jupyterlab==4.4.0 notebook==7.4.0 ipykernel==6.29.5 ipywidgets==8.1.6
if ($LASTEXITCODE -ne 0) {
    throw "Falló la instalación del stack mínimo de notebook."
}

Write-Host "Verificando módulos de notebook..."
& $venvPython -B -c "import importlib.util; print('jupyterlab', bool(importlib.util.find_spec('jupyterlab'))); print('notebook', bool(importlib.util.find_spec('notebook')))"
if ($LASTEXITCODE -ne 0) {
    throw "Falló la verificación del stack de notebook."
}

Write-Host "Registrando kernel del proyecto..."
& $venvPython -m ipykernel install --prefix $venvDir --name $kernelName --display-name $kernelDisplayName
if ($LASTEXITCODE -ne 0) {
    throw "Falló el registro del kernel."
}

$kernelJsonPath = Join-Path $venvDir "share\jupyter\kernels\$kernelName\kernel.json"
$bootstrapPath = Join-Path $projectRoot "scripts\ipykernel_bootstrap.py"
$kernelSpec = [ordered]@{
    argv = @(
        $venvPython,
        "-Xfrozen_modules=off",
        $bootstrapPath,
        "--HistoryManager.hist_file=:memory:",
        "--HistoryManager.enabled=False",
        "-f",
        "{connection_file}"
    )
    display_name = $kernelDisplayName
    language = "python"
    metadata = @{
        debugger = $true
    }
    env = @{
        PROJECT_VENV_DIR = $venvDir
        JUPYTER_PREFER_ENV_PATH = "1"
        JUPYTER_PATH = (Join-Path $venvDir "share\jupyter")
        PYTHON_HOME = ""
        PYTHONHOME = ""
        PYTHONPATH = ""
        PYTHONNOUSERSITE = "1"
        JUPYTER_ALLOW_INSECURE_WRITES = "true"
        JUPYTER_CONFIG_DIR = (Join-Path $projectRoot ".jupyter")
        JUPYTER_DATA_DIR = (Join-Path $projectRoot ".jupyter_data")
        JUPYTER_RUNTIME_DIR = (Join-Path $projectRoot ".jupyter_data\runtime")
        IPYTHONDIR = (Join-Path $projectRoot ".ipython")
    }
}

$kernelJson = $kernelSpec | ConvertTo-Json -Depth 5
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($kernelJsonPath, $kernelJson, $utf8NoBom)

Write-Host "Validando entorno torch/CUDA..."
& $venvPython -B "$projectRoot\scripts\check_torch_env.py"
if ($LASTEXITCODE -ne 0) {
    throw "La validación final de torch/CUDA falló."
}

Write-Host "Stack de notebook listo."
