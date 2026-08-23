$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvCandidates = @(".venv")
$venvDir = $null
$venvPython = $null

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

Remove-Item Env:PYTHON_HOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

$env:PYTHONNOUSERSITE = "1"
$env:PROJECT_VENV_DIR = $venvDir
$env:PATH = "$venvDir\Lib\site-packages\torch\lib;$venvDir\Scripts;$env:PATH"

& $venvPython -B "$projectRoot\scripts\check_torch_env.py"
if ($LASTEXITCODE -ne 0) {
    throw "La validación de torch/CUDA falló. Revise el entorno antes de abrir Streamlit."
}

& $venvPython -m streamlit run "$projectRoot\app\app.py"
