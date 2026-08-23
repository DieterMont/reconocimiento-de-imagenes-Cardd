$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvCandidates = @(".venv")
$venvPythons = @()
$venvPaths = @()
$killed = @()

foreach ($candidateName in $venvCandidates) {
    $candidatePython = Join-Path $projectRoot "$candidateName\Scripts\python.exe"
    if (Test-Path $candidatePython) {
        $venvPythons += $candidatePython
        $venvPaths += (Split-Path -Parent (Split-Path -Parent $candidatePython))
    }
}

$jupyterProcesses = Get-Process jupyter-lab, jupyter-notebook, jupyter-server -ErrorAction SilentlyContinue
foreach ($process in $jupyterProcesses) {
    try {
        Stop-Process -Id $process.Id -Force
        $killed += "$($process.ProcessName)#$($process.Id)"
    } catch {
    }
}

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
foreach ($process in $pythonProcesses) {
    try {
        if ($process.Path -and ($venvPythons -icontains $process.Path)) {
            Stop-Process -Id $process.Id -Force
            $killed += "python#$($process.Id)"
        }
    } catch {
    }
}

if ($killed.Count -eq 0) {
    Write-Host "No se encontraron procesos de Jupyter activos del proyecto."
} else {
    Write-Host "Procesos detenidos:"
    $killed | ForEach-Object { Write-Host " - $_" }
}
