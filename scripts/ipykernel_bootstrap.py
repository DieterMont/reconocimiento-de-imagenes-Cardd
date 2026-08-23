from __future__ import annotations

import os
import runpy
from pathlib import Path

from env_paths import resolve_project_env_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = resolve_project_env_dir(PROJECT_ROOT)
TORCH_LIB_DIR = VENV_DIR / "Lib" / "site-packages" / "torch" / "lib"
VENV_SCRIPTS_DIR = VENV_DIR / "Scripts"
JUPYTER_DIR = PROJECT_ROOT / ".jupyter"
JUPYTER_DATA_DIR = PROJECT_ROOT / ".jupyter_data"
JUPYTER_RUNTIME_DIR = JUPYTER_DATA_DIR / "runtime"
IPYTHON_DIR = PROJECT_ROOT / ".ipython"


def prepend_path(path: Path) -> None:
    if not path.exists():
        return
    current = os.environ.get("PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    path_str = str(path)
    if path_str not in entries:
        os.environ["PATH"] = os.pathsep.join([path_str] + entries)


for variable_name in ("PYTHON_HOME", "PYTHONHOME", "PYTHONPATH"):
    os.environ.pop(variable_name, None)

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "true")
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(JUPYTER_DIR))
os.environ.setdefault("JUPYTER_DATA_DIR", str(JUPYTER_DATA_DIR))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(JUPYTER_RUNTIME_DIR))
os.environ.setdefault("IPYTHONDIR", str(IPYTHON_DIR))

for path in (JUPYTER_DIR, JUPYTER_DATA_DIR, JUPYTER_RUNTIME_DIR, IPYTHON_DIR):
    path.mkdir(parents=True, exist_ok=True)

prepend_path(TORCH_LIB_DIR)
prepend_path(VENV_SCRIPTS_DIR)

if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(TORCH_LIB_DIR))
    except OSError:
        pass
    try:
        os.add_dll_directory(str(VENV_SCRIPTS_DIR))
    except OSError:
        pass

os.chdir(PROJECT_ROOT)
runpy.run_module("ipykernel_launcher", run_name="__main__")
