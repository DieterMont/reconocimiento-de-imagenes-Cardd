from __future__ import annotations

import site
import sys
from pathlib import Path

from env_paths import resolve_project_env_dir, resolve_project_python


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    venv_dir = resolve_project_env_dir(project_root)
    venv_python = resolve_project_python(project_root)

    print("Python activo :", sys.executable)
    print("Python esperado:", venv_python)
    print("Entorno activo :", venv_dir)
    print("sys.prefix    :", sys.prefix)
    print("user site     :", site.getusersitepackages())

    try:
        import torch
    except Exception as exc:
        print("torch         : ERROR", repr(exc))
        return 1

    print("torch         :", torch.__version__)
    print("torch path    :", torch.__file__)
    print("CUDA build    :", torch.version.cuda)
    print("CUDA compilado:", torch.backends.cuda.is_built())
    print("CUDA activo   :", torch.cuda.is_available())
    print("GPU count     :", torch.cuda.device_count())

    using_project_venv = Path(sys.executable).resolve() == venv_python.resolve()
    if not using_project_venv:
        print(
            "[ERROR] El proyecto no está usando el entorno local esperado. "
            f"Ejecute con {venv_python}."
        )
        return 2

    if not torch.backends.cuda.is_built():
        print(
            "[ERROR] El torch activo es CPU-only. Instale una build CUDA de PyTorch "
            "dentro del entorno del proyecto."
        )
        return 3

    if not torch.cuda.is_available():
        print(
            "[WARN] Torch fue compilado con CUDA pero no pudo activar la GPU. "
            "Revise driver, compatibilidad y que el proceso use el entorno correcto."
        )
        return 4

    print("[OK] Entorno PyTorch/CUDA listo para entrenamiento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
