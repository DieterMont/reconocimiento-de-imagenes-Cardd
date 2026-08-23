from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = (".venv",)


def resolve_project_env_dir(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT

    override = os.getenv("PROJECT_VENV_DIR")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        if (candidate / "Scripts" / "python.exe").exists():
            return candidate

    for env_name in ENV_CANDIDATES:
        candidate = root / env_name
        if (candidate / "Scripts" / "python.exe").exists():
            return candidate

    return root / ".venv"


def resolve_project_python(project_root: Path | None = None) -> Path:
    return resolve_project_env_dir(project_root) / "Scripts" / "python.exe"
